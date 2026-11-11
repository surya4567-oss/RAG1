"""Evidence-grounded answer generator using Groq and strict citation validation."""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.llm_factory import get_chat_llm
from src.retrieval.prompts import (
    ANSWER_GENERATION_SYSTEM_PROMPT,
    ANSWER_GENERATION_USER_PROMPT,
)
from src.retrieval.schemas import (
    CandidateChunk,
    Evidence,
    RetrievalResponse,
    RetrievalRoute,
    RetrievalStatus,
)


_STOPWORDS = {
    "the", "and", "is", "in", "to", "of", "for", "with", "a", "an", "on", "at",
    "by", "this", "that", "from", "as", "are", "was", "were", "be", "been",
    "has", "have", "had", "it", "its", "or", "which", "not", "but",
}


def _normalize_identifier(text: str) -> str:
    """Normalize string by removing separators and lowercase for fuzzy matching."""
    return re.sub(r"[\s\-_\.:/]+", "", text).lower()


class EvidenceGroundedAnswerGenerator:
    """Generates strictly evidence-grounded answers and validates citations against candidate chunks."""

    def __init__(self, config: Optional[RetrievalConfig] = None, llm: Optional[BaseChatModel] = None):
        self.config = config or default_retrieval_config
        self._llm = llm

    def _get_llm(self) -> Optional[BaseChatModel]:
        if self._llm is not None:
            return self._llm

        self._llm = get_chat_llm(
            config=self.config,
            role="generator",
            max_tokens=1024,
            json_mode=True,
        )
        if self._llm is None:
            logger.warning("Answer Generator LLM could not be initialized.")
        return self._llm

    def _format_evidence_context(self, candidates: List[CandidateChunk]) -> str:
        """Format candidate chunks into structured text blocks with complete available metadata."""
        if not candidates:
            return "No evidence chunks retrieved."

        blocks = []
        for i, c in enumerate(candidates, start=1):
            meta = c.metadata or {}
            doc_type = meta.get("document_type", "unknown")
            rec_id = (
                meta.get("record_id")
                or meta.get("test_case_id")
                or meta.get("defect_id")
                or meta.get("execution_id")
                or "N/A"
            )
            src_file = (
                meta.get("source_file")
                or meta.get("screenshot_file")
                or meta.get("log_file")
                or meta.get("source_path")
                or meta.get("source")
                or "N/A"
            )

            # Construct structured header with reference index and verified metadata
            meta_lines = [
                f"--- EVIDENCE CHUNK {i} (REFERENCE: CHUNK_{i}) ---",
                f"CHUNK_ID: {c.chunk_id}",
                f"DOCUMENT_TYPE: {doc_type}",
                f"RECORD_ID: {rec_id}",
                f"SOURCE_FILE: {src_file}",
            ]

            # Cross-referencing & relationship identifiers
            if meta.get("defect_id"):
                meta_lines.append(f"DEFECT_ID: {meta['defect_id']}")
            if meta.get("execution_id"):
                meta_lines.append(f"EXECUTION_ID: {meta['execution_id']}")
            if meta.get("test_case_id"):
                meta_lines.append(f"TEST_CASE_ID: {meta['test_case_id']}")
            if meta.get("requirement_id"):
                meta_lines.append(f"REQUIREMENT_ID: {meta['requirement_id']}")

            # Assignment & role identifiers
            if meta.get("assigned_to"):
                meta_lines.append(f"ASSIGNED_TO: {meta['assigned_to']}")
            if meta.get("developer"):
                meta_lines.append(f"DEVELOPER: {meta['developer']}")
            if meta.get("reported_by"):
                meta_lines.append(f"REPORTED_BY: {meta['reported_by']}")
            if meta.get("tester"):
                meta_lines.append(f"TESTER: {meta['tester']}")
            if meta.get("automation_framework"):
                meta_lines.append(f"AUTOMATION_FRAMEWORK: {meta['automation_framework']}")

            # Modality-specific attributes
            if meta.get("screen_name"):
                meta_lines.append(f"SCREEN_NAME: {meta['screen_name']}")
            if meta.get("status"):
                meta_lines.append(f"STATUS: {meta['status']}")
            if meta.get("module"):
                meta_lines.append(f"MODULE: {meta['module']}")
            if meta.get("priority"):
                meta_lines.append(f"PRIORITY: {meta['priority']}")
            if meta.get("severity"):
                meta_lines.append(f"SEVERITY: {meta['severity']}")
            if meta.get("build_number"):
                meta_lines.append(f"BUILD_NUMBER: {meta['build_number']}")
            if meta.get("error_code"):
                meta_lines.append(f"ERROR_CODE: {meta['error_code']}")
            if meta.get("exception_type"):
                meta_lines.append(f"EXCEPTION_TYPE: {meta['exception_type']}")
            if meta.get("http_status"):
                meta_lines.append(f"HTTP_STATUS: {meta['http_status']}")
            if meta.get("has_error_message") is not None:
                meta_lines.append(f"HAS_ERROR_MESSAGE: {meta['has_error_message']}")

            meta_lines.append(f"CONTENT:\n{c.text}\n")
            blocks.append("\n".join(meta_lines))

        return "\n".join(blocks)

    def _resolve_citation_chunk(
        self,
        raw_cid: str,
        valid_chunk_map: Dict[str, CandidateChunk],
        candidates: List[CandidateChunk],
    ) -> Optional[Tuple[str, CandidateChunk]]:
        """Map raw citation identifier back to canonical CandidateChunk.

        Handles:
        1. Exact match on chunk_id
        2. Reference indices (e.g. '[CHUNK_1]', 'CHUNK_1', '1')
        3. Normalized / fuzzy matches (e.g. missing 'batch_' in OCR chunk names)
        4. Metadata identifiers (record_id, defect_id, test_case_id, etc.)
        """
        clean_cid = raw_cid.strip().strip("[]'\"")
        if not clean_cid:
            return None

        # 1. Exact match
        if clean_cid in valid_chunk_map:
            return clean_cid, valid_chunk_map[clean_cid]

        # 2. Chunk reference tag: e.g. CHUNK_1, CHUNK-1, CHUNK 1, or bare integer
        idx_match = re.match(r"^(?:CHUNK[_\s-]?)?(\d+)$", clean_cid, re.IGNORECASE)
        if idx_match:
            try:
                idx = int(idx_match.group(1))
                if 1 <= idx <= len(candidates):
                    target = candidates[idx - 1]
                    return target.chunk_id, target
            except (ValueError, IndexError):
                pass

        # 3. Normalized string matching (handles 'ocr_003' vs 'ocr_batch_003')
        norm_cid = _normalize_identifier(clean_cid)
        # Also strip 'batch' from normalized citation if model omitted it
        norm_cid_nobatch = norm_cid.replace("batch", "")

        for cand_id, cand in valid_chunk_map.items():
            norm_cand_id = _normalize_identifier(cand_id)
            norm_cand_nobatch = norm_cand_id.replace("batch", "")
            if norm_cid == norm_cand_id or norm_cid_nobatch == norm_cand_nobatch:
                return cand.chunk_id, cand

        # 4. Substring matching
        for cand_id, cand in valid_chunk_map.items():
            if clean_cid in cand_id or cand_id in clean_cid:
                return cand.chunk_id, cand

        # 5. Metadata field matching (record_id, document_id, screenshot_id, etc.)
        for cand_id, cand in valid_chunk_map.items():
            meta = cand.metadata or {}
            match_candidates = [
                meta.get("record_id"),
                meta.get("document_id"),
                meta.get("screenshot_id"),
                meta.get("test_case_id"),
                meta.get("defect_id"),
                meta.get("execution_id"),
            ]
            for mval in match_candidates:
                if mval and str(mval).strip() == clean_cid:
                    return cand.chunk_id, cand

        return None

    def _verify_factual_support(
        self,
        supporting_text: str,
        chunk_text: str,
        answer_text: str,
    ) -> Tuple[bool, float, str]:
        """Verify whether chunk text factually supports the claim or supporting excerpt.

        Returns:
            Tuple of (is_supported: bool, confidence: float, verified_supporting_text: str)
        """
        c_text = chunk_text or ""
        s_text = (supporting_text or "").strip()

        # If supporting_text is a verbatim substring of chunk text -> high confidence
        if s_text and len(s_text) >= 15 and s_text in c_text:
            return True, 1.0, s_text

        # Tokenize supporting_text and chunk_text
        s_tokens = [w for w in re.findall(r"\w+", s_text.lower()) if len(w) >= 3 and w not in _STOPWORDS]
        c_tokens = set(re.findall(r"\w+", c_text.lower()))

        if s_tokens:
            overlap = sum(1 for w in s_tokens if w in c_tokens)
            ratio = overlap / len(s_tokens)

            if ratio >= 0.5:
                # Good lexical grounding; locate best matching sentence in chunk_text if possible
                sentences = re.split(r"(?<=[.!?\n])\s+", c_text)
                best_sentence = s_text
                best_score = 0
                for sent in sentences:
                    sent_tokens = set(re.findall(r"\w+", sent.lower()))
                    match_count = sum(1 for w in s_tokens if w in sent_tokens)
                    if match_count > best_score:
                        best_score = match_count
                        best_sentence = sent.strip()

                return True, round(ratio, 2), (best_sentence if best_score >= 2 else s_text)

        # Check overlap between answer key terms and chunk text
        ans_tokens = [w for w in re.findall(r"\w+", answer_text.lower()) if len(w) >= 4 and w not in _STOPWORDS]
        if ans_tokens:
            ans_overlap = sum(1 for w in ans_tokens if w in c_tokens)
            ans_ratio = ans_overlap / len(ans_tokens)
            if ans_ratio >= 0.2:
                # The chunk has key topical terms from the answer
                excerpt = s_text if s_text else c_text[:250].strip()
                return True, round(min(ans_ratio + 0.3, 0.85), 2), excerpt

        # Zero or trivial overlap
        return False, 0.0, s_text if s_text else c_text[:200]

    def _validate_evidence_citations(
        self,
        raw_evidence: List[Dict[str, Any]],
        valid_chunk_map: Dict[str, CandidateChunk],
        candidates: Optional[List[CandidateChunk]] = None,
        answer_text: str = "",
    ) -> List[Evidence]:
        """Verify that every citation matches an actual retrieved chunk and supports the claim."""
        validated: List[Evidence] = []
        seen_canonical_chunks: Set[str] = set()
        active_candidates = candidates if candidates is not None else list(valid_chunk_map.values())

        for item in raw_evidence:
            if not isinstance(item, dict):
                continue

            raw_cids = item.get("chunk_id", "")
            if isinstance(raw_cids, list):
                cid_list = [str(x).strip() for x in raw_cids if str(x).strip()]
            else:
                cid_str = str(raw_cids).strip()
                cid_list = [x.strip() for x in cid_str.split(",") if x.strip()] if "," in cid_str else [cid_str]

            for raw_cid in cid_list:
                resolved = self._resolve_citation_chunk(raw_cid, valid_chunk_map, active_candidates)
                if not resolved:
                    logger.warning(f"Rejected citation: chunk_id '{raw_cid}' cannot be resolved to any retrieved chunk.")
                    continue

                canonical_cid, chunk = resolved
                if canonical_cid in seen_canonical_chunks:
                    continue
                seen_canonical_chunks.add(canonical_cid)

                supporting_text = str(item.get("supporting_text", "")).strip()
                is_supported, conf, verified_text = self._verify_factual_support(
                    supporting_text=supporting_text,
                    chunk_text=chunk.text,
                    answer_text=answer_text,
                )

                meta = chunk.metadata or {}
                validated.append(
                    Evidence(
                        chunk_id=canonical_cid,
                        document_type=str(item.get("document_type") or meta.get("document_type") or ""),
                        source_file=str(
                            item.get("source_file")
                            or meta.get("source_file")
                            or meta.get("source_path")
                            or meta.get("source")
                            or ""
                        ),
                        record_id=str(
                            item.get("record_id")
                            or meta.get("record_id")
                            or meta.get("test_case_id")
                            or meta.get("defect_id")
                            or ""
                        ),
                        supporting_text=verified_text if verified_text else chunk.text[:250],
                        is_supported=is_supported,
                        support_confidence=conf,
                    )
                )

        return validated

    def generate(
        self,
        question: str,
        candidates: List[CandidateChunk],
        retrieval_route: RetrievalRoute = RetrievalRoute.HYBRID,
    ) -> RetrievalResponse:
        """Generate validated, evidence-grounded answer from retrieved chunks."""
        valid_chunk_map = {c.chunk_id: c for c in candidates}

        # Case 1: Empty retrieval candidates -> INSUFFICIENT_EVIDENCE
        if not candidates:
            return RetrievalResponse(
                question=question,
                answer="No relevant evidence chunks could be retrieved to answer this question.",
                status=RetrievalStatus.INSUFFICIENT_EVIDENCE,
                retrieval_route=retrieval_route,
                evidence=[],
                limitations=["No matching records found in the document collection."],
                metadata={"retrieved_chunks_count": 0},
            )

        llm = self._get_llm()

        # Case 2: LLM API key missing or LLM unavailable -> RETRIEVAL_ONLY
        if llm is None:
            top_chunk = candidates[0]
            top_meta = top_chunk.metadata or {}
            fallback_evidence = [
                Evidence(
                    chunk_id=top_chunk.chunk_id,
                    document_type=str(top_meta.get("document_type", "")),
                    source_file=str(top_meta.get("source_file", "")),
                    record_id=str(top_meta.get("record_id") or top_meta.get("test_case_id") or ""),
                    supporting_text=top_chunk.text[:250],
                    is_supported=True,
                    support_confidence=1.0,
                )
            ]
            active_provider = self.config.get_provider(role="generator")
            return RetrievalResponse(
                question=question,
                answer=(
                    f"Retrieved {len(candidates)} relevant evidence chunks, but LLM ({active_provider.upper()}) is not configured "
                    f"for generative synthesis. Top match (Score: {top_chunk.score}): {top_chunk.text[:300]}..."
                ),
                status=RetrievalStatus.RETRIEVAL_ONLY,
                retrieval_route=retrieval_route,
                evidence=fallback_evidence,
                limitations=[f"{active_provider.capitalize()} LLM not configured; output contains raw retrieved evidence only."],
                metadata={"retrieved_chunks_count": len(candidates)},
            )

        # Case 3: Invoke active LLM with structured prompt
        context_str = self._format_evidence_context(candidates)
        messages = [
            SystemMessage(content=ANSWER_GENERATION_SYSTEM_PROMPT),
            HumanMessage(
                content=ANSWER_GENERATION_USER_PROMPT.format(
                    question=question,
                    retrieval_route=retrieval_route.value,
                    num_chunks=len(candidates),
                    evidence_context=context_str,
                )
            ),
        ]

        try:
            response = llm.invoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)
            data = json.loads(raw_text)

            answer_text = str(data.get("answer") or "").strip()

            # Map status safely
            status_str = str(data.get("status", "answered")).lower().strip()
            if status_str in ("answered", "insufficient_evidence", "conflicting_evidence"):
                status = RetrievalStatus(status_str)
            else:
                status = RetrievalStatus.ANSWERED

            # Validate citations against actual candidates and factual support
            raw_evidence = data.get("evidence") or []
            validated_evidence = self._validate_evidence_citations(
                raw_evidence=raw_evidence,
                valid_chunk_map=valid_chunk_map,
                candidates=candidates,
                answer_text=answer_text,
            )

            # Fallback citation if model formulated an answer but omitted citations
            if not validated_evidence and candidates and status == RetrievalStatus.ANSWERED:
                top_c = candidates[0]
                top_m = top_c.metadata or {}
                validated_evidence = [
                    Evidence(
                        chunk_id=top_c.chunk_id,
                        document_type=str(top_m.get("document_type", "")),
                        source_file=str(top_m.get("source_file", "")),
                        record_id=str(top_m.get("record_id") or top_m.get("test_case_id") or ""),
                        supporting_text=top_c.text[:250],
                        is_supported=True,
                        support_confidence=0.8,
                    )
                ]

            limitations = data.get("limitations") or []
            if not isinstance(limitations, list):
                limitations = [str(limitations)]

            return RetrievalResponse(
                question=question,
                answer=answer_text or "No answer could be formulated from evidence.",
                status=status,
                retrieval_route=retrieval_route,
                evidence=validated_evidence,
                limitations=limitations,
                metadata={
                    "model": self.config.get_model_name(role="generator"),
                    "provider": self.config.get_provider(role="generator"),
                    "retrieved_candidates_count": len(candidates),
                    "evidence_citations_count": len(validated_evidence),
                },
            )

        except Exception as e:
            # Case 4: LLM Generation failure (429, timeout, parse error) -> GENERATION_FAILED
            logger.error(f"Answer generation error: {e}. Preserving retrieved evidence chunks.")
            top_chunk = candidates[0]
            top_meta = top_chunk.metadata or {}
            preserved_evidence = [
                Evidence(
                    chunk_id=c.chunk_id,
                    document_type=str((c.metadata or {}).get("document_type", "")),
                    source_file=str((c.metadata or {}).get("source_file", "")),
                    record_id=str((c.metadata or {}).get("record_id") or (c.metadata or {}).get("test_case_id") or ""),
                    supporting_text=c.text[:250],
                    is_supported=True,
                    support_confidence=1.0,
                )
                for c in candidates[:5]
            ]
            return RetrievalResponse(
                question=question,
                answer=f"Evidence retrieved successfully ({len(candidates)} chunks), but LLM generation encountered error: {e}. Top chunk preview: {top_chunk.text[:250]}...",
                status=RetrievalStatus.GENERATION_FAILED,
                retrieval_route=retrieval_route,
                evidence=preserved_evidence,
                limitations=[f"LLM answer generation failed: {e}"],
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "retrieved_candidates_count": len(candidates),
                },
            )
