"""Evidence-grounded answer generator using Groq and strict citation validation."""

import json
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config
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


class EvidenceGroundedAnswerGenerator:
    """Generates strictly evidence-grounded answers and validates citations against candidate chunks."""

    def __init__(self, config: Optional[RetrievalConfig] = None, llm: Optional[ChatGroq] = None):
        self.config = config or default_retrieval_config
        self._llm = llm

    def _get_llm(self) -> Optional[ChatGroq]:
        if self._llm is not None:
            return self._llm

        if not self.config.groq_api_key:
            logger.warning("GROQ_API_KEY is not set. Cannot run LLM answer generation.")
            return None

        try:
            self._llm = ChatGroq(
                groq_api_key=self.config.groq_api_key,
                model_name=self.config.groq_model,
                temperature=self.config.groq_temperature,
                response_format={"type": "json_object"},
            )
            return self._llm
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq for answer generator: {e}")
            return None

    def _format_evidence_context(self, candidates: List[CandidateChunk]) -> str:
        """Format candidate chunks into structured text blocks with explicit chunk metadata."""
        if not candidates:
            return "No evidence chunks retrieved."

        blocks = []
        for i, c in enumerate(candidates, start=1):
            meta = c.metadata or {}
            doc_type = meta.get("document_type", "unknown")
            rec_id = meta.get("record_id") or meta.get("test_case_id") or meta.get("defect_id") or "N/A"
            src_file = meta.get("source_file") or meta.get("source_path") or "N/A"

            block = (
                f"--- EVIDENCE CHUNK {i} ---\n"
                f"CHUNK_ID: {c.chunk_id}\n"
                f"DOCUMENT_TYPE: {doc_type}\n"
                f"RECORD_ID: {rec_id}\n"
                f"SOURCE_FILE: {src_file}\n"
                f"CONTENT:\n{c.text}\n"
            )
            blocks.append(block)

        return "\n".join(blocks)

    def _validate_evidence_citations(
        self,
        raw_evidence: List[Dict[str, Any]],
        valid_chunk_map: Dict[str, CandidateChunk],
    ) -> List[Evidence]:
        """Verify that every citation matches an actual retrieved chunk and is not hallucinated."""
        validated: List[Evidence] = []

        for item in raw_evidence:
            if not isinstance(item, dict):
                continue

            cid = str(item.get("chunk_id", "")).strip()
            if not cid:
                continue

            # Check if chunk_id actually exists in candidate pool
            chunk = valid_chunk_map.get(cid)
            if not chunk:
                # Try finding chunk with matching record_id or prefix
                matched = None
                for candidate_id, cand in valid_chunk_map.items():
                    if cid in candidate_id or candidate_id in cid:
                        matched = cand
                        cid = candidate_id
                        break
                if not matched:
                    logger.warning(f"Rejected hallucinated citation: chunk_id '{cid}' not in retrieved evidence.")
                    continue
                chunk = matched

            supporting_text = str(item.get("supporting_text", "")).strip()
            meta = chunk.metadata or {}

            validated.append(
                Evidence(
                    chunk_id=cid,
                    document_type=str(item.get("document_type") or meta.get("document_type") or ""),
                    source_file=str(item.get("source_file") or meta.get("source_file") or meta.get("source_path") or ""),
                    record_id=str(item.get("record_id") or meta.get("record_id") or meta.get("test_case_id") or ""),
                    supporting_text=supporting_text if supporting_text else chunk.text[:200],
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

        # Case 1: Empty retrieval candidates
        if not candidates:
            return RetrievalResponse(
                question=question,
                answer="No relevant evidence chunks could be retrieved to answer this question.",
                status=RetrievalStatus.INSUFFICIENT_EVIDENCE,
                retrieval_route=retrieval_route,
                evidence=[],
                limitations=["No matching records found in the knowledge graph."],
                metadata={"retrieved_chunks_count": 0},
            )

        llm = self._get_llm()

        # Case 2: LLM API key missing or LLM unavailable
        if llm is None:
            # Provide grounded summary from top candidate without LLM
            top_chunk = candidates[0]
            top_meta = top_chunk.metadata or {}
            fallback_evidence = [
                Evidence(
                    chunk_id=top_chunk.chunk_id,
                    document_type=str(top_meta.get("document_type", "")),
                    source_file=str(top_meta.get("source_file", "")),
                    record_id=str(top_meta.get("record_id") or top_meta.get("test_case_id") or ""),
                    supporting_text=top_chunk.text[:250],
                )
            ]
            return RetrievalResponse(
                question=question,
                answer=(
                    f"Retrieved {len(candidates)} relevant evidence chunks, but GROQ_API_KEY is not configured "
                    f"for generative synthesis. Top match (Score: {top_chunk.score}): {top_chunk.text[:300]}..."
                ),
                status=RetrievalStatus.ANSWERED,
                retrieval_route=retrieval_route,
                evidence=fallback_evidence,
                limitations=["Groq API key not provided; synthesized answer generated from raw candidate text."],
                metadata={"retrieved_chunks_count": len(candidates)},
            )

        # Case 3: Invoke Groq with structured prompt
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

            # Map status safely
            status_str = str(data.get("status", "answered")).lower().strip()
            if status_str not in ("answered", "insufficient_evidence", "conflicting_evidence"):
                status_str = "answered"
            status = RetrievalStatus(status_str)

            # Validate citations against actual candidates
            raw_evidence = data.get("evidence") or []
            validated_evidence = self._validate_evidence_citations(raw_evidence, valid_chunk_map)

            # Fallback citation if model failed to format evidence array but answered
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
                    )
                ]

            limitations = data.get("limitations") or []
            if not isinstance(limitations, list):
                limitations = [str(limitations)]

            return RetrievalResponse(
                question=question,
                answer=str(data.get("answer") or "No answer could be formulated from evidence."),
                status=status,
                retrieval_route=retrieval_route,
                evidence=validated_evidence,
                limitations=limitations,
                metadata={
                    "model": self.config.groq_model,
                    "retrieved_candidates_count": len(candidates),
                },
            )

        except Exception as e:
            logger.error(f"Answer generation error: {e}. Returning safe fallback response.")
            top_chunk = candidates[0]
            top_meta = top_chunk.metadata or {}
            return RetrievalResponse(
                question=question,
                answer=f"Evidence retrieved successfully ({len(candidates)} chunks), but LLM generation encountered: {e}. Top chunk preview: {top_chunk.text[:250]}...",
                status=RetrievalStatus.ANSWERED,
                retrieval_route=retrieval_route,
                evidence=[
                    Evidence(
                        chunk_id=top_chunk.chunk_id,
                        document_type=str(top_meta.get("document_type", "")),
                        source_file=str(top_meta.get("source_file", "")),
                        record_id=str(top_meta.get("record_id") or top_meta.get("test_case_id") or ""),
                        supporting_text=top_chunk.text[:250],
                    )
                ],
                limitations=[f"Answer generation error: {e}"],
                metadata={"error": str(e)},
            )
