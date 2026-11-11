"""Query-adaptive retrieval controller for VanillaRAG.

Responsibilities:
- Run the router-selected dense, keyword, or hybrid retrieval.
- Recover exact metadata matches when vector retrieval returns no results.
- Discover verified relationship IDs from router output and retrieved payloads.
- Expand evidence across related document types in bounded Qdrant batches.
- Deduplicate evidence, assess coverage, and rerank the combined candidates.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.reranker import LocalCrossEncoderReranker
from src.retrieval.schemas import (
    CandidateChunk,
    RetrievalMode,
    RetrievalRoute,
    RouterOutput,
)


RELATIONSHIP_KEYS = (
    "test_case_id",
    "defect_id",
    "execution_id",
    "requirement_id",
    "build_number",
)

# Attribute names are normalized to lowercase before lookup.
ATTRIBUTE_DOCUMENT_TYPE_MAP: Dict[str, List[str]] = {
    # UI / screenshot evidence
    "screen": ["screenshot_evidence"],
    "screen_name": ["screenshot_evidence"],
    "screenshot": ["screenshot_evidence"],
    "screenshots": ["screenshot_evidence"],
    "ui": ["screenshot_evidence"],
    "dialog": ["screenshot_evidence"],
    "menu": ["screenshot_evidence"],
    "ocr": ["screenshot_evidence"],
    "active_menus": ["screenshot_evidence"],
    "ui_buttons": ["screenshot_evidence"],

    # Logs
    "log": ["failure_log"],
    "logs": ["failure_log"],
    "failure_log": ["failure_log"],
    "failure_logs": ["failure_log"],
    "error_log": ["failure_log"],
    "error_logs": ["failure_log"],
    "server_log": ["failure_log"],
    "server_logs": ["failure_log"],
    "traceback": ["failure_log"],
    "stack_trace": ["failure_log"],
    "exception_type": ["failure_log"],
    "error_code": ["failure_log"],
    "http_status": ["failure_log"],
    "database_error": ["failure_log"],

    # Test cases
    "test_step": ["test_case"],
    "test_steps": ["test_case"],
    "steps": ["test_case"],
    "preconditions": ["test_case"],
    "expected_result": ["test_case"],
    "actual_result": ["test_case"],
    "test_cases": ["test_case", "test_execution"],
    "test_type": ["test_case"],
    "feature": ["test_case"],

    # Test executions
    "tester": ["test_execution"],
    "execution_date": ["test_execution"],
    "browser": ["test_execution"],
    "operating_system": ["test_execution"],
    "execution_time_seconds": ["test_execution"],
    "execution_notes": ["test_execution"],
    "test_execution": ["test_execution"],
    "executions": ["test_execution"],

    # Defects
    "root_cause": ["defect"],
    "bug_type": ["defect"],
    "fix_version": ["defect"],
    "found_in_build": ["defect"],
    "fixed_build": ["defect"],
    "regression": ["defect"],
    "reproducibility": ["defect"],
    "defects": ["defect"],
    "bugs": ["defect"],

    # Requirements
    "requirement": ["srs_section"],
    "requirements": ["srs_section"],
    "section_title": ["srs_section"],
    "section_number": ["srs_section"],
    "approval_status": ["srs_section"],
    "acceptance_criteria": ["srs_section"],
}


@dataclass
class RetrievalQualityAssessment:
    """Evidence coverage assessment; this is a heuristic, not a truth guarantee."""

    is_sufficient: bool = False
    has_requested_attribute: bool = False
    quality_status: str = "empty"
    discovered_identifiers: Dict[str, List[str]] = field(default_factory=dict)
    attribute_values_found: Dict[str, List[Any]] = field(default_factory=dict)
    conflicting_values: bool = False
    document_types_present: List[str] = field(default_factory=list)
    reasoning: str = ""


class QueryAdaptiveRetrievalController:
    """Coordinate retrieval, relationship expansion, deduplication, and reranking."""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        client: Optional[QdrantClient] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[LocalCrossEncoderReranker] = None,
    ) -> None:
        self.config = config or default_retrieval_config
        self.client = client
        self.dense_retriever = dense_retriever
        self.keyword_retriever = keyword_retriever
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker

    def _execute_initial_retrieval(
        self,
        strategy: RetrievalRoute,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateChunk]:
        """Run the selected retriever. No hidden fallback is performed here."""
        if strategy == RetrievalRoute.KEYWORD and self.keyword_retriever:
            return self.keyword_retriever.retrieve(
                query=query,
                top_k=self.config.sparse_top_k,
                filters=filters,
            )

        if strategy == RetrievalRoute.DENSE and self.dense_retriever:
            return self.dense_retriever.retrieve(
                query=query,
                top_k=self.config.dense_top_k,
                filters=filters,
            )

        if self.hybrid_retriever:
            return self.hybrid_retriever.retrieve(
                query=query,
                top_k=self.config.reranker_top_k,
                filters=filters,
            )

        logger.warning("No retriever is configured for strategy %s", strategy)
        return []

    def _build_qdrant_filter(
        self,
        filters: Optional[Dict[str, Any]],
    ) -> Optional[models.Filter]:
        """Build a safe AND filter using only configured metadata fields."""
        if not filters:
            return None

        allowed_fields = set(self.config.allowed_filter_fields)
        conditions = []

        for key, value in filters.items():
            if key not in allowed_fields or value is None:
                continue

            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue

            if isinstance(value, (list, tuple, set)):
                values = [v for v in value if v is not None and v != ""]
                if not values:
                    continue
                if len(values) == 1:
                    match = models.MatchValue(value=values[0])
                else:
                    match = models.MatchAny(any=values)
                conditions.append(models.FieldCondition(key=key, match=match))
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _point_to_candidate(
        point: Any,
        rank: int,
        retrieval_route: str,
    ) -> CandidateChunk:
        """Convert a Qdrant record or scored point into a candidate chunk."""
        payload = point.payload or {}
        chunk_id = str(
            payload.get("chunk_id")
            or payload.get("document_id")
            or payload.get("record_id")
            or point.id
        )
        text = str(
            payload.get("page_content")
            or payload.get("text")
            or payload.get("content")
            or ""
        )

        # Handle score safely for both Record (no score attribute) and ScoredPoint
        point_score = getattr(point, "score", None)
        score = float(point_score) if point_score is not None else 0.0

        return CandidateChunk(
            chunk_id=chunk_id,
            text=text,
            score=score,
            original_rank=rank,
            retrieval_route=retrieval_route,
            metadata=dict(payload),
            dense_rank=None,
            sparse_rank=None,
            rrf_score=None,
            rerank_score=None,
        )

    def _retrieve_by_metadata_scroll(
        self,
        filters: Dict[str, Any],
        limit: int,
    ) -> List[CandidateChunk]:
        """Retrieve exact metadata matches from the primary/dense collection."""
        if not self.client:
            return []

        qdrant_filter = self._build_qdrant_filter(filters)
        if qdrant_filter is None:
            return []

        try:
            scroll_res = self.client.scroll(
                collection_name=self.config.dense_collection,
                scroll_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            if isinstance(scroll_res, tuple) and len(scroll_res) == 2:
                points = scroll_res[0]
            elif isinstance(scroll_res, list):
                points = scroll_res
            elif hasattr(scroll_res, "points"):
                points = scroll_res.points
            else:
                points = []

            return [
                self._point_to_candidate(point, rank, "metadata_scroll")
                for rank, point in enumerate(points, start=1)
            ]
        except Exception as exc:
            logger.warning("Metadata scroll failed for filters %s: %s", filters, exc)
            return []

    @staticmethod
    def _add_identifier(
        discovered: Dict[str, Set[str]],
        key: str,
        value: Any,
    ) -> None:
        """Normalize and add scalar or list-valued identifier metadata."""
        if key not in discovered or value is None:
            return

        values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in values:
            if item is None:
                continue
            cleaned = str(item).strip()
            if cleaned:
                discovered[key].add(cleaned)

    def _harvest_relationship_ids(
        self,
        candidates: List[CandidateChunk],
        router_output: RouterOutput,
    ) -> Dict[str, Set[str]]:
        """Collect only identifiers explicitly present in routing or payload metadata."""
        discovered: Dict[str, Set[str]] = {
            key: set() for key in RELATIONSHIP_KEYS
        }

        # Use router-extracted identifiers as the primary source.
        for key, value in (router_output.extracted_identifiers or {}).items():
            self._add_identifier(discovered, key, value)

        # Include the primary entity ID when its type is known.
        primary_id = router_output.primary_entity_id
        primary_type = (router_output.primary_entity_type or "").lower()
        primary_type_to_key = {
            "defect": "defect_id",
            "test_case": "test_case_id",
            "testcase": "test_case_id",
            "test_execution": "execution_id",
            "execution": "execution_id",
            "srs_section": "requirement_id",
            "requirement": "requirement_id",
            "build": "build_number",
        }
        if primary_id and primary_type in primary_type_to_key:
            self._add_identifier(
                discovered,
                primary_type_to_key[primary_type],
                primary_id,
            )

        # Fallback prefix recognition for IDs when router type is not supplied.
        if primary_id and not any(discovered.values()):
            value = str(primary_id).strip()
            prefix_map = (
                (("BUG-", "DEFECT-", "ISSUE-"), "defect_id"),
                (("TC-",), "test_case_id"),
                (("EXEC-", "RUN-"), "execution_id"),
                (("REQ-", "FR-", "NFR-"), "requirement_id"),
                (("B-",), "build_number"),
            )
            for prefixes, key in prefix_map:
                if value.startswith(prefixes):
                    self._add_identifier(discovered, key, value)
                    break

        # Add foreign keys found in every retrieved payload.
        for candidate in candidates:
            metadata = candidate.metadata or {}
            for key in RELATIONSHIP_KEYS:
                self._add_identifier(discovered, key, metadata.get(key))

        return discovered

    def _determine_target_modalities(
        self,
        query: str,
        router_output: RouterOutput,
        initial_candidates: List[CandidateChunk],
    ) -> List[str]:
        """Infer document types relevant to the requested attribute."""
        targets = set(router_output.target_document_types or [])
        requested = (router_output.requested_attribute or "").strip().lower()
        normalized_query = query.lower().replace("-", "_")

        if requested in ATTRIBUTE_DOCUMENT_TYPE_MAP:
            targets.update(ATTRIBUTE_DOCUMENT_TYPE_MAP[requested])

        for attribute, document_types in ATTRIBUTE_DOCUMENT_TYPE_MAP.items():
            if attribute in normalized_query or attribute.replace("_", " ") in query.lower():
                targets.update(document_types)

        # A direct-entity route can still need expansion if the requested
        # attribute belongs to another document type. Do not clear targets
        # merely because the router marked the query as a direct entity query.
        present_types = {
            (candidate.metadata or {}).get("document_type")
            for candidate in initial_candidates
            if (candidate.metadata or {}).get("document_type")
        }
        targets.difference_update(present_types)

        return sorted(targets)

    def _batch_expand_evidence(
        self,
        target_doc_types: List[str],
        discovered_ids: Dict[str, Set[str]],
        limit_per_modality: int = 20,
    ) -> List[CandidateChunk]:
        """Fetch related records with bounded MatchAny queries."""
        if not self.client or not target_doc_types:
            return []

        expanded: List[CandidateChunk] = []
        allowed_fields = set(self.config.allowed_filter_fields)

        for document_type in target_doc_types:
            for foreign_key, identifier_set in discovered_ids.items():
                if not identifier_set or foreign_key not in allowed_fields:
                    continue

                identifiers = sorted(identifier_set)[:25]
                if not identifiers:
                    continue

                conditions = [
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchValue(value=document_type),
                    )
                ]

                if len(identifiers) == 1:
                    identifier_match = models.MatchValue(value=identifiers[0])
                else:
                    identifier_match = models.MatchAny(any=identifiers)

                conditions.append(
                    models.FieldCondition(
                        key=foreign_key,
                        match=identifier_match,
                    )
                )

                try:
                    points, _ = self.client.scroll(
                        collection_name=self.config.dense_collection,
                        scroll_filter=models.Filter(must=conditions),
                        limit=limit_per_modality,
                        with_payload=True,
                        with_vectors=False,
                    )
                    expanded.extend(
                        self._point_to_candidate(point, rank, "relationship_expansion")
                        for rank, point in enumerate(points, start=1)
                    )
                except Exception as exc:
                    logger.debug(
                        "Expansion failed for document_type=%s, key=%s: %s",
                        document_type,
                        foreign_key,
                        exc,
                    )

        return expanded

    @staticmethod
    def _deduplicate_candidates(
        candidates: List[CandidateChunk],
    ) -> List[CandidateChunk]:
        """Deduplicate by chunk ID while preserving scores and filling missing metadata."""
        unique: Dict[str, CandidateChunk] = {}

        for candidate in candidates:
            candidate_id = str(candidate.chunk_id)
            if candidate_id not in unique:
                unique[candidate_id] = candidate
                continue

            existing = unique[candidate_id]
            existing_metadata = existing.metadata or {}
            incoming_metadata = candidate.metadata or {}

            for key, value in incoming_metadata.items():
                if key not in existing_metadata or existing_metadata[key] in (None, "", [], {}):
                    existing_metadata[key] = value
            existing.metadata = existing_metadata

            # Keep the candidate with a real retrieval score when one exists.
            if (existing.score or 0.0) <= 0.0 and (candidate.score or 0.0) > 0.0:
                existing.score = candidate.score
                existing.original_rank = candidate.original_rank
                existing.dense_rank = candidate.dense_rank
                existing.sparse_rank = candidate.sparse_rank
                existing.rrf_score = candidate.rrf_score
                existing.retrieval_route = candidate.retrieval_route

        return list(unique.values())

    def evaluate_retrieval_quality(
        self,
        candidates: List[CandidateChunk],
        requested_attribute: Optional[str],
        query: str,
    ) -> RetrievalQualityAssessment:
        """Estimate whether the retrieved evidence contains the requested attribute."""
        if not candidates:
            return RetrievalQualityAssessment(
                is_sufficient=False,
                has_requested_attribute=False,
                quality_status="empty",
                reasoning="No candidate records were retrieved.",
            )

        document_types = sorted({
            str((candidate.metadata or {}).get("document_type"))
            for candidate in candidates
            if (candidate.metadata or {}).get("document_type")
        })

        values_found: Dict[str, List[Any]] = {}
        has_attribute = True

        if requested_attribute:
            attribute = requested_attribute.strip().lower()
            metadata_keys = {
                attribute,
                attribute.replace(" ", "_"),
                attribute.replace("_", " "),
            }
            found_values: List[Any] = []

            for candidate in candidates:
                metadata = candidate.metadata or {}
                for key in metadata_keys:
                    value = metadata.get(key)
                    if value is not None:
                        found_values.append(value)
                        break

            if found_values:
                unique_values: List[Any] = []
                for value in found_values:
                    if value not in unique_values:
                        unique_values.append(value)
                values_found[attribute] = unique_values
                has_attribute = True
            else:
                # This is a weak lexical check; the answer generator still
                # needs to validate the actual evidence text.
                needle = attribute.replace("_", " ").lower()
                has_attribute = any(
                    needle in (candidate.text or "").lower()
                    for candidate in candidates
                )

        conflict = False
        if requested_attribute:
            normalized_attribute = requested_attribute.strip().lower()
            values = values_found.get(normalized_attribute, [])
            if len(values) > 1 and normalized_attribute in {
                "status", "priority", "severity"
            }:
                conflict = True

        status = "sufficient" if has_attribute else "partial"
        return RetrievalQualityAssessment(
            is_sufficient=bool(candidates) and has_attribute,
            has_requested_attribute=has_attribute,
            quality_status=status,
            attribute_values_found=values_found,
            conflicting_values=conflict,
            document_types_present=document_types,
            reasoning=(
                f"Retrieved {len(candidates)} candidates across document types "
                f"{document_types}. Requested attribute "
                f"'{requested_attribute}' detected: {has_attribute}."
            ),
        )

    def execute_adaptive_retrieval(
        self,
        query: str,
        router_output: RouterOutput,
        override_filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[CandidateChunk], RetrievalQualityAssessment, Dict[str, Any]]:
        """Run retrieval, fallback, evidence expansion, deduplication, assessment, reranking."""
        controller_start = time.perf_counter()
        telemetry: Dict[str, Any] = {}

        strategy = router_output.strategy
        filters = dict(router_output.validated_filters or {})
        if override_filters:
            filters.update(override_filters)

        # 1. Initial vector retrieval with router-selected strategy and filters.
        retrieval_start = time.perf_counter()
        raw_candidates = self._execute_initial_retrieval(
            strategy=strategy,
            query=query,
            filters=filters or None,
        )
        telemetry["initial_retrieval_duration_seconds"] = round(
            time.perf_counter() - retrieval_start, 3
        )
        initial_count = len(raw_candidates)
        telemetry["initial_candidates_count"] = initial_count

        # 2. Controlled fallback: exact metadata lookup first, then query retrieval
        # without restrictive filters. The relaxed retrieval is recorded explicitly.
        fallback_applied = False
        if not raw_candidates:
            if filters:
                logger.info(
                    "Filtered %s retrieval returned no candidates; trying metadata scroll.",
                    strategy.value,
                )
                raw_candidates = self._retrieve_by_metadata_scroll(
                    filters=filters,
                    limit=self.config.dense_top_k,
                )
                if raw_candidates:
                    fallback_applied = True
                    telemetry["fallback_type"] = "exact_metadata_scroll"
                    telemetry["fallback_reason"] = "Retrieved via exact metadata scroll match."

                # If document_type caused an overly restrictive match, retry using
                # the remaining verified entity filters. Do not drop entity IDs.
                if not raw_candidates and "document_type" in filters:
                    relaxed_filters = {
                        key: value
                        for key, value in filters.items()
                        if key != "document_type"
                    }
                    if relaxed_filters:
                        raw_candidates = self._retrieve_by_metadata_scroll(
                            filters=relaxed_filters,
                            limit=self.config.dense_top_k,
                        )
                        if raw_candidates:
                            fallback_applied = True
                            telemetry["fallback_type"] = "metadata_scroll_without_document_type"
                            telemetry["fallback_reason"] = "Relaxed document_type while preserving entity ID filters."

            # Check if an explicit entity identifier was requested or available
            entity_id_keys = ("defect_id", "test_case_id", "execution_id", "requirement_id", "record_id")
            specific_entity_id = (
                router_output.primary_entity_id
                or next((str(filters[k]).strip() for k in entity_id_keys if k in filters and filters[k]), None)
            )

            # Try matching record_id or specific entity field if not yet matched
            if not raw_candidates and specific_entity_id:
                inferred_key = None
                val = str(specific_entity_id).upper().strip()
                if val.startswith(("BUG-", "DEFECT-", "ISSUE-")):
                    inferred_key = "defect_id"
                elif val.startswith("TC-"):
                    inferred_key = "test_case_id"
                elif val.startswith(("EXEC-", "RUN-")):
                    inferred_key = "execution_id"
                elif val.startswith(("REQ-", "FR-", "NFR-")):
                    inferred_key = "requirement_id"

                if inferred_key:
                    raw_candidates = self._retrieve_by_metadata_scroll(
                        filters={inferred_key: val},
                        limit=self.config.dense_top_k,
                    )
                if not raw_candidates:
                    raw_candidates = self._retrieve_by_metadata_scroll(
                        filters={"record_id": specific_entity_id},
                        limit=self.config.dense_top_k,
                    )
                if raw_candidates:
                    fallback_applied = True
                    telemetry["fallback_type"] = "record_id_metadata_scroll"
                    telemetry["fallback_reason"] = f"Entity matched via identifier='{specific_entity_id}'."

            # Step 4 & 5: If still 0 results, execute query-based fallback with logging
            if not raw_candidates:
                fallback_strategy = (
                    router_output.fallback_strategy or RetrievalRoute.HYBRID
                )
                logger.warning(
                    "Filtered %s retrieval returned 0 candidates for filters %s. "
                    "Executing fallback query-based retrieval using %s strategy with relaxed filters.",
                    strategy.value,
                    filters,
                    fallback_strategy.value,
                )
                raw_candidates = self._execute_initial_retrieval(
                    strategy=fallback_strategy,
                    query=query,
                    filters=None,
                )
                if raw_candidates:
                    fallback_applied = True
                    telemetry["fallback_type"] = "unfiltered_" + fallback_strategy.value
                    telemetry["fallback_reason"] = (
                        f"Filtered {strategy.value} retrieval returned 0 results for {filters}; "
                        f"switched to unfiltered {fallback_strategy.value} retrieval."
                    )
                else:
                    logger.warning(
                        "Fallback %s retrieval also returned 0 candidates for query: %r",
                        fallback_strategy.value,
                        query[:80],
                    )
                    telemetry["fallback_type"] = "fallback_empty"
                    telemetry["fallback_reason"] = f"No candidates found after filter relaxation."

        telemetry["filter_fallback_applied"] = fallback_applied

        # 3. Discover only IDs explicitly present in router output or payloads.
        discovered_ids = self._harvest_relationship_ids(
            raw_candidates,
            router_output,
        )
        telemetry["discovered_relationship_ids"] = {
            key: sorted(values)
            for key, values in discovered_ids.items()
            if values
        }

        # 4. Expand to document types relevant to the requested attribute.
        target_modalities = self._determine_target_modalities(
            query,
            router_output,
            raw_candidates,
        )
        telemetry["target_modalities"] = target_modalities

        should_expand = bool(
            target_modalities and any(discovered_ids.values())
        )
        expansion_start = time.perf_counter()
        expanded_candidates: List[CandidateChunk] = []

        if should_expand:
            expanded_candidates = self._batch_expand_evidence(
                target_doc_types=target_modalities,
                discovered_ids=discovered_ids,
                limit_per_modality=self.config.expansion_top_k,
            )

        telemetry["expansion_duration_seconds"] = round(
            time.perf_counter() - expansion_start, 3
        )
        telemetry["expanded_candidates_count"] = len(expanded_candidates)

        # 5. Deduplicate initial and expanded evidence.
        combined = self._deduplicate_candidates(
            raw_candidates + expanded_candidates
        )
        telemetry["deduplicated_candidates_count"] = len(combined)

        # 6. Assess the complete evidence set before selecting final evidence.
        assessment = self.evaluate_retrieval_quality(
            candidates=combined,
            requested_attribute=router_output.requested_attribute,
            query=query,
        )
        assessment.discovered_identifiers = {
            key: sorted(values)
            for key, values in discovered_ids.items()
            if values
        }
        telemetry["quality_status"] = assessment.quality_status
        telemetry["has_requested_attribute"] = assessment.has_requested_attribute
        telemetry["attribute_values_found"] = assessment.attribute_values_found

        # Keep a larger evidence set when multiple modalities were combined.
        final_k = self.config.final_evidence_top_k
        if should_expand or len(assessment.document_types_present) > 1:
            final_k = max(final_k, 12)

        # 7. Rerank AFTER expansion and deduplication, so all evidence competes
        # in one cross-encoder ranking pass.
        rerank_start = time.perf_counter()
        if self.reranker and combined:
            if router_output.retrieval_mode == RetrievalMode.DIRECT_ENTITY and len(combined) <= 5:
                pool_to_rerank = combined
            else:
                pool_to_rerank = combined[: self.config.reranker_top_k]

            evidence_chunks = self.reranker.rerank(
                query=query,
                candidates=pool_to_rerank,
                top_k=final_k,
            )
            telemetry["reranker_used"] = True
        else:
            evidence_chunks = combined[:final_k]
            telemetry["reranker_used"] = False

        telemetry["reranker_duration_seconds"] = round(
            time.perf_counter() - rerank_start, 3
        )
        telemetry["final_evidence_count"] = len(evidence_chunks)
        telemetry["controller_duration_seconds"] = round(
            time.perf_counter() - controller_start, 3
        )

        logger.info(
            "Adaptive retrieval: initial=%d expanded=%d deduplicated=%d "
            "final=%d quality=%s rerank=%.3fs",
            initial_count,
            len(expanded_candidates),
            len(combined),
            len(evidence_chunks),
            assessment.quality_status,
            telemetry["reranker_duration_seconds"],
        )

        return evidence_chunks, assessment, telemetry
