"""End-to-end query routing, adaptive retrieval, and grounded answer generation."""

import time
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient

from src.ingestion.logging_config import logger
from src.retrieval.answer_generator import EvidenceGroundedAnswerGenerator
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.controller import QueryAdaptiveRetrievalController
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.reranker import LocalCrossEncoderReranker
from src.retrieval.router import LLMQueryRouter
from src.retrieval.schemas import (
    CandidateChunk,
    RetrievalMode,
    RetrievalResponse,
    RetrievalRoute,
    RouterOutput,
)


class AdvancedRetrievalPipeline:
    """Run routing -> retrieval/expansion -> reranking -> grounded generation."""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        router: Optional[LLMQueryRouter] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[LocalCrossEncoderReranker] = None,
        generator: Optional[EvidenceGroundedAnswerGenerator] = None,
        qdrant_client: Optional[QdrantClient] = None,
        controller: Optional[QueryAdaptiveRetrievalController] = None,
    ) -> None:
        self.config = config or default_retrieval_config

        # Share one Qdrant client across all retrievers and the controller.
        if qdrant_client is not None:
            self.qdrant_client = qdrant_client
        elif self.config.qdrant_url:
            self.qdrant_client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
            )
        else:
            self.qdrant_client = QdrantClient(path=self.config.qdrant_path)

        self.router = router or LLMQueryRouter(self.config)

        self.dense_retriever = dense_retriever or DenseRetriever(
            self.config,
            client=self.qdrant_client,
        )
        self.keyword_retriever = keyword_retriever or KeywordRetriever(
            self.config,
            client=self.qdrant_client,
        )
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            self.config,
            dense_retriever=self.dense_retriever,
            keyword_retriever=self.keyword_retriever,
        )

        self.reranker = reranker or LocalCrossEncoderReranker(self.config)
        self.generator = generator or EvidenceGroundedAnswerGenerator(self.config)

        self.controller = controller or QueryAdaptiveRetrievalController(
            config=self.config,
            client=self.qdrant_client,
            dense_retriever=self.dense_retriever,
            keyword_retriever=self.keyword_retriever,
            hybrid_retriever=self.hybrid_retriever,
            reranker=self.reranker,
        )

    def run(
        self,
        question: str,
        override_strategy: Optional[RetrievalRoute] = None,
        override_filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResponse:
        """Execute the complete retrieval and answer-generation workflow."""
        total_start = time.perf_counter()
        clean_question = (question or "").strip()

        if not clean_question:
            raise ValueError("question must not be empty")

        logger.info("=" * 80)
        logger.info("PROCESSING RETRIEVAL QUERY: %s", clean_question)
        logger.info("=" * 80)

        # Stage 1: Query routing.
        route_start = time.perf_counter()
        if override_strategy is None:
            router_output = self.router.route(clean_question)
        else:
            # Construct a valid routing object for manual strategy overrides.
            router_output = RouterOutput(
                intent="Manual retrieval strategy override",
                strategy=override_strategy,
                fallback_strategy=(
                    RetrievalRoute.HYBRID
                    if override_strategy != RetrievalRoute.HYBRID
                    else RetrievalRoute.DENSE
                ),
                retrieval_mode=RetrievalMode.CONCEPTUAL,
                extracted_identifiers={},
                validated_filters={},
                reasoning="The caller supplied override_strategy.",
            )

        if override_filters:
            merged_filters = dict(router_output.validated_filters or {})
            merged_filters.update(override_filters)
            router_output.validated_filters = merged_filters

        route_duration = time.perf_counter() - route_start
        strategy = router_output.strategy
        applied_filters = dict(router_output.validated_filters or {})

        logger.info(
            "Router: strategy=%s mode=%s filters=%s identifiers=%s time=%.3fs",
            strategy.value,
            router_output.retrieval_mode.value,
            applied_filters,
            router_output.extracted_identifiers,
            route_duration,
        )

        # Stage 2: Controller performs initial retrieval, fallback, expansion,
        # deduplication, quality assessment, and the final reranking pass.
        retrieval_start = time.perf_counter()
        (
            evidence_chunks,
            quality_assessment,
            controller_telemetry,
        ) = self.controller.execute_adaptive_retrieval(
            query=clean_question,
            router_output=router_output,
            override_filters=override_filters,
        )
        retrieval_duration = time.perf_counter() - retrieval_start

        # Stage 3: Generate the answer using the final, reranked evidence set.
        generation_start = time.perf_counter()
        response = self.generator.generate(
            question=clean_question,
            candidates=evidence_chunks,
            retrieval_route=strategy,
        )
        generation_duration = time.perf_counter() - generation_start
        total_duration = time.perf_counter() - total_start

        # Ensure metadata is always a mutable dictionary.
        if response.metadata is None:
            response.metadata = {}

        rerank_duration = float(controller_telemetry.get("reranker_duration_seconds", 0.0) or 0.0)
        tot_duration = round(total_duration, 3)
        rt_duration = round(route_duration, 3)
        ret_duration = round(retrieval_duration, 3)
        gen_duration = round(generation_duration, 3)

        response.metadata.update(
            {
                # Standard timing metrics with multiple aliases to support diverse evaluators
                "total_duration_seconds": tot_duration,
                "total_latency": tot_duration,
                "total_latency_seconds": tot_duration,
                "routing_duration_seconds": rt_duration,
                "routing_latency": rt_duration,
                "retrieval_duration_seconds": ret_duration,
                "retrieval_latency": ret_duration,
                "generation_duration_seconds": gen_duration,
                "generation_latency": gen_duration,
                "reranker_duration_seconds": round(rerank_duration, 3),
                "rerank_latency": round(rerank_duration, 3),
                "router_intent": router_output.intent,
                "router_reasoning": router_output.reasoning,
                "retrieval_strategy": strategy.value,
                "retrieval_mode": router_output.retrieval_mode.value,
                "primary_entity_type": router_output.primary_entity_type,
                "primary_entity_id": router_output.primary_entity_id,
                "requested_attribute": router_output.requested_attribute,
                "target_document_types": router_output.target_document_types,
                "extracted_identifiers": router_output.extracted_identifiers,
                "applied_filters": applied_filters,
                "quality_status": quality_assessment.quality_status,
                "has_requested_attribute": quality_assessment.has_requested_attribute,
                "attribute_values_found": quality_assessment.attribute_values_found,
                "conflicting_values": quality_assessment.conflicting_values,
                **controller_telemetry,
            }
        )

        # Stable keys for callers/UI code that expect these metrics.
        response.metadata.setdefault("reranker_duration_seconds", round(rerank_duration, 3))
        response.metadata.setdefault("rerank_latency", round(rerank_duration, 3))
        response.metadata.setdefault("reranker_used", bool(rerank_duration > 0))
        response.metadata.setdefault("initial_candidates_count", 0)
        response.metadata.setdefault("expanded_candidates_count", 0)
        response.metadata.setdefault("deduplicated_candidates_count", 0)
        response.metadata.setdefault("final_evidence_count", len(evidence_chunks))

        logger.info(
            "Generation: status=%s evidence=%d total=%.3fs",
            response.status.value,
            len(response.evidence),
            total_duration,
        )
        logger.info("=" * 80)

        return response

    def _expand_relationships(
        self,
        candidates: List[CandidateChunk],
        router_output: RouterOutput,
    ) -> List[CandidateChunk]:
        """Backwards-compatible helper delegating relationship expansion to controller."""
        discovered = self.controller._harvest_relationship_ids(candidates, router_output)
        targets = self.controller._determine_target_modalities("", router_output, candidates)
        if not targets:
            targets = router_output.target_document_types or []
        return self.controller._batch_expand_evidence(targets, discovered)

