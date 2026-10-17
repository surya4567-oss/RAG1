"""End-to-end orchestration pipeline for advanced multi-stage retrieval and evidence grounding."""

import time
from typing import Optional

from qdrant_client import QdrantClient

from src.ingestion.logging_config import logger
from src.retrieval.answer_generator import EvidenceGroundedAnswerGenerator
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.reranker import LocalCrossEncoderReranker
from src.retrieval.router import LLMQueryRouter
from src.retrieval.schemas import (
    CandidateChunk,
    RetrievalResponse,
    RetrievalRoute,
    RouterOutput,
)


class AdvancedRetrievalPipeline:
    """Orchestrates query routing, candidate retrieval, RRF fusion, cross-encoder reranking, and answer generation."""

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
    ):
        self.config = config or default_retrieval_config

        # Shared Qdrant client to avoid multi-load overhead
        if qdrant_client is None:
            if self.config.qdrant_url:
                self.qdrant_client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key,
                )
            else:
                self.qdrant_client = QdrantClient(path=self.config.qdrant_path)
        else:
            self.qdrant_client = qdrant_client

        # Sub-modules
        self.router = router or LLMQueryRouter(self.config)
        self.dense_retriever = dense_retriever or DenseRetriever(self.config, client=self.qdrant_client)
        self.keyword_retriever = keyword_retriever or KeywordRetriever(self.config, client=self.qdrant_client)
        self.hybrid_retriever = hybrid_retriever or HybridRetriever(
            self.config,
            dense_retriever=self.dense_retriever,
            keyword_retriever=self.keyword_retriever,
        )
        self.reranker = reranker or LocalCrossEncoderReranker(self.config)
        self.generator = generator or EvidenceGroundedAnswerGenerator(self.config)

    def run(
        self,
        question: str,
        override_strategy: Optional[RetrievalRoute] = None,
        override_filters: Optional[dict] = None,
    ) -> RetrievalResponse:
        """Execute the complete end-to-end retrieval, reranking, and grounding pipeline."""
        start_total = time.perf_counter()
        clean_question = question.strip()

        logger.info("=" * 80)
        logger.info(f"PROCESSING RETRIEVAL QUERY: '{clean_question}'")
        logger.info("=" * 80)

        # Stage 1: Query Routing & Entity Extraction
        t0_route = time.perf_counter()
        if override_strategy:
            router_output = RouterOutput(
                intent="User manual override",
                strategy=override_strategy,
                extracted_identifiers={},
                validated_filters=override_filters or {},
                reasoning="Manual override supplied by caller.",
            )
        else:
            router_output = self.router.route(clean_question)
            if override_filters:
                router_output.validated_filters.update(override_filters)

        route_duration = time.perf_counter() - t0_route
        strategy = router_output.strategy
        filters = router_output.validated_filters

        logger.info(
            f"Stage 1 [Router]: Strategy='{strategy.value}' | Filters={filters} | "
            f"IDs={router_output.extracted_identifiers} | Duration={route_duration:.3f}s"
        )

        # Stage 2: Candidate Retrieval based on selected strategy
        t0_retrieve = time.perf_counter()
        raw_candidates = []

        if strategy == RetrievalRoute.KEYWORD:
            raw_candidates = self.keyword_retriever.retrieve(
                query=clean_question,
                top_k=self.config.sparse_top_k,
                filters=filters,
            )
        elif strategy == RetrievalRoute.DENSE:
            raw_candidates = self.dense_retriever.retrieve(
                query=clean_question,
                top_k=self.config.dense_top_k,
                filters=filters,
            )
        else:  # RetrievalRoute.HYBRID
            raw_candidates = self.hybrid_retriever.retrieve(
                query=clean_question,
                top_k=self.config.reranker_top_k,
                filters=filters,
            )

        retrieve_duration = time.perf_counter() - t0_retrieve
        logger.info(f"Stage 2 [Retrieval]: Retrieved {len(raw_candidates)} initial candidates in {retrieve_duration:.3f}s")

        # Stage 3: Cross-Encoder Reranking
        t0_rerank = time.perf_counter()
        evidence_chunks = self.reranker.rerank(
            query=clean_question,
            candidates=raw_candidates,
            top_k=self.config.final_evidence_top_k,
        )
        rerank_duration = time.perf_counter() - t0_rerank
        logger.info(
            f"Stage 3 [Reranker]: Scored {len(raw_candidates)} -> selected top {len(evidence_chunks)} evidence chunks in {rerank_duration:.3f}s"
        )

        # Stage 4: Evidence-Grounded Answer Generation & Validation
        t0_gen = time.perf_counter()
        response = self.generator.generate(
            question=clean_question,
            candidates=evidence_chunks,
            retrieval_route=strategy,
        )
        gen_duration = time.perf_counter() - t0_gen
        total_duration = time.perf_counter() - start_total

        # Attach telemetry metadata
        response.metadata.update(
            {
                "total_duration_seconds": round(total_duration, 3),
                "routing_duration_seconds": round(route_duration, 3),
                "retrieval_duration_seconds": round(retrieve_duration, 3),
                "reranker_duration_seconds": round(rerank_duration, 3),
                "generation_duration_seconds": round(gen_duration, 3),
                "router_intent": router_output.intent,
                "router_reasoning": router_output.reasoning,
                "extracted_identifiers": router_output.extracted_identifiers,
                "applied_filters": filters,
                "initial_candidates_count": len(raw_candidates),
                "final_evidence_count": len(evidence_chunks),
            }
        )

        logger.info(
            f"Stage 4 [Generation]: Status='{response.status.value}' | Total Time={total_duration:.2f}s | "
            f"Evidence Citations={len(response.evidence)}"
        )
        logger.info("=" * 80)
        return response
