"""Hybrid retriever combining dense semantic search and sparse BM25 search via RRF."""

from typing import Any, Dict, List, Optional
from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.schemas import CandidateChunk


class HybridRetriever:
    """Orchestrates multi-modal dense and sparse retrieval with rank-based fusion."""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        dense_retriever: Optional[DenseRetriever] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
    ):
        self.config = config or default_retrieval_config
        self.dense_retriever = dense_retriever or DenseRetriever(self.config)
        self.keyword_retriever = keyword_retriever or KeywordRetriever(self.config)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateChunk]:
        """Execute both dense and sparse retrieval, fuse ranks with RRF, and deduplicate."""
        target_top_k = top_k or self.config.reranker_top_k
        dense_top_k = self.config.dense_top_k
        sparse_top_k = self.config.sparse_top_k

        dense_candidates: List[CandidateChunk] = []
        sparse_candidates: List[CandidateChunk] = []
        dense_failed = False
        sparse_failed = False

        # 1. Execute Dense Retrieval
        try:
            dense_candidates = self.dense_retriever.retrieve(
                query=query,
                top_k=dense_top_k,
                filters=filters,
            )
        except Exception as e:
            dense_failed = True
            logger.error(f"Hybrid retrieval: dense branch failed with error: {e}")

        # 2. Execute Sparse / Keyword Retrieval
        try:
            sparse_candidates = self.keyword_retriever.retrieve(
                query=query,
                top_k=sparse_top_k,
                filters=filters,
            )
        except Exception as e:
            sparse_failed = True
            logger.error(f"Hybrid retrieval: sparse branch failed with error: {e}")

        # 3. Handle Degradation Cases
        if dense_failed and sparse_failed:
            raise RuntimeError("Both dense and sparse retrieval failed in hybrid retrieval pipeline.")

        if dense_failed:
            logger.warning("Hybrid degraded: returning sparse-only candidates due to dense branch failure.")
            return sparse_candidates[:target_top_k]

        if sparse_failed:
            logger.warning("Hybrid degraded: returning dense-only candidates due to sparse branch failure.")
            return dense_candidates[:target_top_k]

        if not dense_candidates and not sparse_candidates:
            logger.info("Hybrid retrieval returned 0 candidates from both dense and sparse branches.")
            return []

        # 4. Fuse using Reciprocal Rank Fusion (k=60)
        fused_candidates = reciprocal_rank_fusion(
            dense_candidates=dense_candidates,
            sparse_candidates=sparse_candidates,
            k=self.config.rrf_k,
            top_n=target_top_k,
        )

        logger.info(
            f"Hybrid retrieval: fused {len(dense_candidates)} dense + {len(sparse_candidates)} sparse "
            f"into {len(fused_candidates)} unique candidates (top_n={target_top_k})"
        )
        return fused_candidates
