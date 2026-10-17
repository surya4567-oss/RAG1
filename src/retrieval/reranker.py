"""Local Cross-Encoder Reranker using BAAI/bge-reranker-base."""

import gc
from typing import List, Optional
import torch

from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.schemas import CandidateChunk


class LocalCrossEncoderReranker:
    """Reranker using BAAI/bge-reranker-base with CUDA/CPU support and batch scoring."""

    _instance = None
    _model = None

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or default_retrieval_config
        self.model_name = self.config.reranker_model
        self.batch_size = max(1, self.config.reranker_batch_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        """Lazy load the sentence_transformers CrossEncoder once."""
        if LocalCrossEncoderReranker._model is not None:
            return LocalCrossEncoderReranker._model

        from sentence_transformers import CrossEncoder

        logger.info(
            f"Loading CrossEncoder model '{self.model_name}' on device '{self.device}', "
            f"batch size {self.batch_size}"
        )

        model = CrossEncoder(
            self.model_name,
            max_length=512,
            device=self.device,
        )
        LocalCrossEncoderReranker._model = model
        logger.info(f"CrossEncoder model '{self.model_name}' loaded successfully.")
        return model

    def rerank(
        self,
        query: str,
        candidates: List[CandidateChunk],
        top_k: Optional[int] = None,
    ) -> List[CandidateChunk]:
        """Score each (query, chunk_text) pair and return candidates sorted by reranker score.

        Args:
            query: The user's original question.
            candidates: Retrieved candidate chunks (from dense, sparse, or hybrid).
            top_k: Number of top reranked chunks to return (default: config.final_evidence_top_k).

        Returns:
            List of CandidateChunk reordered by descending reranker score.
        """
        final_k = top_k or self.config.final_evidence_top_k
        if not candidates or not query.strip():
            return []

        # Bound candidates to reranker_top_k limit (default 50)
        pool = candidates[: self.config.reranker_top_k]

        try:
            model = self._load_model()
            pairs = [[query, c.text] for c in pool]

            # CrossEncoder predicts raw relevance logits
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            # Attach rerank score and preserve original scores
            reranked_pool: List[CandidateChunk] = []
            for candidate, score in zip(pool, raw_scores):
                updated_chunk = CandidateChunk(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    score=float(score),  # New primary score is rerank score
                    original_rank=candidate.original_rank,
                    retrieval_route=candidate.retrieval_route,
                    metadata=dict(candidate.metadata),
                    dense_rank=candidate.dense_rank,
                    sparse_rank=candidate.sparse_rank,
                    rrf_score=candidate.rrf_score,
                    rerank_score=round(float(score), 6),
                )
                reranked_pool.append(updated_chunk)

            # Sort descending by rerank_score
            reranked_pool.sort(key=lambda x: x.rerank_score or -999.0, reverse=True)

            logger.info(
                f"Reranked {len(pool)} candidates -> selected top {min(final_k, len(reranked_pool))} "
                f"evidence chunks (top score: {reranked_pool[0].rerank_score if reranked_pool else 'N/A'})"
            )
            return reranked_pool[:final_k]

        except Exception as e:
            logger.error(f"CrossEncoder reranking failed: {e}. Falling back to pre-rerank candidate ordering.")
            return pool[:final_k]
