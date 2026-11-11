"""Retrieval Evaluator for candidate contexts and ranking quality."""

from typing import Any, Dict, List, Optional
from evaluation.metrics.retrieval_metrics import get_ragas_retrieval_evaluator
from src.retrieval.schemas import CandidateChunk


class RetrievalEvaluator:
    """Evaluates retrieval outputs against query and expected answer."""

    def __init__(self, relevance_threshold: float = 0.5) -> None:
        self.relevance_threshold = relevance_threshold

    def evaluate(
        self,
        retrieved_chunks: List[CandidateChunk],
        expected_answer: str,
        query: str = "",
    ) -> Dict[str, Any]:
        """Assess precision, recall, and candidate counts for retrieved evidence using RAGAS."""
        contexts = [c.text for c in retrieved_chunks]
        evaluator = get_ragas_retrieval_evaluator()
        ragas_res = evaluator.evaluate(
            query=query,
            retrieved_contexts=contexts,
            expected_answer=expected_answer,
        )

        return {
            "context_precision": ragas_res.get("context_precision"),
            "context_recall": ragas_res.get("context_recall"),
            "candidate_count": len(retrieved_chunks),
            "retrieved_chunk_ids": [c.chunk_id for c in retrieved_chunks],
            "retrieved_contexts": contexts,
            "metric_status": ragas_res.get("metric_status", {}),
            "metric_errors": ragas_res.get("metric_errors", {}),
        }
