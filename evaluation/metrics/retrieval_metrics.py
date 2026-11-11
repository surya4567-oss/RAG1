"""Retrieval metrics evaluation using official RAGAS metrics alone."""

from typing import Any, Dict, List, Optional
from ragas import SingleTurnSample
from ragas.metrics import ContextPrecision, ContextRecall

from evaluation.metrics.ragas_factory import get_ragas_llm
from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config


class RagasRetrievalEvaluator:
    """Evaluates retrieval quality strictly using RAGAS ContextPrecision and ContextRecall."""

    def __init__(self, config: Optional[RetrievalConfig] = None) -> None:
        self.config = config or default_retrieval_config
        self.context_precision_metric = ContextPrecision()
        self.context_recall_metric = ContextRecall()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            try:
                ragas_llm = get_ragas_llm(self.config)
                self.context_precision_metric.llm = ragas_llm
                self.context_recall_metric.llm = ragas_llm
                self._initialized = True
            except Exception as e:
                logger.error("Failed to initialize Ragas LLM for retrieval metrics: %s", e)

    def evaluate(
        self,
        query: str,
        retrieved_contexts: List[str],
        expected_answer: str,
    ) -> Dict[str, Any]:
        """Compute RAGAS Context Precision and Context Recall for a query.

        Returns:
            Dict containing:
                - 'context_precision': Optional[float] (None if evaluation failed)
                - 'context_recall': Optional[float] (None if evaluation failed)
                - 'metric_status': Dict[str, str] ('SUCCESS' | 'ERROR')
                - 'metric_errors': Dict[str, str]
        """
        metric_status: Dict[str, str] = {}
        metric_errors: Dict[str, str] = {}

        if not retrieved_contexts:
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "metric_status": {"context_precision": "SUCCESS", "context_recall": "SUCCESS"},
                "metric_errors": {},
            }

        self._ensure_initialized()
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=retrieved_contexts,
            reference=expected_answer,
        )

        precision_score: Optional[float] = None
        try:
            raw_prec = self.context_precision_metric.single_turn_score(sample)
            precision_score = max(0.0, min(1.0, float(raw_prec)))
            metric_status["context_precision"] = "SUCCESS"
        except Exception as e:
            logger.warning("Ragas ContextPrecision scoring failed: %s", e)
            precision_score = None
            metric_status["context_precision"] = "ERROR"
            metric_errors["context_precision"] = str(e)

        recall_score: Optional[float] = None
        try:
            raw_rec = self.context_recall_metric.single_turn_score(sample)
            recall_score = max(0.0, min(1.0, float(raw_rec)))
            metric_status["context_recall"] = "SUCCESS"
        except Exception as e:
            logger.warning("Ragas ContextRecall scoring failed: %s", e)
            recall_score = None
            metric_status["context_recall"] = "ERROR"
            metric_errors["context_recall"] = str(e)

        return {
            "context_precision": precision_score,
            "context_recall": recall_score,
            "metric_status": metric_status,
            "metric_errors": metric_errors,
        }


# Global singleton instance for standalone function calls
_retrieval_evaluator: Optional[RagasRetrievalEvaluator] = None


def get_ragas_retrieval_evaluator(config: Optional[RetrievalConfig] = None) -> RagasRetrievalEvaluator:
    global _retrieval_evaluator
    if _retrieval_evaluator is None:
        _retrieval_evaluator = RagasRetrievalEvaluator(config=config)
    return _retrieval_evaluator


def compute_context_precision(
    retrieved_contexts: List[str],
    expected_answer: str,
    query: str = "",
    config: Optional[RetrievalConfig] = None,
) -> Optional[float]:
    """Calculate RAGAS Context Precision."""
    evaluator = get_ragas_retrieval_evaluator(config)
    res = evaluator.evaluate(query=query, retrieved_contexts=retrieved_contexts, expected_answer=expected_answer)
    return res["context_precision"]


def compute_context_recall(
    retrieved_contexts: List[str],
    expected_answer: str,
    query: str = "",
    config: Optional[RetrievalConfig] = None,
) -> Optional[float]:
    """Calculate RAGAS Context Recall."""
    evaluator = get_ragas_retrieval_evaluator(config)
    res = evaluator.evaluate(query=query, retrieved_contexts=retrieved_contexts, expected_answer=expected_answer)
    return res["context_recall"]
