"""Generation Evaluator wrapping faithfulness, relevancy, and correctness metrics."""

from typing import Any, Dict, List, Optional
from evaluation.metrics.generation_metrics import GenerationMetricsEvaluator
from src.retrieval.config import RetrievalConfig


class GenerationEvaluator:
    """Evaluates answer generation quality."""

    def __init__(self, config: Optional[RetrievalConfig] = None) -> None:
        self.metrics_evaluator = GenerationMetricsEvaluator(config=config)

    def evaluate(
        self,
        query: str,
        generated_answer: Optional[str],
        expected_answer: str,
        retrieved_contexts: List[str],
    ) -> Dict[str, Any]:
        """Compute generation metrics for a query response."""
        return self.metrics_evaluator.evaluate_generation(
            query=query,
            generated_answer=generated_answer,
            expected_answer=expected_answer,
            retrieved_contexts=retrieved_contexts,
        )

