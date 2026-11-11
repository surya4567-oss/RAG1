"""Performance and latency evaluator."""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from evaluation.utils.timing_utils import compute_latency_percentiles


@dataclass
class PerformanceEvaluator:
    """Collects and aggregates latency profiles across all pipeline stages."""

    routing_latencies: List[float] = field(default_factory=list)
    retrieval_latencies: List[float] = field(default_factory=list)
    reranking_latencies: List[float] = field(default_factory=list)
    generation_latencies: List[float] = field(default_factory=list)
    evaluation_latencies: List[float] = field(default_factory=list)
    total_latencies: List[float] = field(default_factory=list)

    def record_query_performance(
        self,
        routing_sec: float,
        retrieval_sec: float,
        reranking_sec: float,
        generation_sec: float,
        evaluation_sec: float,
        total_sec: float,
    ) -> None:
        self.routing_latencies.append(routing_sec)
        self.retrieval_latencies.append(retrieval_sec)
        self.reranking_latencies.append(reranking_sec)
        self.generation_latencies.append(generation_sec)
        self.evaluation_latencies.append(evaluation_sec)
        self.total_latencies.append(total_sec)

    def get_summary_statistics(self) -> Dict[str, Any]:
        return {
            "routing": compute_latency_percentiles(self.routing_latencies),
            "retrieval": compute_latency_percentiles(self.retrieval_latencies),
            "reranking": compute_latency_percentiles(self.reranking_latencies),
            "generation": compute_latency_percentiles(self.generation_latencies),
            "evaluation": compute_latency_percentiles(self.evaluation_latencies),
            "total_end_to_end": compute_latency_percentiles(self.total_latencies),
        }

