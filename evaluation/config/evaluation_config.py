"""Evaluation Configuration for VanillaRAG Evaluation Pipeline."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Optional


@dataclass
class EvaluationConfig:
    """Configurable parameters for evaluation execution, reliability, and reporting."""

    # Dataset Location
    dataset_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("EVAL_DATASET_PATH", "vanillarag_evaluation_dataset.json")
        ).resolve()
    )

    # Output Root Directory
    results_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("EVAL_RESULTS_DIR", "evaluation/results")
        ).resolve()
    )

    # Derived Output Subdirectories
    checkpoints_dir: Path = field(init=False)
    per_query_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    charts_dir: Path = field(init=False)

    # Primary checkpoint and per-query JSONL files
    checkpoint_file: Path = field(init=False)
    per_query_jsonl: Path = field(init=False)

    # Reliability and Retry Settings
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("EVAL_MAX_RETRIES", "3"))
    )
    retry_initial_delay: float = field(
        default_factory=lambda: float(os.getenv("EVAL_RETRY_INITIAL_DELAY", "1.0"))
    )
    retry_backoff_factor: float = field(
        default_factory=lambda: float(os.getenv("EVAL_RETRY_BACKOFF_FACTOR", "2.0"))
    )
    retry_max_delay: float = field(
        default_factory=lambda: float(os.getenv("EVAL_RETRY_MAX_DELAY", "30.0"))
    )
    retry_jitter: bool = field(
        default_factory=lambda: os.getenv("EVAL_RETRY_JITTER", "true").lower() == "true"
    )

    # Judge / Evaluation LLM Settings
    judge_role: str = "judge"
    judge_temperature: float = 0.0
    judge_max_tokens: int = 512

    # Retrieval Evaluation Thresholds
    min_retrieval_relevance_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Initialize derived directory paths and ensure directories exist."""
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.per_query_dir = self.results_dir / "per_query"
        self.metrics_dir = self.results_dir / "metrics"
        self.reports_dir = self.results_dir / "reports"
        self.charts_dir = self.results_dir / "charts"

        self.checkpoint_file = self.checkpoints_dir / "checkpoint_state.jsonl"
        self.per_query_jsonl = self.per_query_dir / "per_query_results.jsonl"

    def ensure_directories(self) -> None:
        """Create all required output subdirectories."""
        for d in [
            self.results_dir,
            self.checkpoints_dir,
            self.per_query_dir,
            self.metrics_dir,
            self.reports_dir,
            self.charts_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


default_eval_config = EvaluationConfig()

