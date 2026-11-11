"""Chart generation for evaluation results using Matplotlib."""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from evaluation.metrics.aggregate_metrics import (
    aggregate_category_metrics,
    aggregate_overall_metrics,
    aggregate_strategy_metrics,
)


def generate_evaluation_charts(
    records: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """Generate PNG charts visualizing evaluation metrics, category performance, and latency."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_charts: Dict[str, Path] = {}

    if not records:
        return generated_charts

    overall = aggregate_overall_metrics(records)
    category_metrics = aggregate_category_metrics(records)
    strategy_metrics = aggregate_strategy_metrics(records)

    # 1. Overall Metrics Bar Chart
    try:
        metrics_names = ["Precision", "Recall", "Faithfulness", "Relevancy", "Correctness"]
        metrics_vals = [
            overall["retrieval_metrics"]["mean_context_precision"],
            overall["retrieval_metrics"]["mean_context_recall"],
            overall["generation_metrics"]["mean_faithfulness"],
            overall["generation_metrics"]["mean_answer_relevancy"],
            overall["generation_metrics"]["mean_answer_correctness"],
        ]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(metrics_names, metrics_vals, color=["#2b5c8f", "#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"])
        plt.ylim(0, 1.1)
        plt.title("VanillaRAG Core Evaluation Metrics (Overall Mean)", fontsize=13, pad=12)
        plt.ylabel("Score (0.0 - 1.0)", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            plt.annotate(f"{height:.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=10, weight="bold")

        p1 = output_dir / "metric_comparison.png"
        plt.tight_layout()
        plt.savefig(p1, dpi=300)
        plt.close()
        generated_charts["metric_comparison"] = p1
    except Exception as e:
        plt.close()

    # 2. Category Performance Comparison Chart
    try:
        if category_metrics:
            categories = [c["category"] for c in category_metrics]
            precisions = [c["context_precision"] for c in category_metrics]
            correctness = [c["answer_correctness"] for c in category_metrics]

            x = np.arange(len(categories))
            width = 0.35

            plt.figure(figsize=(12, 6))
            plt.bar(x - width/2, precisions, width, label="Context Precision", color="#3b82f6")
            plt.bar(x + width/2, correctness, width, label="Answer Correctness", color="#10b981")

            plt.ylabel("Score (0.0 - 1.0)", fontsize=11)
            plt.title("Performance by Query Category", fontsize=13, pad=12)
            plt.xticks(x, categories, rotation=45, ha="right", fontsize=9)
            plt.ylim(0, 1.1)
            plt.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.4)

            p2 = output_dir / "category_performance.png"
            plt.tight_layout()
            plt.savefig(p2, dpi=300)
            plt.close()
            generated_charts["category_performance"] = p2
    except Exception as e:
        plt.close()

    # 3. Retrieval Strategy Distribution Pie Chart
    try:
        if strategy_metrics:
            labels = [s["strategy"].upper() for s in strategy_metrics]
            counts = [s["query_count"] for s in strategy_metrics]
            colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"][:len(labels)]

            plt.figure(figsize=(7, 6))
            plt.pie(counts, labels=labels, autopct="%1.1f%%", colors=colors, startangle=140, explode=[0.05]*len(labels))
            plt.title("Dynamic Retrieval Strategy Distribution", fontsize=13, pad=12)

            p3 = output_dir / "retrieval_strategy_distribution.png"
            plt.tight_layout()
            plt.savefig(p3, dpi=300)
            plt.close()
            generated_charts["strategy_distribution"] = p3
    except Exception as e:
        plt.close()

    # 4. Latency Breakdown Chart
    try:
        routing_lats = [r["routing"]["routing_latency"] for r in records if r.get("routing", {}).get("routing_latency") is not None]
        retrieval_lats = [r["retrieval"]["retrieval_latency"] for r in records if r.get("retrieval", {}).get("retrieval_latency") is not None]
        gen_lats = [r["generation"]["generation_latency"] for r in records if r.get("generation", {}).get("generation_latency") is not None]

        if routing_lats and retrieval_lats and gen_lats:
            stages = ["Query Routing", "Adaptive Retrieval", "Answer Generation"]
            means = [float(np.mean(routing_lats)), float(np.mean(retrieval_lats)), float(np.mean(gen_lats))]

            plt.figure(figsize=(8, 5))
            bars = plt.bar(stages, means, color=["#6366f1", "#06b6d4", "#ec4899"])
            plt.ylabel("Mean Latency (seconds)", fontsize=11)
            plt.title("Average Execution Latency by Stage", fontsize=13, pad=12)
            plt.grid(axis="y", linestyle="--", alpha=0.5)

            for bar in bars:
                height = bar.get_height()
                plt.annotate(f"{height:.2f}s",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha="center", va="bottom", fontsize=10, weight="bold")

            p4 = output_dir / "latency_breakdown.png"
            plt.tight_layout()
            plt.savefig(p4, dpi=300)
            plt.close()
            generated_charts["latency_breakdown"] = p4
    except Exception as e:
        plt.close()

    return generated_charts

