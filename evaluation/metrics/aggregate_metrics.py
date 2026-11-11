"""Aggregate metrics calculation across overall dataset, query categories, and retrieval routes."""

from typing import Any, Dict, List
import numpy as np


def _mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def aggregate_overall_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall summary metrics across all processed evaluation records."""
    total = len(records)
    if total == 0:
        return {}

    precisions = [r["evaluation"]["context_precision"] for r in records if r.get("evaluation", {}).get("context_precision") is not None]
    recalls = [r["evaluation"]["context_recall"] for r in records if r.get("evaluation", {}).get("context_recall") is not None]
    faithfulness = [r["evaluation"]["faithfulness"] for r in records if r.get("evaluation", {}).get("faithfulness") is not None]
    relevancy = [r["evaluation"]["answer_relevancy"] for r in records if r.get("evaluation", {}).get("answer_relevancy") is not None]
    correctness = [r["evaluation"]["answer_correctness"] for r in records if r.get("evaluation", {}).get("answer_correctness") is not None]

    routing_latencies = [r["routing"]["routing_latency"] for r in records if r.get("routing", {}).get("routing_latency") is not None]
    retrieval_latencies = [r["retrieval"]["retrieval_latency"] for r in records if r.get("retrieval", {}).get("retrieval_latency") is not None]
    generation_latencies = [r["generation"]["generation_latency"] for r in records if r.get("generation", {}).get("generation_latency") is not None]
    total_latencies = [r["performance"]["total_latency"] for r in records if r.get("performance", {}).get("total_latency") is not None]

    successful_queries = sum(1 for r in records if r.get("execution_status") == "EVALUATION_COMPLETED")
    failed_queries = sum(1 for r in records if r.get("execution_status") in ("FAILED", "RATE_LIMITED"))

    return {
        "total_queries": total,
        "successful_queries": successful_queries,
        "failed_queries": failed_queries,
        "retrieval_metrics": {
            "mean_context_precision": _mean(precisions),
            "mean_context_recall": _mean(recalls),
        },
        "generation_metrics": {
            "mean_faithfulness": _mean(faithfulness),
            "mean_answer_relevancy": _mean(relevancy),
            "mean_answer_correctness": _mean(correctness),
        },
        "latency_metrics": {
            "mean_routing_latency": _mean(routing_latencies),
            "mean_retrieval_latency": _mean(retrieval_latencies),
            "mean_generation_latency": _mean(generation_latencies),
            "mean_total_latency": _mean(total_latencies),
        },
    }


def aggregate_category_metrics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute evaluation metrics grouped by query category."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        cat = r.get("category", "Unknown")
        grouped.setdefault(cat, []).append(r)

    results = []
    for cat, items in sorted(grouped.items()):
        precisions = [r["evaluation"]["context_precision"] for r in items if r.get("evaluation", {}).get("context_precision") is not None]
        recalls = [r["evaluation"]["context_recall"] for r in items if r.get("evaluation", {}).get("context_recall") is not None]
        faith = [r["evaluation"]["faithfulness"] for r in items if r.get("evaluation", {}).get("faithfulness") is not None]
        rel = [r["evaluation"]["answer_relevancy"] for r in items if r.get("evaluation", {}).get("answer_relevancy") is not None]
        corr = [r["evaluation"]["answer_correctness"] for r in items if r.get("evaluation", {}).get("answer_correctness") is not None]
        r_lat = [r["retrieval"]["retrieval_latency"] for r in items if r.get("retrieval", {}).get("retrieval_latency") is not None]
        g_lat = [r["generation"]["generation_latency"] for r in items if r.get("generation", {}).get("generation_latency") is not None]

        results.append({
            "category": cat,
            "query_count": len(items),
            "context_precision": _mean(precisions),
            "context_recall": _mean(recalls),
            "faithfulness": _mean(faith),
            "answer_relevancy": _mean(rel),
            "answer_correctness": _mean(corr),
            "avg_retrieval_latency": _mean(r_lat),
            "avg_generation_latency": _mean(g_lat),
        })
    return results


def aggregate_strategy_metrics(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze retrieval strategy breakdown (Dense vs Keyword vs Hybrid)."""
    strategies: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        strat = r.get("routing", {}).get("selected_strategy") or "unknown"
        strategies.setdefault(strat, []).append(r)

    results = []
    for strat, items in sorted(strategies.items()):
        precisions = [r["evaluation"]["context_precision"] for r in items if r.get("evaluation", {}).get("context_precision") is not None]
        recalls = [r["evaluation"]["context_recall"] for r in items if r.get("evaluation", {}).get("context_recall") is not None]
        corr = [r["evaluation"]["answer_correctness"] for r in items if r.get("evaluation", {}).get("answer_correctness") is not None]
        latencies = [r["retrieval"]["retrieval_latency"] for r in items if r.get("retrieval", {}).get("retrieval_latency") is not None]
        success_count = sum(1 for r in items if r.get("retrieval", {}).get("candidate_count", 0) > 0)

        results.append({
            "strategy": strat,
            "query_count": len(items),
            "retrieval_success_rate": float(success_count / len(items)) if items else 0.0,
            "context_precision": _mean(precisions),
            "context_recall": _mean(recalls),
            "answer_correctness": _mean(corr),
            "avg_retrieval_latency": _mean(latencies),
        })
    return results

