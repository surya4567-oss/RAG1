"""
Evaluation runner for VanillaRAG benchmark dataset with persistent JSONL checkpointing.

Features:
- Checkpoints after every single query to allow seamless resume on Colab / local.
- Tracks full latency breakdown (total, routing, retrieval, reranking, generation).
- Evaluates evidence grounding and support confidence.
- Generates summary statistics by category and overall benchmark metrics.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.pipeline import AdvancedRetrievalPipeline
from src.retrieval.schemas import RetrievalStatus


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_checkpoints(checkpoint_file: str) -> Dict[str, Dict[str, Any]]:
    checkpoints = {}
    path = Path(checkpoint_file)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        qid = record.get("id") or record.get("query_id")
                        if qid:
                            checkpoints[qid] = record
                    except json.JSONDecodeError:
                        continue
    return checkpoints


def run_evaluation(
    dataset_file: str = "vanillarag_evaluation_dataset.json",
    checkpoint_file: str = "evaluation_results.jsonl",
    summary_file: str = "evaluation_summary.json",
    limit: Optional[int] = None,
    delay_between_calls: float = 1.0,
    resume: bool = True,
):
    dataset = load_dataset(dataset_file)
    print(f"Loaded {len(dataset)} evaluation queries from '{dataset_file}'")

    checkpoints = load_checkpoints(checkpoint_file) if resume else {}
    print(f"Loaded {len(checkpoints)} completed checkpoints from '{checkpoint_file}'")

    pipeline = AdvancedRetrievalPipeline()

    cp_path = Path(checkpoint_file)
    cp_path.parent.mkdir(parents=True, exist_ok=True)

    evaluated_count = len(checkpoints)
    total_to_run = min(len(dataset), limit) if limit else len(dataset)

    with open(cp_path, "a", encoding="utf-8") as cp_out:
        for idx, item in enumerate(dataset, start=1):
            if limit and idx > limit:
                break

            qid = item.get("id") or f"Q{idx:03d}"
            query_text = item.get("query") or item.get("question") or ""
            expected_answer = item.get("expected_answer") or ""
            category = item.get("category") or "General"

            if resume and qid in checkpoints:
                continue

            print(f"\n[{idx}/{total_to_run}] Evaluating {qid} ({category}): {query_text[:70]}...")

            start_t = time.perf_counter()
            try:
                response = pipeline.run(question=query_text)
                elapsed = time.perf_counter() - start_t

                meta = response.metadata or {}
                evidence_list = [
                    {
                        "chunk_id": ev.chunk_id,
                        "document_type": ev.document_type,
                        "record_id": ev.record_id,
                        "source_file": ev.source_file,
                        "is_supported": getattr(ev, "is_supported", True),
                        "support_confidence": getattr(ev, "support_confidence", 1.0),
                    }
                    for ev in response.evidence
                ]

                record = {
                    "id": qid,
                    "query": query_text,
                    "category": category,
                    "expected_answer": expected_answer,
                    "generated_answer": response.answer,
                    "status": response.status.value,
                    "retrieval_route": response.retrieval_route.value,
                    "evidence_count": len(response.evidence),
                    "evidence": evidence_list,
                    "limitations": response.limitations,
                    "total_latency": meta.get("total_duration_seconds") or meta.get("total_latency") or round(elapsed, 3),
                    "routing_latency": meta.get("routing_duration_seconds", 0.0),
                    "retrieval_latency": meta.get("retrieval_duration_seconds", 0.0),
                    "generation_latency": meta.get("generation_duration_seconds", 0.0),
                    "rerank_latency": meta.get("reranker_duration_seconds", 0.0),
                    "applied_filters": meta.get("applied_filters", {}),
                    "filter_fallback_applied": meta.get("filter_fallback_applied", False),
                    "fallback_type": meta.get("fallback_type"),
                    "quality_status": meta.get("quality_status"),
                    "error": meta.get("error"),
                }
            except Exception as e:
                elapsed = time.perf_counter() - start_t
                print(f"❌ Error processing query {qid}: {e}")
                record = {
                    "id": qid,
                    "query": query_text,
                    "category": category,
                    "expected_answer": expected_answer,
                    "generated_answer": f"Execution error: {e}",
                    "status": "error",
                    "retrieval_route": "unknown",
                    "evidence_count": 0,
                    "evidence": [],
                    "limitations": [str(e)],
                    "total_latency": round(elapsed, 3),
                    "error": str(e),
                }

            # Checkpoint immediately to disk
            cp_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            cp_out.flush()
            checkpoints[qid] = record
            evaluated_count += 1

            print(
                f"   Done -> Status: [{record['status']}] | Route: [{record.get('retrieval_route')}] | "
                f"Evidence: {record['evidence_count']} | Latency: {record['total_latency']}s"
            )

            if delay_between_calls > 0:
                time.sleep(delay_between_calls)

    # Generate benchmark summary
    all_records = list(checkpoints.values())
    summary = compute_summary_statistics(all_records)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("EVALUATION BENCHMARK COMPLETED")
    print("=" * 80)
    print(f"Total Evaluated: {summary['total_queries']}")
    print(f"Status Breakdown: {summary['status_counts']}")
    print(f"Average Total Latency: {summary['average_latency_seconds']:.3f}s")
    print(f"Summary saved to: '{summary_file}'")
    print("=" * 80 + "\n")


def compute_summary_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"total_queries": 0}

    total = len(records)
    status_counts: Dict[str, int] = {}
    category_stats: Dict[str, Dict[str, Any]] = {}
    latencies: List[float] = []

    for r in records:
        st = r.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

        cat = r.get("category", "General")
        if cat not in category_stats:
            category_stats[cat] = {
                "count": 0,
                "answered": 0,
                "insufficient_evidence": 0,
                "generation_failed": 0,
                "retrieval_only": 0,
                "avg_latency": 0.0,
                "_latencies": [],
            }
        category_stats[cat]["count"] += 1
        if st == "answered":
            category_stats[cat]["answered"] += 1
        elif st == "insufficient_evidence":
            category_stats[cat]["insufficient_evidence"] += 1
        elif st == "generation_failed":
            category_stats[cat]["generation_failed"] += 1
        elif st == "retrieval_only":
            category_stats[cat]["retrieval_only"] += 1

        lat = r.get("total_latency")
        if isinstance(lat, (int, float)):
            latencies.append(float(lat))
            category_stats[cat]["_latencies"].append(float(lat))

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    for cat, data in category_stats.items():
        cat_lats = data.pop("_latencies")
        data["avg_latency"] = round(sum(cat_lats) / len(cat_lats), 3) if cat_lats else 0.0

    return {
        "total_queries": total,
        "status_counts": status_counts,
        "average_latency_seconds": round(avg_latency, 3),
        "category_breakdown": category_stats,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VanillaRAG on benchmark queries.")
    parser.add_argument("--dataset", type=str, default="vanillarag_evaluation_dataset.json")
    parser.add_argument("--checkpoints", type=str, default="evaluation_results.jsonl")
    parser.add_argument("--summary", type=str, default="evaluation_summary.json")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to test")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay in seconds between LLM calls")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from existing checkpoints")

    args = parser.parse_args()
    run_evaluation(
        dataset_file=args.dataset,
        checkpoint_file=args.checkpoints,
        summary_file=args.summary,
        limit=args.limit,
        delay_between_calls=args.delay,
        resume=not args.no_resume,
    )

