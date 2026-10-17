"""CLI interface to execute queries through the Advanced Retrieval Pipeline."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.retrieval.pipeline import AdvancedRetrievalPipeline
from src.retrieval.schemas import RetrievalRoute


def display_response(res, output_json: bool = False):
    """Render retrieval response nicely to console."""
    if output_json:
        print(res.model_dump_json(indent=2))
        return

    print("\n" + "=" * 80)
    print(f"QUESTION: {res.question}")
    print("=" * 80)
    print(f"Status:          [{res.status.value.upper()}]")
    print(f"Retrieval Route: [{res.retrieval_route.value.upper()}]")
    if "total_duration_seconds" in res.metadata:
        m = res.metadata
        print(
            f"Latency:         Total={m['total_duration_seconds']}s "
            f"(Route={m['routing_duration_seconds']}s, Retrieve={m['retrieval_duration_seconds']}s, "
            f"Rerank={m['reranker_duration_seconds']}s, Gen={m['generation_duration_seconds']}s)"
        )
    if "router_intent" in res.metadata:
        print(f"Router Intent:   {res.metadata['router_intent']}")
    if res.metadata.get("applied_filters"):
        print(f"Applied Filters: {res.metadata['applied_filters']}")
    print("-" * 80)
    print(f"ANSWER:\n{res.answer}\n")

    if res.evidence:
        print("-" * 80)
        print(f"EVIDENCE CITATIONS ({len(res.evidence)} chunks):")
        for idx, ev in enumerate(res.evidence, start=1):
            rec = f" | Record: {ev.record_id}" if ev.record_id else ""
            src = f" | Source: {ev.source_file}" if ev.source_file else ""
            print(f"\n[{idx}] Chunk ID: {ev.chunk_id} ({ev.document_type}{rec}{src})")
            print(f"    Excerpt: \"{ev.supporting_text.strip()}\"")

    if res.limitations:
        print("\n" + "-" * 80)
        print("LIMITATIONS / UNCERTAINTIES:")
        for lim in res.limitations:
            print(f"  • {lim}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run query through the Advanced Retrieval Pipeline."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Question to query against the QA knowledge base.",
    )
    parser.add_argument(
        "--route",
        type=str,
        choices=["keyword", "dense", "hybrid"],
        default=None,
        help="Override the LLM router with an explicit strategy.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print the raw validated JSON output.",
    )

    args = parser.parse_args()

    pipeline = AdvancedRetrievalPipeline()

    if args.query:
        override = RetrievalRoute(args.route) if args.route else None
        res = pipeline.run(question=args.query, override_strategy=override)
        display_response(res, output_json=args.json)
    else:
        print("\n" + "=" * 80)
        print("VANILLA RAG ADVANCED RETRIEVAL INTERACTIVE CLI")
        print("Type 'exit' or 'quit' to stop.")
        print("=" * 80)
        while True:
            try:
                user_q = input("\nEnter your question: ").strip()
                if not user_q or user_q.lower() in ("exit", "quit"):
                    break
                override = RetrievalRoute(args.route) if args.route else None
                res = pipeline.run(question=user_q, override_strategy=override)
                display_response(res, output_json=args.json)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
