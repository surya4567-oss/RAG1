"""CLI runner for the Vanilla RAG Ingestion Pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.config import PipelineConfig
from src.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run the Vanilla RAG Ingestion Pipeline with metadata awareness and QA traceability."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run discovery, extraction, validation, and chunking without indexing into Qdrant.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Process only a specific source category (e.g., testcases, test_execution, defects, srs, failure_logs, screenshots).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on records loaded per source (useful for testing and dry-runs).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Target Qdrant collection name (overrides .env / config).",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Remote Qdrant service URL (leave unset to use persistent embedded storage).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for embedding and indexing.",
    )

    args = parser.parse_args()

    # Build config
    config = PipelineConfig()
    config.dry_run = args.dry_run
    config.source_filter = args.source
    config.sample_limit = args.limit

    if args.collection:
        config.qdrant_collection = args.collection
    if args.qdrant_url:
        config.qdrant_url = args.qdrant_url
    if args.batch_size:
        config.index_batch_size = args.batch_size
        config.embedding_batch_size = min(32, args.batch_size)

    pipeline = IngestionPipeline(config=config)
    json_path, md_path = pipeline.run()

    print(f"\nIngestion complete!")
    print(f"JSON Report: {json_path}")
    print(f"Summary:     {md_path}")


if __name__ == "__main__":
    main()
