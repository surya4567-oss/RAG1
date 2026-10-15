"""Resets (deletes and re-creates) a Qdrant collection with explicit confirmation."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.config import PipelineConfig
from src.ingestion.vector_store import QdrantManager


def main():
    parser = argparse.ArgumentParser(
        description="Reset a Qdrant collection. WARNING: This will delete all indexed points."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name to reset (defaults to config).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Explicit confirmation flag required to proceed with reset.",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    if args.collection:
        config.qdrant_collection = args.collection

    if not args.confirm:
        print("\n⚠️ WARNING: Resetting will PERMANENTLY DELETE all points in the collection.")
        print(f"Target collection: {config.qdrant_collection}")
        print("To proceed, you must pass the --confirm flag:")
        print(f"  python scripts/reset_collection.py --confirm")
        sys.exit(1)

    manager = QdrantManager(config)
    print(f"Resetting collection '{manager.collection_name}'...")
    manager.reset_collection(confirm=True)
    print("✅ Collection reset successfully and re-initialized with 768-dim Cosine vectors.")


if __name__ == "__main__":
    main()
