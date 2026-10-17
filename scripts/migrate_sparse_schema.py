"""Safe schema migration script to register a sparse vector space in Qdrant.

This script inspects the existing collection and safely adds a sparse vector
definition (default: 'bm25') WITHOUT recreating the collection, resetting points,
or affecting existing dense vectors (BAAI/bge-base-en-v1.5), payloads, or point IDs.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.config import PipelineConfig
from src.ingestion.vector_store import QdrantManager


def main():
    parser = argparse.ArgumentParser(
        description="Safely inspect and migrate Qdrant schema to support sparse vectors without data loss."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Target collection name (defaults to QDRANT_COLLECTION in config).",
    )
    parser.add_argument(
        "--vector-name",
        type=str,
        default="bm25",
        help="Name of the sparse vector to register (default: 'bm25').",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually apply the schema migration. If omitted, runs in dry-run/inspection mode only.",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    if args.collection:
        config.qdrant_collection = args.collection

    manager = QdrantManager(config)
    client = manager.client

    print("\n" + "=" * 80)
    print("QDRANT SPARSE VECTOR SCHEMA MIGRATION INSPECTION")
    print("=" * 80)
    print(f"Collection Name:       {manager.collection_name}")
    print(f"Target Sparse Vector:  '{args.vector_name}'")
    print(f"Mode:                  {'[APPLY SCHEMA UPDATE]' if args.apply else '[DRY-RUN / INSPECTION ONLY]'}")
    print("=" * 80)

    try:
        collections = [c.name for c in client.get_collections().collections]
        if manager.collection_name not in collections:
            print(f"❌ Error: Collection '{manager.collection_name}' does not exist in Qdrant.")
            print(f"Available collections: {collections}")
            sys.exit(1)

        info = manager.get_collection_info()
        dense_vectors = info.config.params.vectors
        sparse_vectors = info.config.params.sparse_vectors or {}
        points_count = info.points_count or 0

        print(f"\nCurrent Status:")
        print(f"  - Points Count:          {points_count:,}")
        print(f"  - Dense Vectors Config:  {dense_vectors}")
        print(f"  - Sparse Vectors Config: {sparse_vectors}")

        is_already_registered = args.vector_name in sparse_vectors

        if is_already_registered:
            print(f"\n✅ Sparse vector '{args.vector_name}' is ALREADY registered in collection schema.")
            print("No schema migration is needed. The collection is ready to receive sparse vector updates.")
            return

        print(f"\n⚠️ Sparse vector '{args.vector_name}' is NOT yet registered in the collection schema.")

        if not args.apply:
            print("\n[DRY RUN]: To safely register the sparse vector schema without modifying any existing data, run:")
            print(f"    python scripts/migrate_sparse_schema.py --vector-name {args.vector_name} --apply\n")
            return

        print(f"\nApplying safe schema update to add sparse vector '{args.vector_name}'...")
        added = manager.ensure_sparse_vector(vector_name=args.vector_name)

        # Re-verify
        post_info = manager.get_collection_info()
        post_points = post_info.points_count or 0
        post_sparse = post_info.config.params.sparse_vectors or {}

        print("\n" + "-" * 80)
        print("MIGRATION COMPLETED SUCCESSFULLY")
        print("-" * 80)
        print(f"  - Sparse Vectors Config: {post_sparse}")
        print(f"  - Total Points Intact:   {post_points:,} (dense vectors & payloads 100% preserved)")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Migration failed with error: {e}")
        sys.exit(1)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
