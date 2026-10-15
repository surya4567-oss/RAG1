"""Inspects Qdrant collection status, point counts, vector config, and queries."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.config import PipelineConfig
from src.ingestion.embeddings import LocalBGEEmbeddings
from src.ingestion.vector_store import QdrantManager


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Qdrant collection configuration, point count, and run sample queries."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name to inspect (defaults to config).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="permission error access denied",
        help="Search query to test retrieval against the collection.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of search hits to display.",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    if args.collection:
        config.qdrant_collection = args.collection

    manager = QdrantManager(config)
    client = manager.client

    print("\n" + "=" * 80)
    print(f"INSPECTING QDRANT COLLECTION: {manager.collection_name}")
    print("=" * 80)

    try:
        collections = [c.name for c in client.get_collections().collections]
        if manager.collection_name not in collections:
            print(f"❌ Collection '{manager.collection_name}' DOES NOT EXIST in Qdrant.")
            print(f"Available collections: {collections}")
            return

        info = client.get_collection(manager.collection_name)
        print(f"Points count: {info.points_count}")
        print(f"Vectors config: {info.config.params.vectors}")
        print(f"Status: {info.status}")

        if (info.points_count or 0) > 0:
            print("\n--- SAMPLE POINT PAYLOAD ---")
            points, _ = client.scroll(
                collection_name=manager.collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if points:
                sample_pt = points[0]
                print(f"Point ID: {sample_pt.id}")
                print("Payload fields:")
                for k, v in sample_pt.payload.items():
                    if k in ("page_content", "text"):
                        print(f"  {k}: {str(v)[:100]}... [truncated]")
                    else:
                        print(f"  {k}: {v}")

            print(f"\n--- SMOKE TEST SEARCH: '{args.query}' ---")
            embeddings = LocalBGEEmbeddings(config)
            q_vec = embeddings.embed_query(args.query)

            if hasattr(client, "query_points"):
                hits = client.query_points(
                    collection_name=manager.collection_name,
                    query=q_vec,
                    limit=args.limit,
                    with_payload=True,
                ).points
            elif hasattr(client, "search"):
                hits = client.search(
                    collection_name=manager.collection_name,
                    query_vector=q_vec,
                    limit=args.limit,
                    with_payload=True,
                )
            else:
                hits = []

            print(f"Found {len(hits)} hits:")
            for i, hit in enumerate(hits, 1):
                print(f"\n[{i}] Hit ID: {hit.id} | Score: {hit.score:.4f}")
                print(f"    Document Type: {hit.payload.get('document_type')}")
                print(f"    Record ID:     {hit.payload.get('record_id')}")
                print(f"    Source:        {hit.payload.get('source')}")
                snippet = (hit.payload.get('page_content') or '')[:160].replace('\n', ' ')
                print(f"    Snippet:       {snippet}...")
        else:
            print("\n⚠️ Collection is currently empty (0 points). Run ingestion to index records.")

    except Exception as e:
        print(f"❌ Error inspecting collection: {e}")


if __name__ == "__main__":
    main()
