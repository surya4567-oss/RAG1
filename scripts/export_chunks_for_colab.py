"""Utility to export chunk IDs and texts from Qdrant to a lightweight JSONL file for Google Colab.

Exports only 'id' and text ('page_content' / 'text') without downloading dense vectors.
The resulting file is small (typically ~30-50MB for 120k chunks) and can be easily
uploaded to Google Drive / Colab to run sparse generation without transferring the 1.1GB database.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.config import PipelineConfig
from src.ingestion.logging_config import logger
from src.ingestion.vector_store import QdrantManager


def export_chunks(
    output_file: str = "data/chunks_for_colab.jsonl",
    collection_name: str = None,
    qdrant_path: str = None,
    batch_size: int = 500,
    limit: int = None,
):
    config = PipelineConfig()
    if collection_name:
        config.qdrant_collection = collection_name
    if qdrant_path:
        config.qdrant_path = qdrant_path
    elif not (Path(config.qdrant_path) / "collection" / config.qdrant_collection).exists():
        fallback = Path("data/data1/qdrant_db")
        if (fallback / "collection" / config.qdrant_collection).exists():
            config.qdrant_path = str(fallback)

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("EXPORT CHUNKS FOR GOOGLE COLAB SPARSE GENERATION")
    print("=" * 80)
    print(f"Collection:   {config.qdrant_collection}")
    print(f"Qdrant Path:  {config.qdrant_path}")
    print(f"Output File:  {out_path.resolve()}")
    print("=" * 80)

    sqlite_file = Path(config.qdrant_path) / "collection" / config.qdrant_collection / "storage.sqlite"

    # Fast and memory-efficient streaming for local embedded SQLite storage
    if sqlite_file.exists() and not config.qdrant_url:
        import pickle
        import sqlite3

        print(f"Detected local SQLite storage: {sqlite_file}")
        print("Using direct streaming cursor (prevents MemoryError on 100k+ points)...")

        conn = sqlite3.connect(str(sqlite_file))
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM points")
            total_points = cur.fetchone()[0]
            print(f"Total points in collection: {total_points:,}")

            target_total = min(total_points, limit) if limit else total_points
            query = "SELECT point FROM points"
            if limit:
                query += f" LIMIT {limit}"

            exported_count = 0
            start_time = time.perf_counter()

            with open(out_path, "w", encoding="utf-8") as f:
                cur.execute(query)
                while True:
                    rows = cur.fetchmany(batch_size)
                    if not rows:
                        break
                    for (blob,) in rows:
                        pt = pickle.loads(blob)
                        payload = pt.payload or {}
                        text = payload.get("page_content") or payload.get("text") or ""
                        record = {
                            "id": str(pt.id),
                            "text": text,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        exported_count += 1

                    elapsed = time.perf_counter() - start_time
                    rate = exported_count / elapsed if elapsed > 0 else 0.0
                    pct = (exported_count / target_total * 100) if target_total else 0.0
                    print(
                        f"Exported: {exported_count:,}/{target_total:,} ({pct:.1f}%) | "
                        f"Rate: {rate:.1f} pts/s"
                    )

            print("\n" + "-" * 80)
            print(f"Export complete: {exported_count:,} records saved to {out_path}")
            print(f"File size: {out_path.stat().st_size / (1024 * 1024):.2f} MB")
            print("=" * 80 + "\n")
            return
        finally:
            conn.close()

    # Fallback to standard client.scroll for remote Qdrant
    manager = QdrantManager(config)
    client = manager.client

    try:
        info = manager.get_collection_info()
        total_points = info.points_count or 0
        print(f"Total points in collection: {total_points:,}")

        target_total = min(total_points, limit) if limit else total_points

        exported_count = 0
        next_offset = None
        start_time = time.perf_counter()

        with open(out_path, "w", encoding="utf-8") as f:
            while exported_count < target_total:
                fetch_limit = min(batch_size, target_total - exported_count)
                points, next_offset = client.scroll(
                    collection_name=manager.collection_name,
                    limit=fetch_limit,
                    offset=next_offset,
                    with_payload=["page_content", "text"],
                    with_vectors=False,
                )

                if not points:
                    break

                for pt in points:
                    payload = pt.payload or {}
                    text = payload.get("page_content") or payload.get("text") or ""
                    record = {
                        "id": str(pt.id),
                        "text": text,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    exported_count += 1

                elapsed = time.perf_counter() - start_time
                rate = exported_count / elapsed if elapsed > 0 else 0.0
                print(
                    f"Exported: {exported_count:,}/{target_total:,} "
                    f"({exported_count / target_total * 100:.1f}%) | "
                    f"Rate: {rate:.1f} pts/s"
                )

                if next_offset is None:
                    break

        print("\n" + "-" * 80)
        print(f"Export complete: {exported_count:,} records saved to {out_path}")
        print(f"File size: {out_path.stat().st_size / (1024 * 1024):.2f} MB")
        print("=" * 80 + "\n")

    finally:
        manager.close()


def main():
    parser = argparse.ArgumentParser(
        description="Export chunk IDs and texts from Qdrant for Google Colab processing."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/chunks_for_colab.jsonl",
        help="Path to output JSONL file (default: data/chunks_for_colab.jsonl).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=None,
        help="Custom path to local embedded Qdrant storage directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Scroll batch size (default: 500).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of chunks to export.",
    )
    args = parser.parse_args()

    export_chunks(
        output_file=args.output,
        collection_name=args.collection,
        qdrant_path=args.qdrant_path,
        batch_size=args.batch_size,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
