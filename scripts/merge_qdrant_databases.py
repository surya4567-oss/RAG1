"""Utility to merge two separate Qdrant local databases into a single database.

Merges 'qa_knowledge' (dense) and 'qa_knowledge_sparse' (sparse) so both collections
reside in the exact same Qdrant storage path and can be accessed together.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def merge_qdrant_databases(
    dense_db_path: str = "data/data1/qdrant_db",
    sparse_db_path: str = "data/qdrant_sparse_db",
    target_db_path: str = "data/data1/qdrant_db",
):
    dense_path = Path(dense_db_path)
    sparse_path = Path(sparse_db_path)
    target_path = Path(target_db_path)

    print("\n" + "=" * 80)
    print("MERGE DENSE AND SPARSE COLLECTIONS INTO ONE QDRANT DATABASE")
    print("=" * 80)
    print(f"Source Dense DB:   {dense_path.resolve()}")
    print(f"Source Sparse DB:  {sparse_path.resolve()}")
    print(f"Target Unified DB: {target_path.resolve()}")
    print("=" * 80)

    # 1. Validation
    if not (dense_path / "meta.json").exists():
        print(f"❌ Error: Dense meta.json not found in {dense_path}")
        return False
    if not (sparse_path / "meta.json").exists():
        print(f"❌ Error: Sparse meta.json not found in {sparse_path}")
        return False

    with open(dense_path / "meta.json", "r", encoding="utf-8") as f:
        dense_meta = json.load(f)
    with open(sparse_path / "meta.json", "r", encoding="utf-8") as f:
        sparse_meta = json.load(f)

    print(f"\nCollections in Dense DB:  {list(dense_meta.get('collections', {}).keys())}")
    print(f"Collections in Sparse DB: {list(sparse_meta.get('collections', {}).keys())}")

    target_path.mkdir(parents=True, exist_ok=True)
    target_collections_dir = target_path / "collection"
    target_collections_dir.mkdir(parents=True, exist_ok=True)

    # 2. Copy/Link sparse collection into target if not already there
    sparse_src_col = sparse_path / "collection" / "qa_knowledge_sparse"
    sparse_dst_col = target_collections_dir / "qa_knowledge_sparse"

    if sparse_src_col.exists():
        if sparse_dst_col.exists() and sparse_dst_col.resolve() != sparse_src_col.resolve():
            print(f"Notice: Overwriting existing '{sparse_dst_col.name}' in target...")
            shutil.rmtree(sparse_dst_col)

        if sparse_dst_col.resolve() != sparse_src_col.resolve():
            print(f"Copying sparse collection files to {sparse_dst_col} ...")
            shutil.copytree(sparse_src_col, sparse_dst_col)
            print("Sparse collection files copied successfully.")
        else:
            print("Sparse collection is already in target directory.")
    else:
        print(f"❌ Error: Sparse collection directory not found at {sparse_src_col}")
        return False

    # 3. If target is different from dense DB, copy dense collection too
    dense_src_col = dense_path / "collection" / "qa_knowledge"
    dense_dst_col = target_collections_dir / "qa_knowledge"

    if dense_dst_col.resolve() != dense_src_col.resolve():
        if not dense_dst_col.exists():
            print(f"Copying dense collection files to {dense_dst_col} ...")
            shutil.copytree(dense_src_col, dense_dst_col)
            print("Dense collection files copied successfully.")

    # 4. Merge meta.json
    merged_collections = {}
    merged_collections.update(dense_meta.get("collections", {}))
    merged_collections.update(sparse_meta.get("collections", {}))

    merged_meta = {
        "collections": merged_collections,
        "aliases": {
            **dense_meta.get("aliases", {}),
            **sparse_meta.get("aliases", {}),
        },
    }

    target_meta_file = target_path / "meta.json"
    temp_meta = target_path / "meta.json.tmp"
    with open(temp_meta, "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, indent=2)
    temp_meta.replace(target_meta_file)

    print("\n" + "-" * 80)
    print("SUCCESS: Unified Qdrant database created!")
    print(f"Database Location: {target_path.resolve()}")
    print("Collections available in this database:")
    for col_name in merged_collections:
        print(f"  • {col_name}")
    print("=" * 80 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Merge dense and sparse Qdrant local databases into a single database."
    )
    parser.add_argument(
        "--dense-db",
        type=str,
        default="data/data1/qdrant_db",
        help="Path to directory containing dense collection (default: data/data1/qdrant_db).",
    )
    parser.add_argument(
        "--sparse-db",
        type=str,
        default="data/qdrant_sparse_db",
        help="Path to directory containing sparse collection (default: data/qdrant_sparse_db).",
    )
    parser.add_argument(
        "--target-db",
        type=str,
        default="data/data1/qdrant_db",
        help="Target unified database directory (default: data/data1/qdrant_db).",
    )

    args = parser.parse_args()
    merge_qdrant_databases(
        dense_db_path=args.dense_db,
        sparse_db_path=args.sparse_db,
        target_db_path=args.target_db,
    )


if __name__ == "__main__":
    main()
