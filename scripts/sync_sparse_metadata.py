"""
Synchronize metadata from qa_knowledge to qa_knowledge_sparse.
Ensures sparse collection has all metadata fields (chunk_id, document_type, defect_id, etc.)
without modifying the existing sparse vectors.
"""

import os
import sqlite3
import pickle
import time
from pathlib import Path


def sync_metadata(
    dense_db_path: str = "data/data1/qdrant_db/collection/qa_knowledge/storage.sqlite",
    sparse_db_path: str = "data/data1/qdrant_db/collection/qa_knowledge_sparse/storage.sqlite",
    batch_size: int = 5000,
):
    dense_path = Path(dense_db_path)
    sparse_path = Path(sparse_db_path)

    if not dense_path.exists():
        raise FileNotFoundError(f"Dense collection sqlite not found: {dense_path}")
    if not sparse_path.exists():
        raise FileNotFoundError(f"Sparse collection sqlite not found: {sparse_path}")

    print(f"Connecting to:\n Dense: {dense_path}\n Sparse: {sparse_path}")
    start_time = time.time()

    conn_dense = sqlite3.connect(str(dense_path))
    conn_sparse = sqlite3.connect(str(sparse_path))

    cur_dense = conn_dense.cursor()
    cur_sparse = conn_sparse.cursor()

    # Get total count
    cur_dense.execute("SELECT COUNT(*) FROM points")
    total_dense = cur_dense.fetchone()[0]
    print(f"Total points in dense collection: {total_dense}")

    cur_sparse.execute("SELECT COUNT(*) FROM points")
    total_sparse = cur_sparse.fetchone()[0]
    print(f"Total points in sparse collection: {total_sparse}")

    # Fetch dense points in batches
    cur_dense.execute("SELECT id, point FROM points")

    updated = 0
    missing_in_sparse = 0
    batch_updates = []

    while True:
        rows = cur_dense.fetchmany(batch_size)
        if not rows:
            break

        # For these IDs, fetch existing sparse points
        id_list = [r[0] for r in rows]
        placeholders = ",".join("?" for _ in id_list)
        cur_sparse.execute(f"SELECT id, point FROM points WHERE id IN ({placeholders})", id_list)
        sparse_map = {r[0]: r[1] for r in cur_sparse.fetchall()}

        for pid, dense_blob in rows:
            if pid not in sparse_map:
                missing_in_sparse += 1
                continue

            dense_pt = pickle.loads(dense_blob)
            sparse_pt = pickle.loads(sparse_map[pid])

            # Merge dense payload into sparse point, keeping sparse vector intact
            merged_payload = dict(dense_pt.payload or {})
            if "text" in sparse_pt.payload and "text" not in merged_payload:
                merged_payload["text"] = sparse_pt.payload["text"]

            sparse_pt.payload = merged_payload
            new_blob = pickle.dumps(sparse_pt)
            batch_updates.append((new_blob, pid))

        if batch_updates:
            cur_sparse.executemany("UPDATE points SET point = ? WHERE id = ?", batch_updates)
            conn_sparse.commit()
            updated += len(batch_updates)
            batch_updates.clear()
            elapsed = time.time() - start_time
            print(f"Progress: {updated}/{total_dense} points synced ({updated/total_dense*100:.1f}%) in {elapsed:.1f}s")

    conn_dense.close()
    conn_sparse.close()

    total_time = time.time() - start_time
    print(f"\nDone! Synced {updated} points in {total_time:.2f}s. Missing in sparse: {missing_in_sparse}")


if __name__ == "__main__":
    sync_metadata()

