"""
Enrich Qdrant collections (qa_knowledge and qa_knowledge_sparse) with missing source fields:
- defects_rows.csv: assigned_to, developer, reported_by
- testcases.csv: automation_framework, component, risk

Preserves all existing vectors, IDs, and schemas.
"""

import os
import pickle
import sqlite3
import time
from pathlib import Path
import pandas as pd


def enrich_collections(
    dense_db_path: str = "data/data1/qdrant_db/collection/qa_knowledge/storage.sqlite",
    sparse_db_path: str = "data/data1/qdrant_db/collection/qa_knowledge_sparse/storage.sqlite",
    defects_csv: str = "data/defects_rows.csv",
    testcases_csv: str = "data/testcases.csv",
):
    print("=" * 80)
    print("ENRICHING METADATA FROM SOURCE CSVs INTO QDRANT SQLITE")
    print("=" * 80)

    # 1. Load CSV mappings
    print("Loading source CSVs...")
    df_defects = pd.read_csv(defects_csv)
    defect_map = {}
    for _, row in df_defects.iterrows():
        did = str(row.get("defect_id", "")).strip()
        if did:
            defect_map[did] = {
                "assigned_to": str(row.get("assigned_to", "")).strip() if pd.notna(row.get("assigned_to")) else None,
                "developer": str(row.get("developer", "")).strip() if pd.notna(row.get("developer")) else None,
                "reported_by": str(row.get("reported_by", "")).strip() if pd.notna(row.get("reported_by")) else None,
            }

    print(f"Loaded {len(defect_map)} defect records from {defects_csv}")

    df_tc = pd.read_csv(testcases_csv)
    tc_map = {}
    for _, row in df_tc.iterrows():
        tcid = str(row.get("test_case_id", "")).strip()
        if tcid:
            tc_map[tcid] = {
                "automation_framework": str(row.get("automation_framework", "")).strip() if pd.notna(row.get("automation_framework")) else None,
                "component": str(row.get("component", "")).strip() if pd.notna(row.get("component")) else None,
                "risk": str(row.get("risk", "")).strip() if pd.notna(row.get("risk")) else None,
            }

    print(f"Loaded {len(tc_map)} test case records from {testcases_csv}")

    # 2. Enrich collections
    for db_path_str in [dense_db_path, sparse_db_path]:
        db_path = Path(db_path_str)
        if not db_path.exists():
            print(f"⚠️ SQLite path not found: {db_path}")
            continue

        print(f"\nProcessing: {db_path}...")
        start_t = time.time()
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        cur.execute("SELECT id FROM points")
        all_ids = [r[0] for r in cur.fetchall()]
        print(f"Total points to scan: {len(all_ids)}")

        batch_updates = []
        updated_count = 0
        batch_size = 5000

        for i in range(0, len(all_ids), batch_size):
            chunk_ids = all_ids[i : i + batch_size]
            placeholders = ",".join("?" for _ in chunk_ids)
            cur.execute(f"SELECT id, point FROM points WHERE id IN ({placeholders})", chunk_ids)
            rows = cur.fetchall()

            for pid, blob in rows:
                pt = pickle.loads(blob)
                payload = pt.payload or {}
                doc_type = payload.get("document_type")
                modified = False

                if doc_type == "defect":
                    did = payload.get("defect_id") or payload.get("record_id")
                    if did and did in defect_map:
                        d_info = defect_map[did]
                        for k, v in d_info.items():
                            if v and payload.get(k) != v:
                                payload[k] = v
                                modified = True

                        chunk_id = str(payload.get("chunk_id", ""))
                        if chunk_id.endswith("_c0"):
                            text = payload.get("page_content") or payload.get("text") or ""
                            if "Assigned To" not in text and d_info.get("assigned_to"):
                                text += f"\n\n**Assigned To / Developer**: {d_info.get('assigned_to')}"
                                if d_info.get("developer"):
                                    text += f" / {d_info.get('developer')}"
                                if d_info.get("reported_by"):
                                    text += f"\n**Reported By**: {d_info.get('reported_by')}"
                                payload["page_content"] = text
                                payload["text"] = text
                                modified = True

                elif doc_type == "test_case":
                    tcid = payload.get("test_case_id") or payload.get("record_id")
                    if tcid and tcid in tc_map:
                        t_info = tc_map[tcid]
                        for k, v in t_info.items():
                            if v and payload.get(k) != v:
                                payload[k] = v
                                modified = True

                        chunk_id = str(payload.get("chunk_id", ""))
                        if chunk_id.endswith("_c0"):
                            text = payload.get("page_content") or payload.get("text") or ""
                            if "Automation Framework" not in text and t_info.get("automation_framework"):
                                text += f"\n\n**Automation Framework**: {t_info.get('automation_framework')}"
                                payload["page_content"] = text
                                payload["text"] = text
                                modified = True

                if modified:
                    pt.payload = payload
                    new_blob = pickle.dumps(pt)
                    batch_updates.append((new_blob, pid))

            if batch_updates:
                cur.executemany("UPDATE points SET point = ? WHERE id = ?", batch_updates)
                conn.commit()
                updated_count += len(batch_updates)
                batch_updates.clear()

        conn.close()
        print(f"✓ Updated {updated_count} points in {db_path.name} ({time.time() - start_t:.2f}s)")

    print("\n" + "=" * 80)
    print("METADATA ENRICHMENT COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    enrich_collections()

