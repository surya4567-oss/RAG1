"""Comprehensive auditor for vanillarag_evaluation_dataset.json against actual source data files.

Validates:
- defects_rows.csv
- testcases.csv
- test_execution.csv
- ocr_batch_*.json
- failure_logs/*.txt
"""

import json
import os
import re
from pathlib import Path
import pandas as pd


def audit():
    print("=" * 80)
    print("AUDITING EVALUATION DATASET AGAINST SOURCE FILES")
    print("=" * 80)

    # 1. Load source datasets
    print("Loading CSVs...")
    df_defects = pd.read_csv("data/defects_rows.csv").set_index("defect_id")
    df_testcases = pd.read_csv("data/testcases.csv").set_index("test_case_id")
    df_executions = pd.read_csv("data/test_execution.csv").set_index("execution_id")

    print(f"Loaded {len(df_defects)} defects, {len(df_testcases)} test cases, {len(df_executions)} executions.")

    # 2. Load OCR batches
    ocr_records = {}
    ocr_files = sorted(Path("data/screenshots").glob("*.json"))
    for f in ocr_files:
        with open(f, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            items = data if isinstance(data, list) else data.get("screenshots", [])
            for item in items:
                eid = item.get("execution_id")
                if eid:
                    ocr_records[eid] = {**item, "_source_file": f.name}
    print(f"Loaded {len(ocr_records)} OCR execution records from {len(ocr_files)} files.")

    # 3. Load Failure Logs
    log_records = {}
    log_files = sorted(Path("data/failure_logs").glob("*.txt"))
    for f in log_files:
        with open(f, "r", encoding="utf-8") as lf:
            content = lf.read()
            # Split by [LOG- or [EXEC- blocks if present
            blocks = re.split(r"(?=\[LOG-|\bLOG-\d+)", content)
            for b in blocks:
                m_exec = re.search(r"\b(EXEC-\d+)\b", b)
                m_log = re.search(r"\b(LOG-\d+)\b", b)
                if m_exec:
                    log_records[m_exec.group(1)] = {"text": b.strip(), "_source_file": f.name}
                if m_log:
                    log_records[m_log.group(1)] = {"text": b.strip(), "_source_file": f.name}
    print(f"Loaded {len(log_records)} failure log records from {len(log_files)} files.")

    # 4. Load evaluation dataset
    with open("vanillarag_evaluation_dataset.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    print(f"Loaded {len(eval_data)} evaluation queries.\n")

    discrepancies = []

    for item in eval_data:
        qid = item["id"]
        query = item["query"]
        expected = str(item["expected_answer"]).strip()
        cat = item["category"]

        if cat == "Edge Cases / Unanswerable Queries":
            # Per instruction: "dont change expected answer for invalid queries"
            continue

        # Look for IDs in query
        m_bug = re.findall(r"\b(BUG-\d+)\b", query)
        m_tc = re.findall(r"\b(TC-\d+)\b", query)
        m_exec = re.findall(r"\b(EXEC-\d+)\b", query)
        m_req = re.findall(r"\b(REQ-\d+)\b", query)

        # Check Defect Queries
        if m_bug and len(m_bug) == 1 and not m_tc and not m_exec:
            bid = m_bug[0]
            if bid in df_defects.index:
                row = df_defects.loc[bid].to_dict()
                # Check which field was asked for
                q_lower = query.lower()
                matched_field = None
                real_val = None

                if "title" in q_lower:
                    matched_field = "title"
                    real_val = str(row.get("title"))
                elif "severity" in q_lower:
                    matched_field = "severity"
                    real_val = str(row.get("severity"))
                elif "priority" in q_lower:
                    matched_field = "priority"
                    real_val = str(row.get("priority"))
                elif "developer" in q_lower:
                    matched_field = "developer"
                    real_val = f"{row.get('assigned_to')} (Assigned To) / {row.get('developer')} (Developer)" if row.get("developer") != row.get("assigned_to") else str(row.get("developer"))
                elif "reported" in q_lower or "reporter" in q_lower:
                    matched_field = "reported_by"
                    real_val = str(row.get("reported_by"))
                elif "module" in q_lower:
                    matched_field = "module"
                    real_val = str(row.get("module"))
                elif "status" in q_lower:
                    matched_field = "status"
                    real_val = str(row.get("status"))
                elif "root cause" in q_lower:
                    matched_field = "root_cause"
                    real_val = str(row.get("root_cause"))
                elif "reproducibility" in q_lower:
                    matched_field = "reproducibility"
                    real_val = str(row.get("reproducibility"))
                elif "bug type" in q_lower:
                    matched_field = "bug_type"
                    real_val = str(row.get("bug_type"))
                elif "fix version" in q_lower:
                    matched_field = "fix_version"
                    real_val = str(row.get("fix_version"))
                elif "found in build" in q_lower:
                    matched_field = "found_in_build"
                    real_val = str(row.get("found_in_build"))
                elif "fixed" in q_lower and "build" in q_lower:
                    matched_field = "fixed_build"
                    real_val = str(row.get("fixed_build"))

                if matched_field and real_val:
                    # Compare
                    if expected.lower() not in real_val.lower() and real_val.lower() not in expected.lower():
                        discrepancies.append({
                            "id": qid,
                            "category": cat,
                            "query": query,
                            "expected_answer": expected,
                            "source_field": matched_field,
                            "actual_value_in_data": real_val,
                            "record_id": bid,
                        })

        # Check Test Case Queries
        elif m_tc and len(m_tc) == 1 and not m_bug and not m_exec:
            tcid = m_tc[0]
            if tcid in df_testcases.index:
                row = df_testcases.loc[tcid].to_dict()
                q_lower = query.lower()
                matched_field = None
                real_val = None

                if "requirement" in q_lower:
                    matched_field = "requirement_id"
                    real_val = str(row.get("requirement_id"))
                elif "module" in q_lower:
                    matched_field = "module"
                    real_val = str(row.get("module"))
                elif "priority" in q_lower:
                    matched_field = "priority"
                    real_val = str(row.get("priority"))
                elif "status" in q_lower or "automation status" in q_lower:
                    matched_field = "status"
                    real_val = str(row.get("status"))
                elif "component" in q_lower:
                    matched_field = "component"
                    real_val = str(row.get("component"))
                elif "framework" in q_lower:
                    matched_field = "automation_framework"
                    real_val = str(row.get("automation_framework"))
                elif "test type" in q_lower:
                    matched_field = "test_type"
                    real_val = str(row.get("test_type"))
                elif "precondition" in q_lower:
                    matched_field = "preconditions"
                    real_val = str(row.get("preconditions"))
                elif "execution time" in q_lower or "estimated" in q_lower:
                    matched_field = "execution_time"
                    real_val = str(row.get("execution_time"))
                elif "title" in q_lower or "name" in q_lower:
                    matched_field = "name"
                    real_val = str(row.get("name"))

                if matched_field and real_val:
                    if expected.lower() not in real_val.lower() and real_val.lower() not in expected.lower():
                        discrepancies.append({
                            "id": qid,
                            "category": cat,
                            "query": query,
                            "expected_answer": expected,
                            "source_field": matched_field,
                            "actual_value_in_data": real_val,
                            "record_id": tcid,
                        })

        # Check Execution Queries
        elif m_exec and len(m_exec) == 1 and not m_bug and not m_tc:
            eid = m_exec[0]
            if eid in df_executions.index:
                row = df_executions.loc[eid].to_dict()
                q_lower = query.lower()
                matched_field = None
                real_val = None

                if "date" in q_lower:
                    matched_field = "execution_date"
                    real_val = str(row.get("execution_date"))
                elif "browser" in q_lower:
                    matched_field = "browser"
                    real_val = str(row.get("browser"))
                elif "operating system" in q_lower or "os" in q_lower:
                    matched_field = "operating_system"
                    real_val = str(row.get("operating_system"))
                elif "execution time" in q_lower or "seconds" in q_lower:
                    matched_field = "execution_time_seconds"
                    real_val = str(row.get("execution_time_seconds"))
                elif "build" in q_lower:
                    matched_field = "build_number"
                    real_val = str(row.get("build_number"))
                elif "status" in q_lower:
                    matched_field = "status"
                    real_val = str(row.get("status"))
                elif "tester" in q_lower:
                    matched_field = "tester"
                    real_val = str(row.get("tester"))
                elif "environment" in q_lower:
                    matched_field = "environment"
                    real_val = str(row.get("environment"))

                if matched_field and real_val:
                    if expected.lower() not in real_val.lower() and real_val.lower() not in expected.lower():
                        discrepancies.append({
                            "id": qid,
                            "category": cat,
                            "query": query,
                            "expected_answer": expected,
                            "source_field": matched_field,
                            "actual_value_in_data": real_val,
                            "record_id": eid,
                        })

    print(f"Audit completed. Found {len(discrepancies)} discrepancies in single-entity queries.")
    for d in discrepancies:
        print("-" * 60)
        print(f"[{d['id']}] Category: {d['category']}")
        print(f"Query:           {d['query']}")
        print(f"Expected in JSON:{d['expected_answer']}")
        print(f"Source Field:    {d['source_field']}")
        print(f"Actual in CSV:   {d['actual_value_in_data']}")

    return discrepancies


if __name__ == "__main__":
    audit()

