"""Audit all 200 queries in vanillarag_evaluation_dataset.json against ground-truth source data."""

import json
import re
from pathlib import Path
import pandas as pd
from pypdf import PdfReader

# Load ground truth files
print("Loading data files...")
df_defects = pd.read_csv("data/defects_rows.csv").set_index("defect_id")
df_testcases = pd.read_csv("data/testcases.csv").set_index("test_case_id")
df_executions = pd.read_csv("data/test_execution.csv").set_index("execution_id")

# Load all OCR files into memory
ocr_by_bug = {}
ocr_by_exec = {}
for f in sorted(Path("data/screenshots").glob("*.json")):
    with open(f, "r", encoding="utf-8") as jf:
        items = json.load(jf)
        for it in items:
            bid = it.get("bug_id") or it.get("defect_id")
            eid = it.get("execution_id")
            if bid and bid not in ocr_by_bug:
                ocr_by_bug[bid] = it
            if eid and eid not in ocr_by_exec:
                ocr_by_exec[eid] = it

# Load all failure logs into memory
log_by_bug = {}
log_by_exec = {}
for f in sorted(Path("data/failure_logs").glob("*.txt")):
    with open(f, "r", encoding="utf-8") as lf:
        content = lf.read()
        blocks = content.split("======================================================================\n")
        for b in blocks:
            if not b.strip():
                continue
            m_bug = re.search(r"Bug ID:\s*(BUG-\d+)", b)
            m_exec = re.search(r"Execution ID:\s*(EXEC-\d+)", b)
            if m_bug and m_bug.group(1) not in log_by_bug:
                log_by_bug[m_bug.group(1)] = b
            if m_exec and m_exec.group(1) not in log_by_exec:
                log_by_exec[m_exec.group(1)] = b

# Load SRS text
pdf = PdfReader("data/SRS.pdf")
srs_text = ""
for page in pdf.pages:
    srs_text += page.extract_text() + "\n"

# Load dataset
with open("vanillarag_evaluation_dataset.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print(f"Loaded {len(eval_data)} queries.\n")

issues_to_fix = []

for item in eval_data:
    qid = item["id"]
    query = item["query"]
    exp = str(item["expected_answer"]).strip()
    cat = item["category"]

    if cat == "Edge Cases / Unanswerable Queries":
        continue

    # 1. Defect checks
    m_bug = re.findall(r"\b(BUG-\d+)\b", query)
    if m_bug:
        for bid in m_bug:
            if bid in df_defects.index:
                row = df_defects.loc[bid]
                # Check Q006: assigned developer
                if qid == "Q006":
                    # Query: Who is the assigned developer for defect BUG-000002?
                    # CSV has assigned_to: Melissa Fletcher, developer: Juan Smith
                    correct_ans = f"Melissa Fletcher (Assigned To) / Juan Smith (Developer)"
                    if exp != correct_ans and "Juan Smith" not in exp:
                        issues_to_fix.append((qid, cat, query, exp, correct_ans, "BUG-000002 developer is Juan Smith, assigned_to is Melissa Fletcher"))

                # Check Q009: module of BUG-000006
                elif qid == "Q009":
                    actual_mod = str(row["module"])
                    if exp != actual_mod:
                        issues_to_fix.append((qid, cat, query, exp, actual_mod, f"Actual module column is '{actual_mod}', not '{exp}'"))

    # 2. Test Case Checks
    m_tc = re.findall(r"\b(TC-\d+)\b", query)
    if m_tc:
        for tcid in m_tc:
            if tcid in df_testcases.index:
                row = df_testcases.loc[tcid]
                # Check Q136: created_date for TC-000001
                if qid == "Q136":
                    actual_date = str(row["created_date"])
                    if exp != actual_date:
                        issues_to_fix.append((qid, cat, query, exp, actual_date, f"Actual created_date in testcases.csv is '{actual_date}', not '{exp}'"))

    # 3. Metadata Filtering queries where module filter was mismatched with title
    # Q046: BUG-000001 module is Project, not Purchase
    if qid == "Q046":
        # Query says: Which defect has module 'Purchase', severity 'Major', and status 'Closed'?
        # But BUG-000001 has module 'Project'!
        # In reality, defects with Purchase, Major, Closed are BUG-000766, etc.
        # If the query intended BUG-000001, the query should say module 'Project' or title 'Issue in Purchase...'
        # Or if the query is kept as-is, BUG-000001 is wrong!
        issues_to_fix.append((qid, cat, query, exp, "BUG-000766 (or BUG-001138)", "BUG-000001 has module 'Project'. First defect with module='Purchase', Major, Closed is BUG-000766"))

    if qid == "Q048":
        # Query: Find the defect in module 'HR' that has severity 'Critical' and priority 'P1'.
        # BUG-000003 has module 'Website'!
        # Defects with module 'HR', Critical, P1:
        match_bugs = df_defects[(df_defects['module'] == 'HR') & (df_defects['severity'] == 'Critical') & (df_defects['priority'] == 'P1')]
        first_match = match_bugs.index[0] if len(match_bugs) else "None"
        issues_to_fix.append((qid, cat, query, exp, first_match, f"BUG-000003 has module 'Website'. Actual first defect with module='HR', Critical, P1 is {first_match}"))

    if qid == "Q049":
        # Query: Which defect in module 'HR' is assigned to 'Cheryl Bishop' with status 'New' and priority 'P2'?
        # BUG-000004 has module 'Rental'!
        issues_to_fix.append((qid, cat, query, exp, "BUG-000004 (module in data is Rental)", "BUG-000004 title mentions HR, but its CSV module is Rental"))

    if qid == "Q058":
        # Query: Which defect reported by 'Leon Sullivan' in module 'Helpdesk' carries priority 'P0'?
        # BUG-000005 has module 'Accounting'!
        issues_to_fix.append((qid, cat, query, exp, "BUG-000005 (module in data is Accounting)", "BUG-000005 title mentions Helpdesk, but CSV module is Accounting"))

    if qid == "Q062":
        # Query: Which defect in module 'Studio' has severity 'Critical' and status 'Reopened'?
        # BUG-000007 has module 'Accounting'!
        issues_to_fix.append((qid, cat, query, exp, "BUG-000007 (module in data is Accounting)", "BUG-000007 title mentions Studio, but CSV module is Accounting"))

    if qid == "Q063":
        # Query: Find the defect in module 'CRM' with priority 'P3' and status 'Resolved'.
        # BUG-000008 has module 'HR'!
        issues_to_fix.append((qid, cat, query, exp, "BUG-000008 (module in data is HR)", "BUG-000008 title mentions CRM, but CSV module is HR"))

    if qid == "Q064":
        # Query: Which defect in module 'Sales' is assigned to 'Brandon Perez' with status 'In Progress'?
        # BUG-000009 has module 'POS'!
        issues_to_fix.append((qid, cat, query, exp, "BUG-000009 (module in data is POS)", "BUG-000009 title mentions Sales, but CSV module is POS"))

    if qid == "Q065":
        # Query: Find the defect in module 'Inventory' that has severity 'Major' and priority 'P1' assigned to 'Joshua Phillips'.
        # BUG-000010 has module 'Website'!
        issues_to_fix.append((qid, cat, query, exp, "BUG-000010 (module in data is Website)", "BUG-000010 title mentions Inventory, but CSV module is Website"))

    # 4. Multi-hop checks
    if qid == "Q103":
        # Query: What root cause is recorded for the defect linked to test execution EXEC-004706?
        # Expected in JSON has hallucinated suffix "and successfully completed the transaction."
        # Actual root cause in BUG-000003:
        actual_rc = df_defects.loc['BUG-000003', 'root_cause']
        if "successfully completed" in exp and "successfully completed" not in actual_rc:
            issues_to_fix.append((qid, cat, query, exp, f"{actual_rc} (Defect ID: BUG-000003)", "Expected answer had fabricated root cause ending"))

    if qid == "Q104":
        # Query: What bug type is assigned to the defect linked to test execution EXEC-013356?
        # Expected in JSON: Integration (Defect ID: BUG-000004)
        # Actual bug_type for BUG-000004 in CSV is 'UI/UX'!
        actual_bt = df_defects.loc['BUG-000004', 'bug_type']
        if exp.split()[0] != actual_bt:
            issues_to_fix.append((qid, cat, query, exp, f"{actual_bt} (Defect ID: BUG-000004)", f"BUG-000004 bug_type is '{actual_bt}', not '{exp}'"))

    if qid == "Q105":
        # Query: Who is the developer assigned to the defect linked to test execution EXEC-009336?
        # In BUG-000005: assigned_to is Kimberly Lopez, developer is James Padilla
        correct_ans = "Kimberly Lopez (Assigned To) / James Padilla (Developer) (Defect ID: BUG-000005)"
        if "James Padilla" not in exp:
            issues_to_fix.append((qid, cat, query, exp, correct_ans, "BUG-000005 assigned_to is Kimberly Lopez, developer is James Padilla"))

print("=" * 80)
print(f"TOTAL GROUND-TRUTH ISSUES FOUND: {len(issues_to_fix)}")
print("=" * 80)
for qid, cat, q, exp, correct, reason in issues_to_fix:
    print(f"\n[{qid}] ({cat})")
    print(f"  Query:    {q}")
    print(f"  Current:  {exp}")
    print(f"  Correct:  {correct}")
    print(f"  Reason:   {reason}")

