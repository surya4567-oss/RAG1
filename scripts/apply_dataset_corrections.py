"""Apply grounded, verified data corrections to vanillarag_evaluation_dataset.json.

Preserves:
- All 200 questions and categories
- All 35 edge-case / unanswerable questions (unchanged)
- Ground-truth factual accuracy against actual source CSVs, JSONs, logs, and SRS PDF
"""

import json
import shutil
from pathlib import Path

DATASET_FILE = "vanillarag_evaluation_dataset.json"
BACKUP_FILE = "vanillarag_evaluation_dataset.json.bak"

CORRECTIONS = {
    "Q006": {
        "expected_answer": "Melissa Fletcher (Assigned To) / Juan Smith (Developer)",
        "reason": "In defects_rows.csv for BUG-000002, assigned_to is 'Melissa Fletcher' and developer is 'Juan Smith'.",
    },
    "Q009": {
        "expected_answer": "HR",
        "reason": "In defects_rows.csv for BUG-000006, the module column is 'HR' (title mentions Subscriptions).",
    },
    "Q040": {
        "expected_answer": "Defect BUG-000005 does not report an AccessError (its root cause records that the sales representative escalated the incoming shipment triggering a workflow notification).",
        "reason": "BUG-000005 description and root_cause in defects_rows.csv do not contain any AccessError.",
    },
    "Q046": {
        "expected_answer": "BUG-000766 (First defect matching module 'Purchase', severity 'Major', status 'Closed'; note: BUG-000001 has module 'Project' with title referencing Purchase)",
        "reason": "BUG-000001 has module 'Project'. In defects_rows.csv, the first record matching module 'Purchase', severity 'Major', status 'Closed' is BUG-000766.",
    },
    "Q048": {
        "expected_answer": "BUG-000072 (First defect matching module 'HR', severity 'Critical', priority 'P1'; note: BUG-000003 has module 'Website' with title referencing HR)",
        "reason": "BUG-000003 has module 'Website'. In defects_rows.csv, the first record matching module 'HR', severity 'Critical', priority 'P1' is BUG-000072.",
    },
    "Q049": {
        "expected_answer": "BUG-000004 (Defect assigned to Cheryl Bishop with status 'New' and priority 'P2'; note: record module in CSV is 'Rental' while title references HR)",
        "reason": "BUG-000004 is assigned to Cheryl Bishop with status 'New' and priority 'P2', but its CSV module is 'Rental' with title 'Issue in HR...'.",
    },
    "Q058": {
        "expected_answer": "BUG-000005 (Defect reported by Leon Sullivan carrying priority 'P0'; note: record module in CSV is 'Accounting' while title references Helpdesk)",
        "reason": "BUG-000005 was reported by Leon Sullivan with priority 'P0', but its CSV module is 'Accounting' with title 'Issue in Helpdesk...'.",
    },
    "Q062": {
        "expected_answer": "BUG-000007 (Defect with severity 'Critical' and status 'Reopened'; note: record module in CSV is 'Accounting' while title references Studio)",
        "reason": "BUG-000007 has severity 'Critical' and status 'Reopened', but its CSV module is 'Accounting' with title 'Issue in Studio...'.",
    },
    "Q063": {
        "expected_answer": "BUG-000008 (Defect with priority 'P3' and status 'Resolved'; note: record module in CSV is 'HR' while title references CRM)",
        "reason": "BUG-000008 has priority 'P3' and status 'Resolved', but its CSV module is 'HR' with title 'Issue in CRM...'.",
    },
    "Q064": {
        "expected_answer": "BUG-000009 (Defect assigned to Brandon Perez with status 'In Progress'; note: record module in CSV is 'POS' while title references Sales)",
        "reason": "BUG-000009 is assigned to Brandon Perez with status 'In Progress', but its CSV module is 'POS' with title 'Issue in Sales...'.",
    },
    "Q065": {
        "expected_answer": "BUG-000010 (Defect assigned to Joshua Phillips with severity 'Major' and priority 'P1'; note: record module in CSV is 'Website' while title references Inventory)",
        "reason": "BUG-000010 is assigned to Joshua Phillips with severity 'Major' and priority 'P1', but its CSV module is 'Website' with title 'Issue in Inventory...'.",
    },
    "Q103": {
        "expected_answer": "The warehouse operator configured the manufacturing order after verifying the related documents. The manager attempted to confirm the incoming shipment but the system raised a validation exception. The system processed the POS receipt which resulted in an AccessError. (Defect ID: BUG-000003)",
        "reason": "Exact root_cause text in defects_rows.csv for BUG-000003 (previous answer had a fabricated ending).",
    },
    "Q104": {
        "expected_answer": "UI/UX (Defect ID: BUG-000004)",
        "reason": "In defects_rows.csv, BUG-000004 has bug_type 'UI/UX', not 'Integration'.",
    },
    "Q105": {
        "expected_answer": "Kimberly Lopez (Assigned To) / James Padilla (Developer) (Defect ID: BUG-000005)",
        "reason": "In defects_rows.csv for BUG-000005, assigned_to is 'Kimberly Lopez' and developer is 'James Padilla'.",
    },
    "Q136": {
        "expected_answer": "2025-04-26",
        "reason": "In testcases.csv for TC-000001, created_date is '2025-04-26', not '2025-08-27'.",
    },
}


def apply_corrections():
    # 1. Create backup
    shutil.copyfile(DATASET_FILE, BACKUP_FILE)
    print(f"Backed up original dataset to '{BACKUP_FILE}'")

    # 2. Load dataset
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} items from '{DATASET_FILE}'")

    modified_count = 0
    for item in data:
        qid = item["id"]
        if qid in CORRECTIONS:
            old_ans = item["expected_answer"]
            new_ans = CORRECTIONS[qid]["expected_answer"]
            reason = CORRECTIONS[qid]["reason"]
            item["expected_answer"] = new_ans
            modified_count += 1
            print(f"Updated {qid}:")
            print(f"  Old: {old_ans}")
            print(f"  New: {new_ans}")
            print(f"  Reason: {reason}\n")

    # 3. Save updated dataset
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✓ Successfully updated {modified_count} expected answers in '{DATASET_FILE}'")


if __name__ == "__main__":
    apply_corrections()

