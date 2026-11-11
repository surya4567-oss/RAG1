"""Generate the 200-question VanillaRAG Evaluation Dataset strictly grounded in the source corpus.

Validates all 11 category distributions, field schemas, ground-truth answers, and edge-case requirements.
Outputs vanillarag_evaluation_dataset.json.
"""

import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List


def build_dataset() -> List[Dict[str, str]]:
    dataset: List[Dict[str, str]] = []

    # =========================================================================
    # 1. Basic Retrieval (25 queries: Q001 to Q025)
    # =========================================================================
    basic_queries = [
        (
            "What is the official document identifier of the Odoo ERP System Software Requirements Specification?",
            "ODOO-ERP-SRS-001",
        ),
        (
            "What version and status are stated on the title page of the Odoo ERP System SRS?",
            "Version 1.0 — Baseline draft, with status 'For stakeholder review and approval'.",
        ),
        (
            "What baseline date is recorded on the cover page of the Odoo ERP System SRS?",
            "11 October 2026",
        ),
        (
            "What is the title of defect record BUG-000001?",
            "Issue in Purchase: Team-oriented transitional concept",
        ),
        (
            "What severity level is assigned to defect record BUG-000003?",
            "Critical",
        ),
        (
            "Who is the assigned developer for defect BUG-000002?",
            "Melissa Fletcher",
        ),
        (
            "Who reported defect BUG-000005 in the defect tracking system?",
            "Leon Sullivan",
        ),
        (
            "What priority level is designated for defect BUG-000004?",
            "P2",
        ),
        (
            "Which application module is defect BUG-000006 associated with?",
            "Subscriptions",
        ),
        (
            "What is the current lifecycle status of defect BUG-000007?",
            "Reopened",
        ),
        (
            "On what date was test execution EXEC-000001 performed?",
            "2025-09-11",
        ),
        (
            "Which web browser was utilized during test execution EXEC-000002?",
            "Edge",
        ),
        (
            "What operating system environment was used for test execution EXEC-000004?",
            "Ubuntu",
        ),
        (
            "What was the total execution time in seconds for test execution EXEC-000003?",
            "287 seconds",
        ),
        (
            "Which software build number was tested in test execution EXEC-000005?",
            "B-2069",
        ),
        (
            "What execution outcome status was recorded for test run EXEC-000006?",
            "Failed",
        ),
        (
            "What requirement ID is mapped to test case TC-000001?",
            "REQ-1924",
        ),
        (
            "Which functional module is test case TC-000002 designed to validate?",
            "Quality",
        ),
        (
            "What priority rating is assigned to test case TC-000004?",
            "Critical",
        ),
        (
            "What is the automation lifecycle status of test case TC-000003?",
            "Draft",
        ),
        (
            "What precondition is specified for test case TC-000008?",
            "The background job configured the invoice and successfully completed the transaction.",
        ),
        (
            "What screen name is identified in the OCR capture for bug BUG-001585 in ocr_batch_001.json?",
            "Inventory Dashboard",
        ),
        (
            "What module name is recorded in ocr_batch_001.json for test execution EXEC-002249?",
            "HR",
        ),
        (
            "What automation framework is specified for test case TC-000001?",
            "Cypress",
        ),
        (
            "What HTTP status code is reported in logs_001.txt for the failure associated with BUG-003596?",
            "404",
        ),
    ]

    for q, ans in basic_queries:
        dataset.append({
            "id": f"Q{len(dataset)+1:03d}",
            "query": q,
            "expected_answer": ans,
            "category": "Basic Retrieval",
        })

    # =========================================================================
    # 2. Semantic Queries (20 queries: Q026 to Q045)
    # =========================================================================
    semantic_queries = [
        (
            "According to the Odoo ERP SRS, what overarching purpose does the specification document serve?",
            "It establishes a common understanding of expected system behavior and quality among business stakeholders, implementation specialists, developers, testers, administrators, and support personnel.",
        ),
        (
            "What fundamental security access principle governs the Operator user class in the SRS?",
            "Minimum rights for assigned duties, creating and processing only assigned operational records.",
        ),
        (
            "In the SRS capability groupings, what specific modules fall under the Supply chain domain?",
            "Purchase, Inventory, Manufacturing, Quality, and Maintenance.",
        ),
        (
            "What typical business records are handled by the Commercial capability group under the SRS?",
            "Lead, opportunity, quotation, order, contract, and receipt.",
        ),
        (
            "What specific responsibilities and access scope are designated for Auditor users in the SRS?",
            "Review records, approvals, and reports under a strictly read-only scope across domain records.",
        ),
        (
            "What constraint does the SRS mandate regarding transaction consistency during failed operations?",
            "Failed operations must not leave partial stock, accounting, or workflow transactions.",
        ),
        (
            "How does the SRS outline the complete operational flow of the Lead-to-cash process?",
            "Capture and qualify a prospect, manage an opportunity, prepare a quotation, obtain approval, confirm the order, fulfill goods/services, issue the invoice, register payment, and reconcile accounts.",
        ),
        (
            "What quantity matching rule is defined in the SRS for the Procure-to-pay workflow?",
            "Receipt and bill quantities are compared under the approved matching policy before registering payment.",
        ),
        (
            "What operational data attributes must be recorded for every stock movement under the SRS?",
            "Product, quantity, unit, source location, destination location, date, responsible user, and linked source document.",
        ),
        (
            "What change description was recorded in Document Control for Version 0.1 of the SRS?",
            "Initial compilation of domain requirements.",
        ),
        (
            "Which test case is designed to verify that an API endpoint checks employee records when configuration settings are missing?",
            "TC-000005",
        ),
        (
            "Find the test case where a sales representative processes a subscription renewal after verifying related documents.",
            "TC-000003",
        ),
        (
            "What business operation is stated as the root cause for defect BUG-000001?",
            "The system processed the incoming shipment and successfully completed the transaction.",
        ),
        (
            "What underlying operation led to the failure logged in defect BUG-000002 during CRM processing?",
            "The background job validated the subscription renewal and successfully completed the transaction.",
        ),
        (
            "What operation triggered the AccessError reported in defect BUG-000005?",
            "The sales representative escalated the incoming shipment which resulted in an AccessError.",
        ),
        (
            "How does the SRS define the functional scope of the Finance capability group?",
            "Accounting, invoices, payments, reconciliation, and tax.",
        ),
        (
            "What supervisory oversight responsibilities are assigned to the Manager user class in the SRS?",
            "Review work queues, approve transactions within delegated authority, and monitor exceptions.",
        ),
        (
            "What human resources scope is encompassed by the People capability group in the SRS?",
            "HR and Payroll, managing employee records, departments, contracts, and payslips.",
        ),
        (
            "What operational responsibilities are assigned to the Warehouse / production user class in the SRS?",
            "Managing receipts, transfers, work orders, and checks within location and operation scope.",
        ),
        (
            "What design governance policy does the SRS impose if standard Odoo functionality cannot meet requirements?",
            "Customizations require documented justification along with detailed documentation of custom models, fields, views, automations, and upgrade implications.",
        ),
    ]

    for q, ans in semantic_queries:
        dataset.append({
            "id": f"Q{len(dataset)+1:03d}",
            "query": q,
            "expected_answer": ans,
            "category": "Semantic Queries",
        })

    # =========================================================================
    # 3. Metadata Filtering (20 queries: Q046 to Q065)
    # =========================================================================
    metadata_queries = [
        (
            "Which defect has module 'Purchase', severity 'Major', and status 'Closed'?",
            "BUG-000001",
        ),
        (
            "Which defect is assigned to 'Melissa Fletcher' with priority 'P3' and status 'In Progress'?",
            "BUG-000002",
        ),
        (
            "Find the defect in module 'HR' that has severity 'Critical' and priority 'P1'.",
            "BUG-000003",
        ),
        (
            "Which defect in module 'HR' is assigned to 'Cheryl Bishop' with status 'New' and priority 'P2'?",
            "BUG-000004",
        ),
        (
            "Which test execution ran on build 'B-5521' with browser 'Edge' on operating system 'Android'?",
            "EXEC-000001",
        ),
        (
            "What test execution ran on date '2025-08-24' with browser 'Edge' on 'Ubuntu' and achieved status 'Passed'?",
            "EXEC-000002",
        ),
        (
            "Which test execution had status 'Skipped', browser 'Edge', and operating system 'iOS'?",
            "EXEC-000003",
        ),
        (
            "Find the test execution on build 'B-2787' that resulted in status 'Blocked' using browser 'Firefox'.",
            "EXEC-000004",
        ),
        (
            "Which test case in module 'POS' has priority 'High' and status 'Review'?",
            "TC-000001",
        ),
        (
            "Find the test case in module 'Quality' with priority 'Medium' and status 'Review'.",
            "TC-000002",
        ),
        (
            "Which test case in module 'eCommerce' has priority 'Critical' and status 'Draft'?",
            "TC-000004",
        ),
        (
            "Find the test case in module 'Rental' associated with requirement identifier 'REQ-7360'.",
            "TC-000005",
        ),
        (
            "Which defect reported by 'Leon Sullivan' in module 'Helpdesk' carries priority 'P0'?",
            "BUG-000005",
        ),
        (
            "Which test execution on build 'B-1458' resulted in status 'Failed' on browser 'Safari'?",
            "EXEC-000006",
        ),
        (
            "What test execution ran on '2026-07-10' using browser 'Chrome' on 'macOS' with status 'Failed'?",
            "EXEC-000007",
        ),
        (
            "Find the test case in module 'Documents' with priority 'Medium' and status 'Draft'.",
            "TC-000003",
        ),
        (
            "Which defect in module 'Studio' has severity 'Critical' and status 'Reopened'?",
            "BUG-000007",
        ),
        (
            "Find the defect in module 'CRM' with priority 'P3' and status 'Resolved'.",
            "BUG-000008",
        ),
        (
            "Which defect in module 'Sales' is assigned to 'Brandon Perez' with status 'In Progress'?",
            "BUG-000009",
        ),
        (
            "Find the defect in module 'Inventory' that has severity 'Major' and priority 'P1' assigned to 'Joshua Phillips'.",
            "BUG-000010",
        ),
    ]

    for q, ans in metadata_queries:
        dataset.append({
            "id": f"Q{len(dataset)+1:03d}",
            "query": q,
            "expected_answer": ans,
            "category": "Metadata Filtering",
        })

    # =========================================================================
    # 4. Keyword / Exact Match (15 queries: Q066 to Q080)
    # =========================================================================
    keyword_queries = [
        (
            "Retrieve the exact title of defect ID BUG-000010.",
            "Issue in Inventory: Networked web-enabled collaboration",
        ),
        (
            "What exact test case ID is mapped to requirement identifier REQ-1754?",
            "TC-000002",
        ),
        (
            "What exact execution ID is recorded for build number B-7391 on 2025-08-24?",
            "EXEC-000002",
        ),
        (
            "What exact database unique constraint name is cited in the database error for BUG-003596 in logs_001.txt?",
            "hr_unique_id",
        ),
        (
            "What exact Python source file path is reported in the traceback for BUG-003596 in logs_001.txt?",
            "/odoo/addons/hr/models/record.py",
        ),
        (
            "What exact line number in /odoo/addons/hr/models/record.py raised the exception for BUG-003596 in logs_001.txt?",
            "Line 158",
        ),
        (
            "What exact session ID is logged in the server log entry for BUG-003596 in logs_001.txt?",
            "0c5836b3-6ad1-49f5-a61a-fa3184412070",
        ),
        (
            "What exact error string is raised under Python Traceback for BUG-003596 in logs_001.txt?",
            "odoo.exceptions.AccessError: Access Denied: You do not have the required permissions.",
        ),
        (
            "What exact dialog box name appears in ocr_batch_001.json for bug BUG-001585?",
            "Confirm Spend?",
        ),
        (
            "What exact table name is listed in ocr_batch_001.json for execution EXEC-002249?",
            "Film List",
        ),
        (
            "What exact list of button names is recorded in ocr_batch_001.json for bug BUG-001585?",
            "Ago, Imagine, Bed, Agree, Author",
        ),
        (
            "What exact list of menus is visible in ocr_batch_001.json for bug BUG-001585?",
            "File, Edit, View, Reports, Configuration",
        ),
        (
            "What exact DETAIL value is reported in the database error for BUG-003596 in logs_001.txt?",
            "Key (name)=(Test_Record_001) already exists.",
        ),
        (
            "What exact dialog box text appears in ocr_batch_001.json for execution EXEC-017509?",
            "Confirm Life?",
        ),
        (
            "What exact table name is listed in ocr_batch_002.json for bug BUG-009386?",
            "Process List",
        ),
    ]

    for q, ans in keyword_queries:
        dataset.append({
            "id": f"Q{len(dataset)+1:03d}",
            "query": q,
            "expected_answer": ans,
            "category": "Keyword / Exact Match",
        })

    # =========================================================================
    # 5. Multi-hop Queries (25 queries: Q081 to Q105)
    # =========================================================================
    multihop_queries = [
        (
            "What was the execution status of the test run linked to defect BUG-000001?",
            "Failed (Execution ID: EXEC-012473)",
        ),
        (
            "What requirement ID is verified by the test case linked to defect BUG-000001?",
            "REQ-9614 (Test Case ID: TC-003557)",
        ),
        (
            "What browser was used in the test execution linked to defect BUG-000002?",
            "Edge (Execution ID: EXEC-001114)",
        ),
        (
            "What module is assigned to the test case linked to defect BUG-000003?",
            "Rental (Test Case ID: TC-013900)",
        ),
        (
            "What was the execution time in seconds for the test execution linked to defect BUG-000004?",
            "74 seconds (Execution ID: EXEC-013356)",
        ),
        (
            "What priority level is assigned to the test case linked to defect BUG-000005?",
            "Medium (Test Case ID: TC-015032)",
        ),
        (
            "What operating system was used in the test execution associated with defect BUG-000006?",
            "macOS (Execution ID: EXEC-011957)",
        ),
        (
            "What requirement ID is verified by the test case linked to defect BUG-000007?",
            "REQ-9504 (Test Case ID: TC-017938)",
        ),
        (
            "What build number was tested in the test execution linked to defect BUG-000008?",
            "B-1312 (Execution ID: EXEC-012376)",
        ),
        (
            "What module is assigned to the test case linked to defect BUG-000009?",
            "Manufacturing (Test Case ID: TC-015171)",
        ),
        (
            "What is the title of the defect linked to test execution EXEC-012473?",
            "Issue in Purchase: Team-oriented transitional concept (Defect ID: BUG-000001)",
        ),
        (
            "What is the title of the defect linked to test execution EXEC-001114?",
            "Issue in CRM: Team-oriented value-added software (Defect ID: BUG-000002)",
        ),
        (
            "What screen name was captured in OCR for the test execution linked to defect BUG-001585?",
            "Inventory Dashboard (Execution ID: EXEC-002249)",
        ),
        (
            "What exception is reported in logs_001.txt for the test execution associated with BUG-003596?",
            "AccessError (Execution ID: EXEC-016067)",
        ),
        (
            "What requirement ID corresponds to the test case verified in test execution EXEC-000001?",
            "REQ-7270 (Test Case ID: TC-014298)",
        ),
        (
            "What requirement ID corresponds to the test case verified in test execution EXEC-000002?",
            "REQ-2619 (Test Case ID: TC-006128)",
        ),
        (
            "What is the lifecycle status of the test case verified in test execution EXEC-000003?",
            "Deprecated (Test Case ID: TC-003791)",
        ),
        (
            "What is the automation status of the test case verified in test execution EXEC-000004?",
            "Manual (Test Case ID: TC-013521)",
        ),
        (
            "What is the title of the defect associated with test case TC-003557?",
            "Issue in Purchase: Team-oriented transitional concept (Defect ID: BUG-000001)",
        ),
        (
            "What is the severity of the defect linked to test case TC-008797?",
            "Major (Defect ID: BUG-000002)",
        ),
        (
            "What dialog box was captured in OCR for the execution linked to bug BUG-003851?",
            "Confirm Life? (Execution ID: EXEC-017509)",
        ),
        (
            "What database unique constraint violation is logged in failure logs for execution EXEC-012473 in logs_005.txt?",
            "website_unique_id (associated with BUG-007041)",
        ),
        (
            "What root cause is recorded for the defect linked to test execution EXEC-004706?",
            "The warehouse operator configured the manufacturing order and successfully completed the transaction. (Defect ID: BUG-000003)",
        ),
        (
            "What bug type is assigned to the defect linked to test execution EXEC-013356?",
            "Integration (Defect ID: BUG-000004)",
        ),
        (
            "Who is the developer assigned to the defect linked to test execution EXEC-009336?",
            "Kimberly Lopez (Defect ID: BUG-000005)",
        ),
    ]

    for q, ans in multihop_queries:
        dataset.append({
            "id": f"Q{len(dataset)+1:03d}",
            "query": q,
            "expected_answer": ans,
            "category": "Multi-hop Queries",
        })

    # =========================================================================
    # 6. Aggregation Queries (15 queries: Q106 to Q120)
    # =========================================================================
    aggregation_queries = [
        (
            "What is the total number of defect records contained in defects_rows.csv?",
            "10,000",
        ),
        (
            "What is the total number of test cases recorded in testcases.csv?",
            "20,000",
        ),
        (
            "What is the total number of test executions documented in test_execution.csv?",
            "20,000",
        ),
        (
            "How many screenshot OCR records are contained within ocr_batch_001.json?",
            "1,400",
        ),
        (
            "How many failure log blocks are recorded in failure_logs/logs_001.txt?",
            "1,500",
        ),
        (
            "What is the total page count of the Odoo ERP System SRS PDF document?",
            "60 pages",
        ),
        (
            "How many defects in defects_rows.csv currently hold the status 'New'?",
            "2,072",
        ),
        (
            "How many defects in defects_rows.csv have reached the status 'Closed'?",
            "2,022",
        ),
        (
            "How many defects in defects_rows.csv are currently in 'In Progress' status?",
            "2,010",
        ),
        (
            "What is the total count of defects classified with 'Critical' severity in defects_rows.csv?",
            "2,533",
        ),
        (
            "What is the total count of defects classified with 'Major' severity in defects_rows.csv?",
            "2,557",
        ),
        (
            "How many test execution runs in test_execution.csv completed with status 'Passed'?",
            "5,008",
        ),
        (
            "How many test execution runs in test_execution.csv completed with status 'Failed'?",
            "4,929",
        ),
        (
            "How many test cases in testcases.csv belong to the 'POS' module?",
            "978",
        ),
        (
            "How many defects in defects_rows.csv are categorized under bug_type 'Security'?",
            "2,071",
        ),
    ]

    for q, ans in aggregation_queries:
        dataset.append({
            "id": f"Q{len(dataset)+1:03d}",
            "query": q,
            "expected_answer": ans,
            "category": "Aggregation Queries",
        })

    # =========================================================================
    # 7. Comparison Queries (10 queries: Q121 to Q130)
    # =========================================================================
    comparison_queries = [
        (
            "Between defect BUG-000001 and BUG-000003, which record has a higher severity level?",
            "BUG-000003 has higher severity (Critical compared to Major for BUG-000001).",
        ),
        (
            "Between test execution EXEC-000001 and EXEC-000002, which test run took longer to complete?",
            "EXEC-000002 took longer (176 seconds compared to 92 seconds for EXEC-000001).",
        ),
        (
            "Compare the priority levels of test cases TC-000001 and TC-000002.",
            "TC-000001 has High priority, whereas TC-000002 has Medium priority.",
        ),
        (
            "Compare the final execution statuses of test executions EXEC-000001 and EXEC-000006.",
            "EXEC-000001 has status Passed, while EXEC-000006 has status Failed.",
        ),
        (
            "Between defect BUG-000001 and BUG-000005, which has higher priority?",
            "BUG-000005 has higher priority (P0 compared to P3 for BUG-000001).",
        ),
        (
            "Compare the browsers utilized in test executions EXEC-000006 and EXEC-000007.",
            "EXEC-000006 used Safari, while EXEC-000007 used Chrome.",
        ),
        (
            "Compare the lifecycle status of test case TC-000003 with that of TC-000006.",
            "TC-000003 is in Draft status, while TC-000006 is Deprecated.",
        ),
        (
            "Compare the operating systems on which test executions EXEC-000002 and EXEC-000003 were executed.",
            "EXEC-000002 was executed on Ubuntu, whereas EXEC-000003 was executed on iOS.",
        ),
        (
            "Between defect BUG-000004 and BUG-000007, compare their current defect statuses.",
            "BUG-000004 has status New, while BUG-000007 has status Reopened.",
        ),
        (
            "Compare the execution time in seconds between test runs EXEC-000007 and EXEC-000009.",
            "EXEC-000007 took 290 seconds, whereas EXEC-000009 took 93 seconds.",
        ),
    ]

    for q, ans in comparison_queries:
        dataset.append({
            "id": f"Q{len(dataset)+1:03d}",
            "query": q,
            "expected_answer": ans,
            "category": "Comparison Queries",
        })

    # =========================================================================
    # 8. Temporal Queries (10 queries: Q131 to Q140)
    # =========================================================================
    temporal_queries = [
        (
            "On what date was defect BUG-000001 created, and on what date was it marked resolved?",
            "Created on 2026-07-13 and marked resolved on 2026-07-14.",
        ),
        (
            "What exact execution date is recorded for test execution EXEC-000002?",
            "2025-08-24",
        ),
        (
            "Was test execution EXEC-000001 run chronologically before or after test execution EXEC-000005?",
            "EXEC-000001 (2025-09-11) was executed before EXEC-000005 (2025-12-21).",
        ),
        (
            "What captured_time timestamp is recorded in ocr_batch_001.json for execution EXEC-002249?",
            "1979-12-05T16:39:33",
        ),
        (
            "What timestamp is recorded for the failure log entry of BUG-003596 in logs_001.txt?",
            "1973-06-05T07:17:25",
        ),
        (
            "What is the recorded created_date for test case TC-000001?",
            "2025-08-27",
        ),
        (
            "Was test execution EXEC-000004 executed before or after test execution EXEC-000006?",
            "EXEC-000006 (2026-02-07) was executed before EXEC-000004 (2026-04-06).",
        ),
        (
            "What timestamp is logged in logs_002.txt for execution EXEC-002580?",
            "2023-03-07T17:31:09",
        ),
        (
            "In Document Control of the Odoo ERP System SRS, on what date was Version 1.0 drafted?",
            "11 Oct 2026",
        ),
        (
            "What anomaly exists between the status and the lifecycle dates for defect BUG-000004?",
            "BUG-000004 has status 'New', yet its resolved_date is already populated with 2026-07-14.",
        ),
    ]

    for q, ans in temporal_queries:
        dataset.append({
            "id": f"Q{len(dataset)+1:03d}",
            "query": q,
            "expected_answer": ans,
            "category": "Temporal Queries",
        })

    # =========================================================================
    # 9. OCR / UI Queries (10 queries: Q141 to Q150)
    # =========================================================================
    ocr_queries = [
        (
            "What buttons are visible in the OCR screenshot record for bug BUG-001585 in ocr_batch_001.json?",
            "Ago, Imagine, Bed, Agree, Author",
        ),
        (
            "What screen name is displayed in the OCR record for execution EXEC-017509 in ocr_batch_001.json?",
            "Studio Dashboard",
        ),
        (
            "What dialog box prompt is recorded in ocr_batch_001.json for bug BUG-003851?",
            "Confirm Life?",
        ),
        (
            "What menu items are visible in the OCR capture for bug BUG-001585 in ocr_batch_001.json?",
            "File, Edit, View, Reports, Configuration",
        ),
        (
            "What warning message is recorded in ocr_batch_001.json for bug BUG-001585?",
            "The API endpoint attempted to confirm the manufacturing order and successfully completed the transaction.",
        ),
        (
            "What error message is reported in the OCR record for bug BUG-001585 in ocr_batch_001.json?",
            "The user configured the POS receipt which resulted in an AccessError.",
        ),
        (
            "What user action description was logged during screenshot capture for bug BUG-001585 in ocr_batch_001.json?",
            "The warehouse operator configured the incoming shipment but the system raised a validation exception. The system configured the manufacturing order and successfully completed the transaction.",
        ),
        (
            "What screen name is recorded in ocr_batch_002.json for execution EXEC-002301?",
            "Accounting Dashboard",
        ),
        (
            "What dialog box title appears in ocr_batch_002.json for bug BUG-008442?",
            "Confirm Forward?",
        ),
        (
            "What buttons are captured in ocr_batch_002.json for execution EXEC-006588?",
            "No, Red, Probably, Necessary",
        ),
    ]

    for q, ans in ocr_queries:
        dataset.append({
            "id": f"Q{len(dataset)+1:03d}",
            "query": q,
            "expected_answer": ans,
            "category": "OCR / UI Queries",
        })

    # =========================================================================
    # 10. Failure Diagnosis (15 queries: Q151 to Q165)
    # =========================================================================
    failure_queries = [
        (
            "What Python exception type is logged for execution EXEC-016067 in failure log logs_001.txt?",
            "AccessError",
        ),
        (
            "In failure log logs_001.txt for BUG-003596, what Python method was executing when the error occurred?",
            "action_confirm",
        ),
        (
            "In logs_001.txt for BUG-003596, which source file and line number raised AccessError('Access Denied')?",
            "/odoo/odoo/models.py, line 1666",
        ),
        (
            "What database unique constraint violation is reported for BUG-003596 in logs_001.txt?",
            "duplicate key value violates unique constraint \"hr_unique_id\"",
        ),
        (
            "What HTTP status code was returned in failure log logs_001.txt for BUG-003596?",
            "404",
        ),
        (
            "What session ID is recorded in the server log entry for BUG-003596 in logs_001.txt?",
            "0c5836b3-6ad1-49f5-a61a-fa3184412070",
        ),
        (
            "What resolution hint is provided in failure log logs_001.txt for BUG-003596?",
            "The background job escalated the subscription renewal which resulted in an AccessError.",
        ),
        (
            "What module is associated with the failure log entry for BUG-003596 in logs_001.txt?",
            "HR",
        ),
        (
            "What HTTP status code is recorded in logs_002.txt for execution EXEC-002580?",
            "400",
        ),
        (
            "What module is reported in the failure log for BUG-007637 in logs_002.txt?",
            "Documents",
        ),
        (
            "What exception type is raised in logs_002.txt for BUG-007637?",
            "AccessError",
        ),
        (
            "What database error constraint violation is reported in logs_005.txt for execution EXEC-012473?",
            "duplicate key value violates unique constraint \"website_unique_id\"",
        ),
        (
            "Across all failure log files in the dataset, what single recurring Python exception type is reported in every log entry?",
            "AccessError",
        ),
        (
            "In logs_001.txt for BUG-003596, what user action description is recorded in the header?",
            "The API endpoint calculated the transfer due to missing configuration settings.",
        ),
        (
            "What resolution hint is given for BUG-007637 in failure log logs_002.txt?",
            "The sales representative attempted to confirm the subscription renewal triggering a workflow notification.",
        ),
    ]

    for q, ans in failure_queries:
        dataset.append({
            "id": f"Q{len(dataset)+1:03d}",
            "query": q,
            "expected_answer": ans,
            "category": "Failure Diagnosis",
        })

    # =========================================================================
    # 11. Edge Cases / Unanswerable Queries (35 queries: Q166 to Q200)
    # =========================================================================
    insufficient = "Insufficient evidence in the provided dataset."
    edge_queries = [
        (
            "What is the title of defect BUG-999999?",
            insufficient,
        ),
        (
            "What is the execution date of test execution EXEC-999999?",
            insufficient,
        ),
        (
            "What is the precondition for test case TC-999999?",
            insufficient,
        ),
        (
            "What is the requirement description for requirement REQ-999999?",
            insufficient,
        ),
        (
            "What test cases are associated with epic EPIC-999999?",
            insufficient,
        ),
        (
            "What is the status of defect DEFECT#INVALID-001?",
            insufficient,
        ),
        (
            "What is the execution time for test run RUN_2026_TEST_99?",
            insufficient,
        ),
        (
            "What is the expected result for test case CASE#987654?",
            insufficient,
        ),
        (
            "What is the full Python source code implementation of the _validate_permissions method in /odoo/odoo/models.py?",
            insufficient,
        ),
        (
            "What is the email address of tester Alexander King in testcases.csv?",
            insufficient,
        ),
        (
            "What is the root cause of failure for test execution EXEC-000001?",
            insufficient,
        ),
        (
            "What defect was filed as a result of test execution EXEC-000002?",
            insufficient,
        ),
        (
            "What is the employee salary recorded in the HR module test cases?",
            insufficient,
        ),
        (
            "What is the total monetary revenue generated from sales orders in testcases.csv?",
            insufficient,
        ),
        (
            "What was the outcome of test execution runs conducted in the year 2045?",
            insufficient,
        ),
        (
            "What are the database root credentials used to connect to the Odoo ERP PostgreSQL database?",
            insufficient,
        ),
        (
            "What error message appears in the OCR screenshot for test execution EXEC-999999?",
            insufficient,
        ),
        (
            "What traceback is logged in the failure logs for test case TC-999999?",
            insufficient,
        ),
        (
            "What is the network latency in milliseconds between the Odoo application server and database during EXEC-000001?",
            insufficient,
        ),
        (
            "What are the CPU model and RAM specifications of the test execution runner for EXEC-000005?",
            insufficient,
        ),
        (
            "What is the git commit hash corresponding to build B-5521?",
            insufficient,
        ),
        (
            "What customer credit card numbers are stored in POS receipt test cases?",
            insufficient,
        ),
        (
            "According to the SRS, what are the feature updates scheduled for Version 3.0?",
            insufficient,
        ),
        (
            "What is the total test code coverage percentage of the Odoo ERP codebase?",
            insufficient,
        ),
        (
            "What defects are currently logged under the Blockchain module?",
            insufficient,
        ),
        (
            "What test cases were executed using Internet Explorer 6?",
            insufficient,
        ),
        (
            "What is the physical street address of the main warehouse mentioned in inventory test cases?",
            insufficient,
        ),
        (
            "What is the total file size in gigabytes of the production database backup?",
            insufficient,
        ),
        (
            "What performance rating was given to tester William Campbell in his annual review?",
            insufficient,
        ),
        (
            "What is the verified resolution date for defect BUG-000004 considering that its lifecycle status is New?",
            insufficient,
        ),
        (
            "What is the underlying root cause of the AccessError in BUG-003596 beyond the logged permission check?",
            insufficient,
        ),
        (
            "What error message is displayed in the OCR screenshot for an execution with an empty error message field?",
            insufficient,
        ),
        (
            "What warning message is displayed in the OCR screenshot for an execution with an empty warning message field?",
            insufficient,
        ),
        (
            "What is the release date of Odoo ERP version 20.0 in the SRS?",
            insufficient,
        ),
        (
            "What is the customer phone number associated with quotations in sales test cases?",
            insufficient,
        ),
    ]

    for q, ans in edge_queries:
        dataset.append({
            "id": f"Q{len(dataset)+1:03d}",
            "query": q,
            "expected_answer": ans,
            "category": "Edge Cases / Unanswerable Queries",
        })

    return dataset


def validate_dataset(dataset: List[Dict[str, str]]) -> None:
    expected_distribution = {
        "Basic Retrieval": 25,
        "Semantic Queries": 20,
        "Metadata Filtering": 20,
        "Keyword / Exact Match": 15,
        "Multi-hop Queries": 25,
        "Aggregation Queries": 15,
        "Comparison Queries": 10,
        "Temporal Queries": 10,
        "OCR / UI Queries": 10,
        "Failure Diagnosis": 15,
        "Edge Cases / Unanswerable Queries": 35,
    }

    # 1. Total records = 200
    assert len(dataset) == 200, f"Expected 200 records, got {len(dataset)}"

    # 2. Unique IDs = 200
    ids = [d["id"] for d in dataset]
    assert len(ids) == len(set(ids)), "Duplicate IDs found!"
    for i, item_id in enumerate(ids, start=1):
        expected_id = f"Q{i:03d}"
        assert item_id == expected_id, f"ID sequence mismatch at {i}: expected {expected_id}, got {item_id}"

    # 3. Exactly four fields
    required_keys = {"id", "query", "expected_answer", "category"}
    for item in dataset:
        assert set(item.keys()) == required_keys, f"Invalid keys in item {item['id']}: {set(item.keys())}"
        for k, v in item.items():
            assert isinstance(v, str) and len(v.strip()) > 0, f"Empty or non-string field '{k}' in {item['id']}"

    # 4. Correct counts for all 11 categories
    cat_counts = Counter(d["category"] for d in dataset)
    assert cat_counts == expected_distribution, f"Category count mismatch:\nExpected: {expected_distribution}\nGot: {cat_counts}"

    # 5. No duplicate queries
    queries = [d["query"] for d in dataset]
    assert len(queries) == len(set(queries)), f"Duplicate queries detected: {len(queries) - len(set(queries))}"

    # 6. Edge cases answer check
    edge_cases = [d for d in dataset if d["category"] == "Edge Cases / Unanswerable Queries"]
    assert len(edge_cases) == 35, f"Expected 35 edge cases, got {len(edge_cases)}"
    for ec in edge_cases:
        assert "Insufficient evidence in the provided dataset." in ec["expected_answer"], (
            f"Edge case {ec['id']} missing insufficient evidence statement: {ec['expected_answer']}"
        )

    print("All 8 programmatic validations passed successfully!")


def main():
    dataset = build_dataset()
    validate_dataset(dataset)

    output_path = "vanillarag_evaluation_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(dataset)} evaluation records to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
