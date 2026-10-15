"""Forms logical LangChain Document objects from raw extracted records."""

from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

from src.ingestion.cleaners import DataCleaner
from src.ingestion.logging_config import logger


class DocumentBuilder:
    """Constructs readable, retrieval-optimized LangChain Documents for all record types."""

    @staticmethod
    def build_test_case_doc(raw_record: Dict[str, Any], source_file: str) -> Document:
        tc_id = DataCleaner.clean_id(raw_record.get("test_case_id")) or "UNKNOWN_TC"
        req_id = DataCleaner.clean_id(raw_record.get("requirement_id"))
        feature = raw_record.get("feature") or ""
        module = raw_record.get("module") or ""
        submodule = raw_record.get("submodule") or ""
        epic = raw_record.get("epic") or ""
        story = raw_record.get("story") or ""
        priority = raw_record.get("priority") or ""
        severity = raw_record.get("severity") or ""
        status = raw_record.get("status") or ""
        precondition = raw_record.get("precondition") or ""
        test_steps = raw_record.get("test_steps") or ""
        expected = raw_record.get("expected_result") or ""
        actual = raw_record.get("actual_result") or ""
        acceptance = raw_record.get("acceptance_criteria") or ""
        tester = raw_record.get("tester") or ""
        env = raw_record.get("environment") or ""

        # Format retrieval-optimized readable representation
        content_lines = [
            f"# Test Case: {tc_id}",
            f"**Requirement ID**: {req_id or 'None'}",
            f"**Module / Submodule**: {module} / {submodule}" if submodule else f"**Module**: {module}",
            f"**Feature**: {feature}",
            f"**Epic / Story**: {epic} / {story}" if epic or story else "",
            f"**Priority / Severity**: {priority} / {severity}",
            f"**Status**: {status}",
            f"**Preconditions**:\n{precondition}",
            f"**Test Steps**:\n{test_steps}",
            f"**Expected Result**:\n{expected}",
            f"**Actual Result**:\n{actual}" if actual else "",
            f"**Acceptance Criteria**:\n{acceptance}" if acceptance else "",
            f"**Assigned Tester**: {tester}",
            f"**Environment**: {env}",
        ]
        page_content = "\n\n".join(line for line in content_lines if line)

        # Store raw record in document metadata temporarily for enricher
        metadata = {
            "source": source_file,
            "raw_record": raw_record,
            "record_id": tc_id,
            "document_type": "test_case",
        }
        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def build_test_execution_doc(raw_record: Dict[str, Any], source_file: str) -> Document:
        exec_id = DataCleaner.clean_id(raw_record.get("execution_id")) or "UNKNOWN_EXEC"
        tc_id = DataCleaner.clean_id(raw_record.get("test_case_id")) or "UNKNOWN_TC"
        build_no = raw_record.get("build_number") or ""
        exec_date = raw_record.get("execution_date") or ""
        tester = raw_record.get("tester") or ""
        env = raw_record.get("environment") or ""
        browser = raw_record.get("browser") or ""
        os_name = raw_record.get("operating_system") or ""
        exec_time = raw_record.get("execution_time_seconds")
        status = raw_record.get("status") or ""
        notes = raw_record.get("execution_notes") or ""

        content_lines = [
            f"# Test Execution: {exec_id}",
            f"**Linked Test Case**: {tc_id}",
            f"**Build Number**: {build_no}",
            f"**Execution Date**: {exec_date}",
            f"**Status**: {status}",
            f"**Tester**: {tester}",
            f"**Environment**: {env}",
            f"**Browser / OS**: {browser} on {os_name}",
            f"**Execution Time**: {exec_time} seconds" if exec_time is not None else "",
            f"**Execution Notes**:\n{notes}" if notes else "",
        ]
        page_content = "\n\n".join(line for line in content_lines if line)

        metadata = {
            "source": source_file,
            "raw_record": raw_record,
            "record_id": exec_id,
            "document_type": "test_execution",
        }
        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def build_defect_doc(raw_record: Dict[str, Any], source_file: str) -> Document:
        defect_id = DataCleaner.clean_id(raw_record.get("defect_id")) or "UNKNOWN_BUG"
        tc_id = DataCleaner.clean_id(raw_record.get("linked_test_case")) or ""
        exec_id = DataCleaner.clean_id(raw_record.get("linked_execution")) or ""
        title = raw_record.get("title") or ""
        desc = raw_record.get("description") or ""
        severity = raw_record.get("severity") or ""
        priority = raw_record.get("priority") or ""
        module = raw_record.get("module") or ""
        status = raw_record.get("status") or ""
        root_cause = raw_record.get("root_cause") or ""
        bug_type = raw_record.get("bug_type") or ""
        reproducibility = raw_record.get("reproducibility") or ""
        found_in = raw_record.get("found_in_build") or ""
        fixed_in = raw_record.get("fixed_build") or ""
        fix_ver = raw_record.get("fix_version") or ""
        env = raw_record.get("environment") or ""
        comments = raw_record.get("comments") or ""

        content_lines = [
            f"# Defect: {defect_id} - {title}",
            f"**Linked Test Case**: {tc_id or 'None'}",
            f"**Linked Execution**: {exec_id or 'None'}",
            f"**Module**: {module}",
            f"**Severity / Priority**: {severity} / {priority}",
            f"**Status**: {status}",
            f"**Bug Type**: {bug_type}",
            f"**Reproducibility**: {reproducibility}",
            f"**Environment**: {env}",
            f"**Found In Build / Fixed In Build**: {found_in} / {fixed_in}",
            f"**Fix Version**: {fix_ver}",
            f"**Description**:\n{desc}",
            f"**Root Cause Analysis**:\n{root_cause}" if root_cause else "",
            f"**Comments**:\n{comments}" if comments else "",
        ]
        page_content = "\n\n".join(line for line in content_lines if line)

        metadata = {
            "source": source_file,
            "raw_record": raw_record,
            "record_id": defect_id,
            "document_type": "defect",
        }
        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def build_srs_doc(raw_record: Dict[str, Any], source_file: str) -> Document:
        title = raw_record.get("section_title") or "SRS Section"
        sec_num = raw_record.get("section_number") or ""
        req_id = raw_record.get("requirement_id")
        page_num = raw_record.get("page_number", 1)
        approval_status = raw_record.get("approval_status") or "Unknown"
        text = raw_record.get("content") or ""

        record_id = req_id if req_id else f"SRS_P{page_num}_{sec_num or 'sec'}"

        content_lines = [
            f"# SRS Section: {title}" + (f" ({sec_num})" if sec_num else ""),
            f"**Document**: {source_file} (Page {page_num})",
            f"**Requirement ID**: {req_id}" if req_id else "",
            f"**Approval Status**: {approval_status}",
            f"**Section Content**:\n{text}",
        ]
        page_content = "\n\n".join(line for line in content_lines if line)

        metadata = {
            "source": source_file,
            "raw_record": raw_record,
            "record_id": record_id,
            "document_type": "srs_section",
        }
        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def build_failure_log_doc(raw_record: Dict[str, Any], source_file: str) -> Document:
        log_id = raw_record.get("log_id") or "UNKNOWN_LOG"
        bug_id = raw_record.get("defect_id") or ""
        exec_id = raw_record.get("execution_id") or ""
        tc_id = raw_record.get("test_case_id") or ""
        ts = raw_record.get("timestamp") or ""
        module = raw_record.get("module") or ""
        user_action = raw_record.get("user_action") or ""
        traceback = raw_record.get("traceback") or ""
        db_err = raw_record.get("database_error") or ""
        http_status = raw_record.get("http_status")
        server_log = raw_record.get("server_log") or ""
        exc_type = raw_record.get("exception_type") or ""
        hint = raw_record.get("resolution_hint") or ""
        line_start = raw_record.get("line_start", 1)
        line_end = raw_record.get("line_end", 1)

        content_lines = [
            f"# Failure Log Event: {log_id}",
            f"**Source Log**: {source_file} (Lines {line_start}-{line_end})",
            f"**Timestamp**: {ts}",
            f"**Linked Defect**: {bug_id or 'None'}",
            f"**Linked Execution**: {exec_id or 'None'}",
            f"**Linked Test Case**: {tc_id or 'None'}",
            f"**Module**: {module}",
            f"**HTTP Status**: {http_status}" if http_status is not None else "",
            f"**Exception Type**: {exc_type}" if exc_type else "",
            f"**User Action**:\n{user_action}" if user_action else "",
            f"**Python Traceback**:\n```\n{traceback}\n```" if traceback else "",
            f"**Database Error**:\n{db_err}" if db_err else "",
            f"**Server Log**:\n{server_log}" if server_log else "",
            f"**Resolution Hint**:\n{hint}" if hint else "",
        ]
        page_content = "\n\n".join(line for line in content_lines if line)

        metadata = {
            "source": source_file,
            "raw_record": raw_record,
            "record_id": log_id,
            "document_type": "failure_log",
        }
        return Document(page_content=page_content, metadata=metadata)

    @staticmethod
    def build_screenshot_doc(raw_record: Dict[str, Any], source_file: str) -> Document:
        shot_id = raw_record.get("screenshot_id") or "UNKNOWN_SHOT"
        bug_id = raw_record.get("defect_id") or ""
        exec_id = raw_record.get("execution_id") or ""
        tc_id = raw_record.get("test_case_id") or ""
        screen_name = raw_record.get("screen_name") or ""
        module = raw_record.get("module") or ""
        captured_time = raw_record.get("captured_time") or ""
        user_action = raw_record.get("user_action") or ""

        buttons = raw_record.get("buttons") or []
        labels = raw_record.get("labels") or []
        menus = raw_record.get("menus") or []
        dialog_boxes = raw_record.get("dialog_boxes") or []
        tables = raw_record.get("tables") or []
        warning_messages = raw_record.get("warning_messages") or []
        error_messages = raw_record.get("error_messages") or []
        field_values = raw_record.get("field_values") or {}

        content_lines = [
            f"# Screenshot & OCR Evidence: {screen_name or shot_id}",
            f"**Evidence ID**: {shot_id}",
            f"**Source File**: {source_file}",
            f"**Captured Time**: {captured_time}",
            f"**Linked Defect**: {bug_id or 'None'}",
            f"**Linked Execution**: {exec_id or 'None'}",
            f"**Linked Test Case**: {tc_id or 'None'}",
            f"**Module**: {module}",
            f"**User Action**: {user_action}" if user_action else "",
            f"**Error Messages**:\n" + "\n".join(f"- {e}" for e in error_messages) if error_messages else "",
            f"**Warning Messages**:\n" + "\n".join(f"- {w}" for w in warning_messages) if warning_messages else "",
            f"**Dialog Boxes**: {', '.join(dialog_boxes)}" if dialog_boxes else "",
            f"**Active Menus**: {', '.join(menus)}" if menus else "",
            f"**UI Buttons**: {', '.join(buttons)}" if buttons else "",
            f"**Labels**: {', '.join(labels)}" if labels else "",
            f"**Tables**: {', '.join(tables)}" if tables else "",
            f"**Field Values**:\n" + "\n".join(f"- {k}: {v}" for k, v in field_values.items()) if field_values else "",
        ]
        page_content = "\n\n".join(line for line in content_lines if line)

        metadata = {
            "source": source_file,
            "raw_record": raw_record,
            "record_id": shot_id,
            "document_type": "screenshot_evidence",
        }
        return Document(page_content=page_content, metadata=metadata)
