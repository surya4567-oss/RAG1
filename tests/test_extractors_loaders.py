"""Unit tests for extractors and data loaders."""

import pytest
from pathlib import Path

from src.ingestion.extractors import FailureLogExtractor, ScreenshotExtractor, SRSExtractor
from src.ingestion.loaders import DatasetLoader


def test_failure_log_extractor():
    sample_text = """
======================================================================
Bug ID: BUG-001234 | Execution ID: EXEC-005678 | Test Case ID: TC-009999
Timestamp: 2026-04-02T12:28:37
Module: Accounting
User Action: User confirmed invoice.
----------------------------------------------------------------------
Python Traceback:
Traceback (most recent call last):
  File "models/invoice.py", line 12, in confirm
    raise AccessError("Access Denied")
AccessError: Access Denied

Database Error:
ERROR: unique constraint violation

HTTP Status: 403

Server Log:
INFO accounting - Access denied

Exception: AccessError
Resolution Hint: Verify user role permissions.
======================================================================
"""
    tmp_file = Path("tests/sample_log.txt")
    tmp_file.write_text(sample_text, encoding="utf-8")

    try:
        events = FailureLogExtractor.extract_from_file(tmp_file)
        assert len(events) == 1
        ev = events[0]
        assert ev["defect_id"] == "BUG-001234"
        assert ev["execution_id"] == "EXEC-005678"
        assert ev["test_case_id"] == "TC-009999"
        assert ev["module"] == "Accounting"
        assert ev["http_status"] == 403
        assert ev["exception_type"] == "AccessError"
        assert "Access Denied" in ev["traceback"]
        assert ev["line_start"] == 3
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


def test_screenshot_extractor():
    sample_records = [
        {
            "bug_id": "BUG-000001",
            "execution_id": "EXEC-000001",
            "test_case_id": "TC-000001",
            "screen_name": "Login Screen",
            "module": "Auth",
            "buttons": ["Submit", "Cancel"],
            "labels": ["Username", "Password"],
            "error_messages": ["Invalid password"],
            "warning_messages": [],
            "captured_time": "2026-05-01T10:00:00",
            "user_action": "Click submit",
        }
    ]
    extracted = ScreenshotExtractor.extract_from_records(sample_records, "test.json")
    assert len(extracted) == 1
    rec = extracted[0]
    assert rec["defect_id"] == "BUG-000001"
    assert rec["has_error_message"] is True
    assert rec["has_warning_message"] is False
    assert "Submit" in rec["buttons"]


def test_srs_extractor():
    pdf_path = Path("data/SRS.pdf")
    if pdf_path.exists():
        sections = SRSExtractor.extract_from_file(pdf_path)
        assert len(sections) > 0
        first_sec = sections[0]
        assert "page_number" in first_sec
        assert "content" in first_sec
        assert first_sec["page_number"] == 1
