"""Unit tests for relationship validation and conflict flagging."""

import pytest
from src.ingestion.validators import CrossFileValidator


def test_validator_preserves_conflicting_values():
    validator = CrossFileValidator()

    # Canonical sets
    validator.register_test_cases([{"test_case_id": "TC-001"}])
    validator.register_executions([{"execution_id": "EXEC-001", "test_case_id": "TC-001"}])
    validator.register_defects([
        {"defect_id": "BUG-001", "linked_test_case": "TC-001", "linked_execution": "EXEC-001"}
    ])

    # Inconsistent defect referencing non-existent execution and mismatched test case
    inconsistent_defects = [
        {
            "defect_id": "BUG-999",
            "linked_test_case": "TC-999",  # Non-existent
            "linked_execution": "EXEC-001",  # Maps to TC-001, mismatch!
        }
    ]

    report = validator.validate_all(
        test_cases=[],
        executions=[],
        defects=inconsistent_defects,
        failure_logs=[],
        screenshots=[],
    )

    assert report.inconsistent_records_count == 1
    # Check that original values are NOT mutated or overwritten!
    bug = inconsistent_defects[0]
    assert bug["linked_test_case"] == "TC-999"
    assert bug["linked_execution"] == "EXEC-001"
    assert bug["is_cross_consistent"] is False
    assert len(bug["inconsistency_reasons"]) > 0


def test_validator_screenshot_conflicts():
    validator = CrossFileValidator()
    validator.register_test_cases([{"test_case_id": "TC-001"}])
    validator.register_executions([{"execution_id": "EXEC-001", "test_case_id": "TC-001"}])

    screenshot = [
        {
            "screenshot_id": "shot_01",
            "defect_id": "BUG-UNKNOWN",
            "execution_id": "EXEC-001",
            "test_case_id": "TC-DIFFERENT",  # Conflicts with EXEC-001's TC-001
        }
    ]

    report = validator.validate_all(
        test_cases=[],
        executions=[],
        defects=[],
        failure_logs=[],
        screenshots=screenshot,
    )

    assert report.inconsistent_records_count == 1
    assert screenshot[0]["test_case_id"] == "TC-DIFFERENT"
    assert screenshot[0]["is_cross_consistent"] is False
