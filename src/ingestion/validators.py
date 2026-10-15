"""Data quality and cross-file relationship validator."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from src.ingestion.logging_config import logger


@dataclass
class ValidationIssue:
    source_file: str
    record_id: str
    field_name: str
    issue_type: str  # missing_reference, conflict, format_error
    description: str
    observed_value: Any


@dataclass
class RelationshipReport:
    total_records_checked: int = 0
    consistent_records_count: int = 0
    inconsistent_records_count: int = 0
    missing_test_case_refs: int = 0
    missing_execution_refs: int = 0
    missing_defect_refs: int = 0
    missing_requirement_refs: int = 0
    cross_link_mismatches: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_checked": self.total_records_checked,
            "consistent_count": self.consistent_records_count,
            "inconsistent_count": self.inconsistent_records_count,
            "missing_test_case_refs": self.missing_test_case_refs,
            "missing_execution_refs": self.missing_execution_refs,
            "missing_defect_refs": self.missing_defect_refs,
            "missing_requirement_refs": self.missing_requirement_refs,
            "cross_link_mismatches": self.cross_link_mismatches,
            "sample_issues": [
                {
                    "source": iss.source_file,
                    "record_id": iss.record_id,
                    "type": iss.issue_type,
                    "description": iss.description,
                }
                for iss in self.issues[:20]
            ],
        }


class CrossFileValidator:
    """Validates cross-dataset referential integrity without mutating source values."""

    def __init__(self):
        self.test_case_ids: Set[str] = set()
        self.execution_ids: Set[str] = set()
        self.defect_ids: Set[str] = set()
        self.srs_req_ids: Set[str] = set()

        # Canonical relationship mappings
        self.execution_to_tc: Dict[str, str] = {}
        self.defect_to_tc: Dict[str, str] = {}
        self.defect_to_exec: Dict[str, str] = {}

    def register_test_cases(self, test_cases: List[Dict[str, Any]]):
        for tc in test_cases:
            tc_id = tc.get("test_case_id")
            if tc_id:
                self.test_case_ids.add(str(tc_id).strip())

    def register_executions(self, executions: List[Dict[str, Any]]):
        for ex in executions:
            ex_id = ex.get("execution_id")
            tc_id = ex.get("test_case_id")
            if ex_id:
                ex_id_str = str(ex_id).strip()
                self.execution_ids.add(ex_id_str)
                if tc_id:
                    self.execution_to_tc[ex_id_str] = str(tc_id).strip()

    def register_defects(self, defects: List[Dict[str, Any]]):
        for bug in defects:
            b_id = bug.get("defect_id")
            tc_id = bug.get("linked_test_case")
            ex_id = bug.get("linked_execution")
            if b_id:
                b_id_str = str(b_id).strip()
                self.defect_ids.add(b_id_str)
                if tc_id:
                    self.defect_to_tc[b_id_str] = str(tc_id).strip()
                if ex_id:
                    self.defect_to_exec[b_id_str] = str(ex_id).strip()

    def register_srs_sections(self, srs_sections: List[Dict[str, Any]]):
        for sec in srs_sections:
            req_id = sec.get("requirement_id")
            if req_id:
                self.srs_req_ids.add(str(req_id).strip())

    def validate_all(
        self,
        test_cases: List[Dict[str, Any]],
        executions: List[Dict[str, Any]],
        defects: List[Dict[str, Any]],
        failure_logs: List[Dict[str, Any]],
        screenshots: List[Dict[str, Any]],
    ) -> RelationshipReport:
        report = RelationshipReport()
        logger.info("Starting cross-dataset referential integrity validation")

        # 1. Validate Executions
        for ex in executions:
            report.total_records_checked += 1
            ex_id = ex.get("execution_id") or "UNKNOWN"
            tc_id = ex.get("test_case_id")
            inconsistencies = []

            if not tc_id or tc_id not in self.test_case_ids:
                report.missing_test_case_refs += 1
                inconsistencies.append(f"Referenced test_case_id '{tc_id}' not found in test cases")

            if inconsistencies:
                report.inconsistent_records_count += 1
                report.issues.append(
                    ValidationIssue(
                        source_file=ex.get("source", "test_execution.csv"),
                        record_id=str(ex_id),
                        field_name="test_case_id",
                        issue_type="missing_reference",
                        description="; ".join(inconsistencies),
                        observed_value=tc_id,
                    )
                )
                ex["is_cross_consistent"] = False
                ex["inconsistency_reasons"] = inconsistencies
            else:
                report.consistent_records_count += 1
                ex["is_cross_consistent"] = True
                ex["inconsistency_reasons"] = []

        # 2. Validate Defects
        for bug in defects:
            report.total_records_checked += 1
            bug_id = bug.get("defect_id") or "UNKNOWN"
            tc_id = bug.get("linked_test_case")
            ex_id = bug.get("linked_execution")
            inconsistencies = []

            if tc_id and tc_id not in self.test_case_ids:
                report.missing_test_case_refs += 1
                inconsistencies.append(f"Linked test_case '{tc_id}' not in canonical test cases")
            if ex_id and ex_id not in self.execution_ids:
                report.missing_execution_refs += 1
                inconsistencies.append(f"Linked execution '{ex_id}' not in canonical executions")

            # Check cross-consistency between execution and test case
            if ex_id and ex_id in self.execution_to_tc and tc_id:
                exec_tc = self.execution_to_tc[ex_id]
                if exec_tc != tc_id:
                    report.cross_link_mismatches += 1
                    inconsistencies.append(
                        f"Execution {ex_id} belongs to {exec_tc}, but defect links to {tc_id}"
                    )

            if inconsistencies:
                report.inconsistent_records_count += 1
                report.issues.append(
                    ValidationIssue(
                        source_file=bug.get("source", "defects_rows.csv"),
                        record_id=str(bug_id),
                        field_name="linked_ids",
                        issue_type="conflict" if "mismatches" in str(inconsistencies) else "missing_reference",
                        description="; ".join(inconsistencies),
                        observed_value={"test_case": tc_id, "execution": ex_id},
                    )
                )
                bug["is_cross_consistent"] = False
                bug["inconsistency_reasons"] = inconsistencies
            else:
                report.consistent_records_count += 1
                bug["is_cross_consistent"] = True
                bug["inconsistency_reasons"] = []

        # 3. Validate Failure Logs
        for log in failure_logs:
            report.total_records_checked += 1
            log_id = log.get("log_id") or "UNKNOWN"
            bug_id = log.get("defect_id")
            ex_id = log.get("execution_id")
            tc_id = log.get("test_case_id")
            inconsistencies = []

            if bug_id and bug_id not in self.defect_ids:
                report.missing_defect_refs += 1
                inconsistencies.append(f"Defect '{bug_id}' not found in defects")
            if ex_id and ex_id not in self.execution_ids:
                report.missing_execution_refs += 1
                inconsistencies.append(f"Execution '{ex_id}' not found in executions")
            if tc_id and tc_id not in self.test_case_ids:
                report.missing_test_case_refs += 1
                inconsistencies.append(f"Test case '{tc_id}' not found in test cases")

            # Cross-check execution and test case consistency
            if ex_id and ex_id in self.execution_to_tc and tc_id:
                exec_tc = self.execution_to_tc[ex_id]
                if exec_tc != tc_id:
                    report.cross_link_mismatches += 1
                    inconsistencies.append(
                        f"Log execution {ex_id} maps to {exec_tc}, but log specifies test case {tc_id}"
                    )

            if inconsistencies:
                report.inconsistent_records_count += 1
                report.issues.append(
                    ValidationIssue(
                        source_file=log.get("log_file", "unknown_log"),
                        record_id=str(log_id),
                        field_name="references",
                        issue_type="conflict",
                        description="; ".join(inconsistencies),
                        observed_value={"bug": bug_id, "exec": ex_id, "tc": tc_id},
                    )
                )
                log["is_cross_consistent"] = False
                log["inconsistency_reasons"] = inconsistencies
            else:
                report.consistent_records_count += 1
                log["is_cross_consistent"] = True
                log["inconsistency_reasons"] = []

        # 4. Validate Screenshots
        for shot in screenshots:
            report.total_records_checked += 1
            shot_id = shot.get("screenshot_id") or "UNKNOWN"
            bug_id = shot.get("defect_id")
            ex_id = shot.get("execution_id")
            tc_id = shot.get("test_case_id")
            inconsistencies = []

            if bug_id and bug_id not in self.defect_ids:
                report.missing_defect_refs += 1
                inconsistencies.append(f"Defect '{bug_id}' not found in defects")
            if ex_id and ex_id not in self.execution_ids:
                report.missing_execution_refs += 1
                inconsistencies.append(f"Execution '{ex_id}' not found in executions")
            if tc_id and tc_id not in self.test_case_ids:
                report.missing_test_case_refs += 1
                inconsistencies.append(f"Test case '{tc_id}' not found in test cases")

            if ex_id and ex_id in self.execution_to_tc and tc_id:
                exec_tc = self.execution_to_tc[ex_id]
                if exec_tc != tc_id:
                    report.cross_link_mismatches += 1
                    inconsistencies.append(
                        f"Screenshot execution {ex_id} maps to {exec_tc}, but screenshot specifies {tc_id}"
                    )

            if inconsistencies:
                report.inconsistent_records_count += 1
                report.issues.append(
                    ValidationIssue(
                        source_file=shot.get("screenshot_file", "unknown_shot"),
                        record_id=str(shot_id),
                        field_name="references",
                        issue_type="conflict",
                        description="; ".join(inconsistencies),
                        observed_value={"bug": bug_id, "exec": ex_id, "tc": tc_id},
                    )
                )
                shot["is_cross_consistent"] = False
                shot["inconsistency_reasons"] = inconsistencies
            else:
                report.consistent_records_count += 1
                shot["is_cross_consistent"] = True
                shot["inconsistency_reasons"] = []

        logger.info(
            f"Validation complete: checked {report.total_records_checked} records. "
            f"Consistent: {report.consistent_records_count}, "
            f"Inconsistent: {report.inconsistent_records_count}"
        )
        return report
