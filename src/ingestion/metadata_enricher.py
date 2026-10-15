"""Enriches logical documents with standardized common and source-specific metadata."""

import hashlib
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

from src.ingestion.config import PipelineConfig, default_config
from src.ingestion.logging_config import logger
from src.ingestion.metadata_schema import (
    BaseMetadata,
    DefectMetadata,
    DocumentType,
    FailureLogMetadata,
    ScreenshotEvidenceMetadata,
    SourceType,
    SRSSectionMetadata,
    TestCaseMetadata,
    TestExecutionMetadata,
    sanitize_payload,
)


class MetadataEnricher:
    """Applies common and source-specific metadata enrichment according to schema."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Compute stable SHA-256 hash of normalized document text."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def enrich_document(self, doc: Document) -> Document:
        """Enrich a single document based on its document_type."""
        doc_type = doc.metadata.get("document_type")
        raw = doc.metadata.get("raw_record", {})
        source_file = doc.metadata.get("source", "unknown")
        record_id = doc.metadata.get("record_id", "unknown")

        content_hash = self.compute_content_hash(doc.page_content)
        doc_id = f"doc_{doc_type}_{record_id}"

        # Preserve relationship validation flags if present
        is_consistent = raw.get("is_cross_consistent", True)
        inconsistency_reasons = raw.get("inconsistency_reasons", [])

        if doc_type == DocumentType.TEST_CASE.value or doc_type == "test_case":
            meta = self._enrich_test_case(
                doc_id=doc_id,
                record_id=record_id,
                source_file=source_file,
                content_hash=content_hash,
                raw=raw,
                is_consistent=is_consistent,
                reasons=inconsistency_reasons,
            )
        elif doc_type == DocumentType.TEST_EXECUTION.value or doc_type == "test_execution":
            meta = self._enrich_test_execution(
                doc_id=doc_id,
                record_id=record_id,
                source_file=source_file,
                content_hash=content_hash,
                raw=raw,
                is_consistent=is_consistent,
                reasons=inconsistency_reasons,
            )
        elif doc_type == DocumentType.DEFECT.value or doc_type == "defect":
            meta = self._enrich_defect(
                doc_id=doc_id,
                record_id=record_id,
                source_file=source_file,
                content_hash=content_hash,
                raw=raw,
                is_consistent=is_consistent,
                reasons=inconsistency_reasons,
            )
        elif doc_type == DocumentType.SRS_SECTION.value or doc_type == "srs_section":
            meta = self._enrich_srs_section(
                doc_id=doc_id,
                record_id=record_id,
                source_file=source_file,
                content_hash=content_hash,
                raw=raw,
                is_consistent=is_consistent,
                reasons=inconsistency_reasons,
            )
        elif doc_type == DocumentType.FAILURE_LOG.value or doc_type == "failure_log":
            meta = self._enrich_failure_log(
                doc_id=doc_id,
                record_id=record_id,
                source_file=source_file,
                content_hash=content_hash,
                raw=raw,
                is_consistent=is_consistent,
                reasons=inconsistency_reasons,
            )
        elif doc_type == DocumentType.SCREENSHOT_EVIDENCE.value or doc_type == "screenshot_evidence":
            meta = self._enrich_screenshot_evidence(
                doc_id=doc_id,
                record_id=record_id,
                source_file=source_file,
                content_hash=content_hash,
                raw=raw,
                is_consistent=is_consistent,
                reasons=inconsistency_reasons,
            )
        else:
            meta = BaseMetadata(
                source=source_file,
                source_path=source_file,
                source_type=SourceType.CSV,
                document_type=DocumentType.TEST_CASE,
                record_id=record_id,
                document_id=doc_id,
                content_hash=content_hash,
                dataset_version=self.config.dataset_version,
                schema_version=self.config.schema_version,
                is_cross_consistent=is_consistent,
                inconsistency_reasons=inconsistency_reasons,
            )

        # Sanitize dictionary for vector store payload
        payload = sanitize_payload(meta.model_dump())
        doc.metadata = payload
        return doc

    def _enrich_test_case(
        self, doc_id: str, record_id: str, source_file: str, content_hash: str, raw: Dict, is_consistent: bool, reasons: List[str]
    ) -> TestCaseMetadata:
        return TestCaseMetadata(
            source=source_file,
            source_path=source_file,
            source_type=SourceType.CSV,
            document_type=DocumentType.TEST_CASE,
            record_id=record_id,
            document_id=doc_id,
            content_hash=content_hash,
            dataset_version=self.config.dataset_version,
            schema_version=self.config.schema_version,
            is_cross_consistent=is_consistent,
            inconsistency_reasons=reasons,
            test_case_id=record_id,
            requirement_id=raw.get("requirement_id"),
            module=raw.get("module"),
            submodule=raw.get("submodule"),
            feature=raw.get("feature"),
            priority=raw.get("priority"),
            severity=raw.get("severity"),
            test_type=raw.get("test_type"),
            srs_type=raw.get("srs_type"),
            source_file=source_file,
            status=raw.get("status"),
            automation_status=raw.get("automation_status"),
            created_date=raw.get("created_date"),
            updated_date=raw.get("updated_date"),
        )

    def _enrich_test_execution(
        self, doc_id: str, record_id: str, source_file: str, content_hash: str, raw: Dict, is_consistent: bool, reasons: List[str]
    ) -> TestExecutionMetadata:
        exec_time = None
        if raw.get("execution_time_seconds") is not None:
            try:
                exec_time = float(raw["execution_time_seconds"])
            except (ValueError, TypeError):
                pass

        return TestExecutionMetadata(
            source=source_file,
            source_path=source_file,
            source_type=SourceType.CSV,
            document_type=DocumentType.TEST_EXECUTION,
            record_id=record_id,
            document_id=doc_id,
            content_hash=content_hash,
            dataset_version=self.config.dataset_version,
            schema_version=self.config.schema_version,
            is_cross_consistent=is_consistent,
            inconsistency_reasons=reasons,
            execution_id=record_id,
            test_case_id=raw.get("test_case_id", "UNKNOWN"),
            build_number=raw.get("build_number"),
            execution_date=raw.get("execution_date"),
            tester=raw.get("tester"),
            environment=raw.get("environment"),
            browser=raw.get("browser"),
            operating_system=raw.get("operating_system"),
            execution_time_seconds=exec_time,
            status=raw.get("status", "Unknown"),
        )

    def _enrich_defect(
        self, doc_id: str, record_id: str, source_file: str, content_hash: str, raw: Dict, is_consistent: bool, reasons: List[str]
    ) -> DefectMetadata:
        reg_val = raw.get("regression")
        reg_bool = None
        if isinstance(reg_val, bool):
            reg_bool = reg_val
        elif isinstance(reg_val, str):
            reg_bool = reg_val.lower() in ("yes", "true", "1")

        return DefectMetadata(
            source=source_file,
            source_path=source_file,
            source_type=SourceType.CSV,
            document_type=DocumentType.DEFECT,
            record_id=record_id,
            document_id=doc_id,
            content_hash=content_hash,
            dataset_version=self.config.dataset_version,
            schema_version=self.config.schema_version,
            is_cross_consistent=is_consistent,
            inconsistency_reasons=reasons,
            defect_id=record_id,
            test_case_id=raw.get("linked_test_case"),
            execution_id=raw.get("linked_execution"),
            severity=raw.get("severity"),
            priority=raw.get("priority"),
            module=raw.get("module"),
            status=raw.get("status"),
            bug_type=raw.get("bug_type"),
            root_cause=raw.get("root_cause"),
            reproducibility=raw.get("reproducibility"),
            environment=raw.get("environment"),
            fix_version=raw.get("fix_version"),
            found_in_build=raw.get("found_in_build"),
            fixed_build=raw.get("fixed_build"),
            regression=reg_bool,
        )

    def _enrich_srs_section(
        self, doc_id: str, record_id: str, source_file: str, content_hash: str, raw: Dict, is_consistent: bool, reasons: List[str]
    ) -> SRSSectionMetadata:
        page_num = raw.get("page_number", 1)
        return SRSSectionMetadata(
            source=source_file,
            source_path=source_file,
            source_type=SourceType.PDF,
            document_type=DocumentType.SRS_SECTION,
            record_id=record_id,
            document_id=doc_id,
            content_hash=content_hash,
            dataset_version=self.config.dataset_version,
            schema_version=self.config.schema_version,
            is_cross_consistent=is_consistent,
            inconsistency_reasons=reasons,
            section_title=raw.get("section_title"),
            section_number=raw.get("section_number"),
            requirement_id=raw.get("requirement_id"),
            page_number=page_num,
            page_start=raw.get("page_start", page_num),
            page_end=raw.get("page_end", page_num),
            approval_status=raw.get("approval_status"),
        )

    def _enrich_failure_log(
        self, doc_id: str, record_id: str, source_file: str, content_hash: str, raw: Dict, is_consistent: bool, reasons: List[str]
    ) -> FailureLogMetadata:
        return FailureLogMetadata(
            source=source_file,
            source_path=source_file,
            source_type=SourceType.TEXT_LOG,
            document_type=DocumentType.FAILURE_LOG,
            record_id=record_id,
            document_id=doc_id,
            content_hash=content_hash,
            dataset_version=self.config.dataset_version,
            schema_version=self.config.schema_version,
            is_cross_consistent=is_consistent,
            inconsistency_reasons=reasons,
            log_file=source_file,
            log_record_index=raw.get("log_record_index", 0),
            log_id=record_id,
            defect_id=raw.get("defect_id"),
            execution_id=raw.get("execution_id"),
            test_case_id=raw.get("test_case_id"),
            timestamp=raw.get("timestamp"),
            module=raw.get("module"),
            log_level=raw.get("log_level"),
            error_code=raw.get("error_code"),
            exception_type=raw.get("exception_type"),
            http_status=raw.get("http_status"),
            line_start=raw.get("line_start", 1),
            line_end=raw.get("line_end", 1),
        )

    def _enrich_screenshot_evidence(
        self, doc_id: str, record_id: str, source_file: str, content_hash: str, raw: Dict, is_consistent: bool, reasons: List[str]
    ) -> ScreenshotEvidenceMetadata:
        return ScreenshotEvidenceMetadata(
            source=source_file,
            source_path=source_file,
            source_type=SourceType.JSON,
            document_type=DocumentType.SCREENSHOT_EVIDENCE,
            record_id=record_id,
            document_id=doc_id,
            content_hash=content_hash,
            dataset_version=self.config.dataset_version,
            schema_version=self.config.schema_version,
            is_cross_consistent=is_consistent,
            inconsistency_reasons=reasons,
            screenshot_id=record_id,
            screenshot_file=source_file,
            record_index=raw.get("record_index", 0),
            defect_id=raw.get("defect_id"),
            execution_id=raw.get("execution_id"),
            test_case_id=raw.get("test_case_id"),
            screen_name=raw.get("screen_name"),
            module=raw.get("module"),
            captured_time=raw.get("captured_time"),
            has_error_message=raw.get("has_error_message", False),
            has_warning_message=raw.get("has_warning_message", False),
        )

    def enrich_batch(self, docs: List[Document]) -> List[Document]:
        """Enrich a batch of documents."""
        return [self.enrich_document(d) for d in docs]
