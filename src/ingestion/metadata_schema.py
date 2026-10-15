"""Metadata schema definition and validation for Vanilla RAG Ingestion."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SourceType(str, Enum):
    CSV = "csv"
    PDF = "pdf"
    TEXT_LOG = "text_log"
    JSON = "json"


class DocumentType(str, Enum):
    TEST_CASE = "test_case"
    TEST_EXECUTION = "test_execution"
    DEFECT = "defect"
    SRS_SECTION = "srs_section"
    FAILURE_LOG = "failure_log"
    SCREENSHOT_EVIDENCE = "screenshot_evidence"


class BaseMetadata(BaseModel):
    """Common metadata schema applied to every document and chunk."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    source: str
    source_path: str
    source_type: SourceType
    document_type: DocumentType
    record_id: str
    document_id: str
    chunk_id: Optional[str] = None
    parent_document_id: Optional[str] = None
    content_hash: str
    dataset_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    ingestion_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    chunk_index: int = 0

    # Cross-referencing and relationship flags
    is_cross_consistent: bool = True
    inconsistency_reasons: List[str] = Field(default_factory=list)


class TestCaseMetadata(BaseMetadata):
    document_type: DocumentType = DocumentType.TEST_CASE
    test_case_id: str
    requirement_id: Optional[str] = None
    module: Optional[str] = None
    submodule: Optional[str] = None
    feature: Optional[str] = None
    priority: Optional[str] = None
    severity: Optional[str] = None
    test_type: Optional[str] = None
    srs_type: Optional[str] = None
    source_file: Optional[str] = None
    srs_page: Optional[int] = None
    status: Optional[str] = None
    automation_status: Optional[str] = None
    created_date: Optional[str] = None
    updated_date: Optional[str] = None


class TestExecutionMetadata(BaseMetadata):
    document_type: DocumentType = DocumentType.TEST_EXECUTION
    execution_id: str
    test_case_id: str
    build_number: Optional[str] = None
    execution_date: Optional[str] = None
    tester: Optional[str] = None
    environment: Optional[str] = None
    browser: Optional[str] = None
    operating_system: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    status: str


class DefectMetadata(BaseMetadata):
    document_type: DocumentType = DocumentType.DEFECT
    defect_id: str
    test_case_id: Optional[str] = None
    execution_id: Optional[str] = None
    severity: Optional[str] = None
    priority: Optional[str] = None
    module: Optional[str] = None
    status: Optional[str] = None
    bug_type: Optional[str] = None
    root_cause: Optional[str] = None
    reproducibility: Optional[str] = None
    environment: Optional[str] = None
    fix_version: Optional[str] = None
    found_in_build: Optional[str] = None
    fixed_build: Optional[str] = None
    regression: Optional[bool] = None


class SRSSectionMetadata(BaseMetadata):
    document_type: DocumentType = DocumentType.SRS_SECTION
    section_title: Optional[str] = None
    section_number: Optional[str] = None
    requirement_id: Optional[str] = None
    page_number: int = 1
    page_start: int = 1
    page_end: int = 1
    approval_status: Optional[str] = None


class FailureLogMetadata(BaseMetadata):
    document_type: DocumentType = DocumentType.FAILURE_LOG
    log_file: str
    log_record_index: int = 0
    log_id: str
    defect_id: Optional[str] = None
    execution_id: Optional[str] = None
    test_case_id: Optional[str] = None
    timestamp: Optional[str] = None
    module: Optional[str] = None
    log_level: Optional[str] = None
    error_code: Optional[str] = None
    exception_type: Optional[str] = None
    http_status: Optional[int] = None
    line_start: int = 1
    line_end: int = 1


class ScreenshotEvidenceMetadata(BaseMetadata):
    document_type: DocumentType = DocumentType.SCREENSHOT_EVIDENCE
    screenshot_id: Optional[str] = None
    screenshot_file: str
    record_index: int = 0
    defect_id: Optional[str] = None
    execution_id: Optional[str] = None
    test_case_id: Optional[str] = None
    screen_name: Optional[str] = None
    module: Optional[str] = None
    captured_time: Optional[str] = None
    has_error_message: bool = False
    has_warning_message: bool = False


def sanitize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Clean a metadata dictionary for Qdrant payload storage.
    
    Ensures:
    1. No literal strings 'None', 'null', 'NaN', 'undefined'.
    2. Floats with NaN or Inf are converted to None or omitted.
    3. Types are JSON compatible.
    4. None values are pruned or cleanly stored.
    """
    clean_dict: Dict[str, Any] = {}
    invalid_literals = {"none", "null", "nan", "undefined", ""}

    for k, v in payload.items():
        if v is None:
            continue
        if isinstance(v, str):
            val_stripped = v.strip()
            if val_stripped.lower() in invalid_literals:
                continue
            clean_dict[k] = val_stripped
        elif isinstance(v, float):
            import math
            if math.isnan(v) or math.isinf(v):
                continue
            clean_dict[k] = v
        elif isinstance(v, (int, bool)):
            clean_dict[k] = v
        elif isinstance(v, list):
            clean_list = []
            for item in v:
                if isinstance(item, str) and item.strip().lower() in invalid_literals:
                    continue
                clean_list.append(item)
            clean_dict[k] = clean_list
        elif isinstance(v, dict):
            clean_dict[k] = sanitize_payload(v)
        else:
            clean_dict[k] = str(v)
            
    return clean_dict
