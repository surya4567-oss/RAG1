"""Dataset discovery and source inventory scanner."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
from pypdf import PdfReader

from src.ingestion.config import PipelineConfig, default_config
from src.ingestion.logging_config import logger


@dataclass
class DiscoveredSource:
    category: str  # testcases, test_execution, defects, srs, failure_logs, screenshots
    path: Path
    relative_path: str
    file_type: str  # csv, pdf, text_log, json
    size_bytes: int
    record_count: int
    is_valid: bool = True
    error_message: Optional[str] = None
    sample_info: Dict = field(default_factory=dict)


@dataclass
class InventoryReport:
    discovered_sources: List[DiscoveredSource] = field(default_factory=list)
    missing_sources: List[str] = field(default_factory=list)
    empty_sources: List[str] = field(default_factory=list)
    unsupported_sources: List[str] = field(default_factory=list)
    total_files_discovered: int = 0
    total_records_discovered: int = 0

    def summary(self) -> Dict:
        return {
            "total_files": self.total_files_discovered,
            "total_records": self.total_records_discovered,
            "categories": {
                s.category: [ds.relative_path for ds in self.discovered_sources if ds.category == s.category]
                for s in self.discovered_sources
            },
            "missing": self.missing_sources,
            "empty": self.empty_sources,
            "unsupported": self.unsupported_sources,
        }


class DatasetDiscoverer:
    """Discovers, validates, and reports all dataset files."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config

    def discover(self) -> InventoryReport:
        logger.info(f"Starting dataset discovery in base data directory: {self.config.data_dir}")
        report = InventoryReport()

        data_root = self.config.data_dir
        if not data_root.exists():
            report.missing_sources.append(str(data_root))
            logger.error(f"Data root directory not found: {data_root}")
            return report

        # Excluded directories (processed, quarantine, qdrant_db, reports, caches)
        excluded_dirs: Set[str] = {
            "processed", "quarantine", "qdrant_db", "reports", "__pycache__", ".pytest_cache"
        }

        # 1. Testcases CSV
        self._discover_csv(
            report,
            category="testcases",
            configured_path=self.config.testcases_file,
            filename_candidates=["testcases.csv", "testcase.csv"],
            id_column="test_case_id"
        )

        # 2. Test Execution CSV
        self._discover_csv(
            report,
            category="test_execution",
            configured_path=self.config.test_execution_file,
            filename_candidates=["test_execution.csv", "testexecution.csv"],
            id_column="execution_id"
        )

        # 3. Defects CSV
        self._discover_csv(
            report,
            category="defects",
            configured_path=self.config.defects_file,
            filename_candidates=["defects_rows.csv", "defects.csv", "defect.csv"],
            id_column="defect_id"
        )

        # 4. SRS PDF
        self._discover_srs(
            report,
            configured_path=self.config.srs_file,
            filename_candidates=["SRS.pdf", "srs.pdf"]
        )

        # 5. Failure Logs Directory (Expect 10 log files)
        self._discover_failure_logs(
            report,
            configured_path=self.config.failure_logs_dir,
            dir_candidates=["failure_logs", "failurelogs", "logs"]
        )

        # 6. Screenshots Directory (Expect 10 JSON files)
        self._discover_screenshots(
            report,
            configured_path=self.config.screenshots_dir,
            dir_candidates=["screenshots", "ocr", "screen_shots"]
        )

        report.total_files_discovered = len(report.discovered_sources)
        report.total_records_discovered = sum(s.record_count for s in report.discovered_sources if s.is_valid)

        logger.info(
            f"Discovery complete: found {report.total_files_discovered} files with "
            f"{report.total_records_discovered} logical records. "
            f"Missing categories: {len(report.missing_sources)}"
        )
        return report

    def _discover_csv(
        self,
        report: InventoryReport,
        category: str,
        configured_path: Optional[Path],
        filename_candidates: List[str],
        id_column: str,
    ):
        target_path: Optional[Path] = configured_path
        if not target_path or not target_path.exists():
            for cand in filename_candidates:
                p = self.config.data_dir / cand
                if p.exists():
                    target_path = p
                    break
                # Try data/raw/cand
                p_raw = self.config.data_dir / "raw" / cand
                if p_raw.exists():
                    target_path = p_raw
                    break

        if not target_path or not target_path.exists():
            report.missing_sources.append(f"{category} (tried {filename_candidates})")
            logger.warning(f"Could not find source CSV for category: {category}")
            return

        size = target_path.stat().st_size
        if size == 0:
            report.empty_sources.append(str(target_path))
            return

        try:
            # Count rows and inspect headers
            df = pd.read_csv(target_path, nrows=5)
            # Count total rows safely
            with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                total_lines = sum(1 for _ in f) - 1  # exclude header
            
            headers = list(df.columns)
            report.discovered_sources.append(
                DiscoveredSource(
                    category=category,
                    path=target_path,
                    relative_path=str(target_path.as_posix()),
                    file_type="csv",
                    size_bytes=size,
                    record_count=max(0, total_lines),
                    sample_info={"headers": headers, "sample_rows": len(df)},
                )
            )
            logger.info(f"Discovered {category} at {target_path} ({total_lines} rows)")
        except Exception as e:
            report.discovered_sources.append(
                DiscoveredSource(
                    category=category,
                    path=target_path,
                    relative_path=str(target_path.as_posix()),
                    file_type="csv",
                    size_bytes=size,
                    record_count=0,
                    is_valid=False,
                    error_message=str(e),
                )
            )
            logger.error(f"Failed to inspect CSV {target_path}: {e}")

    def _discover_srs(
        self,
        report: InventoryReport,
        configured_path: Optional[Path],
        filename_candidates: List[str],
    ):
        target_path: Optional[Path] = configured_path
        if not target_path or not target_path.exists():
            for cand in filename_candidates:
                p = self.config.data_dir / cand
                if p.exists():
                    target_path = p
                    break
                p_raw = self.config.data_dir / "raw" / cand
                if p_raw.exists():
                    target_path = p_raw
                    break

        if not target_path or not target_path.exists():
            report.missing_sources.append(f"srs (tried {filename_candidates})")
            logger.warning("Could not find SRS PDF")
            return

        size = target_path.stat().st_size
        if size == 0:
            report.empty_sources.append(str(target_path))
            return

        try:
            reader = PdfReader(str(target_path))
            pages = len(reader.pages)
            report.discovered_sources.append(
                DiscoveredSource(
                    category="srs",
                    path=target_path,
                    relative_path=str(target_path.as_posix()),
                    file_type="pdf",
                    size_bytes=size,
                    record_count=pages,
                    sample_info={"pages": pages},
                )
            )
            logger.info(f"Discovered SRS PDF at {target_path} ({pages} pages)")
        except Exception as e:
            report.discovered_sources.append(
                DiscoveredSource(
                    category="srs",
                    path=target_path,
                    relative_path=str(target_path.as_posix()),
                    file_type="pdf",
                    size_bytes=size,
                    record_count=0,
                    is_valid=False,
                    error_message=str(e),
                )
            )
            logger.error(f"Failed to inspect PDF {target_path}: {e}")

    def _discover_failure_logs(
        self,
        report: InventoryReport,
        configured_path: Optional[Path],
        dir_candidates: List[str],
    ):
        target_dir: Optional[Path] = configured_path
        if not target_dir or not target_dir.exists():
            for cand in dir_candidates:
                d = self.config.data_dir / cand
                if d.is_dir():
                    target_dir = d
                    break
                d_raw = self.config.data_dir / "raw" / cand
                if d_raw.is_dir():
                    target_dir = d_raw
                    break

        if not target_dir or not target_dir.is_dir():
            report.missing_sources.append(f"failure_logs (tried {dir_candidates})")
            logger.warning("Could not find failure logs directory")
            return

        # Find all log files (.txt, .log)
        log_files = sorted(list(target_dir.glob("*.txt")) + list(target_dir.glob("*.log")))
        if not log_files:
            report.empty_sources.append(str(target_dir))
            logger.warning(f"No log files found in {target_dir}")
            return

        logger.info(f"Found {len(log_files)} failure log files in {target_dir}")
        for f in log_files:
            size = f.stat().st_size
            if size == 0:
                report.empty_sources.append(str(f))
                continue

            # Estimate blocks by delimiter
            block_count = 0
            try:
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
                    # Count blocks delimited by ======================================================================
                    blocks = [b.strip() for b in content.split("======================================================================") if b.strip()]
                    block_count = len(blocks)

                report.discovered_sources.append(
                    DiscoveredSource(
                        category="failure_logs",
                        path=f,
                        relative_path=str(f.as_posix()),
                        file_type="text_log",
                        size_bytes=size,
                        record_count=block_count,
                        sample_info={"events": block_count},
                    )
                )
            except Exception as e:
                report.discovered_sources.append(
                    DiscoveredSource(
                        category="failure_logs",
                        path=f,
                        relative_path=str(f.as_posix()),
                        file_type="text_log",
                        size_bytes=size,
                        record_count=0,
                        is_valid=False,
                        error_message=str(e),
                    )
                )

    def _discover_screenshots(
        self,
        report: InventoryReport,
        configured_path: Optional[Path],
        dir_candidates: List[str],
    ):
        target_dir: Optional[Path] = configured_path
        if not target_dir or not target_dir.exists():
            for cand in dir_candidates:
                d = self.config.data_dir / cand
                if d.is_dir():
                    target_dir = d
                    break
                d_raw = self.config.data_dir / "raw" / cand
                if d_raw.is_dir():
                    target_dir = d_raw
                    break

        if not target_dir or not target_dir.is_dir():
            report.missing_sources.append(f"screenshots (tried {dir_candidates})")
            logger.warning("Could not find screenshots directory")
            return

        json_files = sorted(list(target_dir.glob("*.json")))
        if not json_files:
            report.empty_sources.append(str(target_dir))
            logger.warning(f"No json files found in {target_dir}")
            return

        logger.info(f"Found {len(json_files)} screenshot JSON files in {target_dir}")
        for f in json_files:
            size = f.stat().st_size
            if size == 0:
                report.empty_sources.append(str(f))
                continue

            rec_count = 0
            try:
                with open(f, "r", encoding="utf-8", errors="replace") as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        rec_count = len(data)
                        struct = "array"
                    elif isinstance(data, dict):
                        rec_count = 1
                        struct = "object"
                    else:
                        struct = "unknown"

                report.discovered_sources.append(
                    DiscoveredSource(
                        category="screenshots",
                        path=f,
                        relative_path=str(f.as_posix()),
                        file_type="json",
                        size_bytes=size,
                        record_count=rec_count,
                        sample_info={"structure": struct, "count": rec_count},
                    )
                )
            except Exception as e:
                report.discovered_sources.append(
                    DiscoveredSource(
                        category="screenshots",
                        path=f,
                        relative_path=str(f.as_posix()),
                        file_type="json",
                        size_bytes=size,
                        record_count=0,
                        is_valid=False,
                        error_message=str(e),
                    )
                )
