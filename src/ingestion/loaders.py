"""Unified data loaders for CSV, PDF, Failure Logs, and Screenshot OCR JSON."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import pandas as pd

from src.ingestion.extractors import FailureLogExtractor, ScreenshotExtractor, SRSExtractor
from src.ingestion.logging_config import logger


class DatasetLoader:
    """Provides high-level loading functions for all source modalities."""

    @staticmethod
    def load_csv(
        file_path: Path, sample_limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load CSV rows as dictionaries with clean keys."""
        file_path = Path(file_path)
        logger.info(f"Loading CSV from {file_path} (limit={sample_limit})")

        # Read using pandas to handle quoted multi-line fields cleanly
        df = pd.read_csv(file_path, nrows=sample_limit)
        # Convert NaN values to None
        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient="records")
        logger.info(f"Loaded {len(records)} rows from {file_path.name}")
        return records

    @staticmethod
    def load_srs_pdf(file_path: Path) -> List[Dict[str, Any]]:
        """Load and extract logical sections from SRS PDF."""
        file_path = Path(file_path)
        logger.info(f"Loading SRS PDF from {file_path}")
        return SRSExtractor.extract_from_file(file_path)

    @staticmethod
    def load_failure_logs(
        log_files: List[Path], sample_limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load all failure log events across the provided list of files."""
        all_events: List[Dict[str, Any]] = []
        for file_path in log_files:
            file_path = Path(file_path)
            events = FailureLogExtractor.extract_from_file(file_path)
            all_events.extend(events)
            if sample_limit and len(all_events) >= sample_limit:
                all_events = all_events[:sample_limit]
                break

        logger.info(
            f"Loaded {len(all_events)} failure log events across {len(log_files)} files"
        )
        return all_events

    @staticmethod
    def load_screenshot_jsons(
        json_files: List[Path], sample_limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load all screenshot/OCR records across the provided list of JSON files."""
        all_records: List[Dict[str, Any]] = []
        for file_path in json_files:
            file_path = Path(file_path)
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    raw_data = json.load(f)

                if isinstance(raw_data, list):
                    raw_records = raw_data
                elif isinstance(raw_data, dict):
                    raw_records = [raw_data]
                else:
                    logger.warning(f"Unexpected JSON structure in {file_path.name}")
                    continue

                extracted = ScreenshotExtractor.extract_from_records(
                    raw_records, file_path.name
                )
                all_records.extend(extracted)

                if sample_limit and len(all_records) >= sample_limit:
                    all_records = all_records[:sample_limit]
                    break
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        logger.info(
            f"Loaded {len(all_records)} screenshot records across {len(json_files)} files"
        )
        return all_records
