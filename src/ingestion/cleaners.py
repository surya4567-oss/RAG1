"""Text and metadata cleaning, normalization, and quarantine utilities."""

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.ingestion.logging_config import logger


class DataCleaner:
    """Provides conservative cleaning and normalization functions."""

    @staticmethod
    def normalize_text(text: Optional[str]) -> str:
        """Normalize line endings and excess whitespace while preserving structure and code indentations."""
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        # Normalize carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Strip unprintable control characters, but preserve tabs and newlines
        text = "".join(ch for ch in text if ch in ("\n", "\t") or ch >= " ")

        # Trim trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]
        # Remove excessive blank lines (more than 2 consecutive)
        cleaned_lines: List[str] = []
        consecutive_blank = 0
        for line in lines:
            if not line.strip():
                consecutive_blank += 1
                if consecutive_blank <= 2:
                    cleaned_lines.append("")
            else:
                consecutive_blank = 0
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def clean_id(value: Optional[Any], prefix: str = "") -> Optional[str]:
        """Normalize and validate an identifier string."""
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.lower() in ("none", "null", "nan", "undefined", "n/a"):
            return None
        if prefix and not s.startswith(prefix):
            # Do not force prefix, but preserve exact value
            pass
        return s

    @staticmethod
    def parse_iso_date(value: Optional[Any]) -> Optional[str]:
        """Attempt to parse and normalize date string to ISO-8601 YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS."""
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.lower() in ("none", "null", "nan", "undefined", "n/a"):
            return None
        # Common formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y",
            "%m/%d/%Y",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(s, fmt)
                return dt.isoformat()
            except ValueError:
                pass
        return s  # return original string if unrecognized

    @staticmethod
    def clean_float(value: Optional[Any]) -> Optional[float]:
        """Safely parse float, discarding NaN/Inf."""
        if value is None:
            return None
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        except (ValueError, TypeError):
            return None

    @staticmethod
    def clean_int(value: Optional[Any]) -> Optional[int]:
        """Safely parse int."""
        if value is None:
            return None
        try:
            # Handle float strings like '42.0'
            f = float(value)
            if math.isnan(f) or math.isinf(f):
                return None
            return int(f)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def clean_bool(value: Optional[Any]) -> Optional[bool]:
        """Safely parse boolean."""
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f"):
            return False
        return None


class QuarantineManager:
    """Manages quarantined records that fail fundamental integrity or parsing."""

    def __init__(self, quarantine_dir: Path):
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.quarantined_records: List[Dict[str, Any]] = []

    def quarantine(self, record_id: str, source_file: str, reason: str, raw_data: Any):
        """Record an unprocessable or severely malformed record to quarantine."""
        entry = {
            "record_id": record_id,
            "source_file": source_file,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "raw_data": raw_data if isinstance(raw_data, (dict, list, str, int, float)) else str(raw_data),
        }
        self.quarantined_records.append(entry)
        logger.warning(f"Quarantined record '{record_id}' from '{source_file}': {reason}")

    def save_quarantine_file(self) -> Path:
        """Save quarantined records to a timestamped file in quarantine_dir."""
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = self.quarantine_dir / f"quarantined_records_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.quarantined_records, f, indent=2)
        logger.info(f"Saved {len(self.quarantined_records)} quarantined records to {out_path}")
        return out_path
