"""Specialized extractors for CSV, PDF, Failure Logs, and Screenshot OCR JSON."""

import re
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from pypdf import PdfReader

from src.ingestion.cleaners import DataCleaner
from src.ingestion.logging_config import logger


class FailureLogExtractor:
    """Extracts structured failure events from text logs delimited by '======='."""

    DELIMITER = "======================================================================"
    HEADER_REGEX = re.compile(
        r"Bug\s*ID:\s*(?P<bug_id>[^\s|]+)?\s*\|\s*Execution\s*ID:\s*(?P<exec_id>[^\s|]+)?\s*\|\s*Test\s*Case\s*ID:\s*(?P<tc_id>[^\s|]+)?",
        re.IGNORECASE,
    )

    @classmethod
    def extract_from_file(cls, file_path: Path) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        filename = file_path.name
        events: List[Dict[str, Any]] = []

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total_lines = len(lines)
        current_block_lines: List[str] = []
        block_start_line = 1
        in_block = False
        event_idx = 0

        for line_num, line in enumerate(lines, start=1):
            if cls.DELIMITER in line:
                if in_block:
                    # Closing current block
                    raw_text = "".join(current_block_lines)
                    event_data = cls._parse_block(
                        raw_text=raw_text,
                        filename=filename,
                        event_idx=event_idx,
                        line_start=block_start_line,
                        line_end=line_num - 1,
                    )
                    if event_data:
                        events.append(event_data)
                        event_idx += 1
                    current_block_lines = []
                    in_block = False
                else:
                    # Opening a new block
                    in_block = True
                    block_start_line = line_num + 1
                    current_block_lines = []
            elif in_block:
                current_block_lines.append(line)

        # Handle trailing block if not closed with delimiter
        if in_block and current_block_lines:
            raw_text = "".join(current_block_lines)
            event_data = cls._parse_block(
                raw_text=raw_text,
                filename=filename,
                event_idx=event_idx,
                line_start=block_start_line,
                line_end=total_lines,
            )
            if event_data:
                events.append(event_data)

        logger.debug(f"Extracted {len(events)} events from {filename}")
        return events

    @classmethod
    def _parse_block(
        cls, raw_text: str, filename: str, event_idx: int, line_start: int, line_end: int
    ) -> Optional[Dict[str, Any]]:
        text = raw_text.strip()
        if not text:
            return None

        bug_id = None
        exec_id = None
        tc_id = None
        timestamp = None
        module = None
        user_action = None
        traceback_text = None
        database_error = None
        http_status = None
        server_log = None
        exception_type = None
        resolution_hint = None

        # Parse key fields using line-by-line / section matching
        sections = re.split(r"-{40,}", text)
        top_part = sections[0] if sections else text
        bottom_part = sections[1] if len(sections) > 1 else ""

        # Extract IDs from top part
        match = cls.HEADER_REGEX.search(top_part)
        if match:
            bug_id = DataCleaner.clean_id(match.group("bug_id"))
            exec_id = DataCleaner.clean_id(match.group("exec_id"))
            tc_id = DataCleaner.clean_id(match.group("tc_id"))

        # Extract metadata lines in top part
        for line in top_part.splitlines():
            line_str = line.strip()
            if line_str.startswith("Timestamp:"):
                timestamp = line_str.split("Timestamp:", 1)[1].strip()
            elif line_str.startswith("Module:"):
                module = line_str.split("Module:", 1)[1].strip()
            elif line_str.startswith("User Action:"):
                user_action = line_str.split("User Action:", 1)[1].strip()

        # Extract sections in bottom part
        # Markers: Python Traceback:, Database Error:, HTTP Status:, Server Log:, Exception:, Resolution Hint:
        markers = [
            ("Python Traceback:", "traceback"),
            ("Database Error:", "db_error"),
            ("HTTP Status:", "http_status"),
            ("Server Log:", "server_log"),
            ("Exception:", "exception"),
            ("Resolution Hint:", "resolution_hint"),
        ]

        # Regex find marker positions
        marker_positions = []
        target_text = bottom_part if bottom_part else top_part
        for marker_str, key in markers:
            pos = target_text.find(marker_str)
            if pos != -1:
                marker_positions.append((pos, len(marker_str), key))

        marker_positions.sort(key=lambda x: x[0])

        parsed_sections: Dict[str, str] = {}
        for i, (pos, mlen, key) in enumerate(marker_positions):
            start = pos + mlen
            end = marker_positions[i + 1][0] if i + 1 < len(marker_positions) else len(target_text)
            parsed_sections[key] = target_text[start:end].strip()

        traceback_text = parsed_sections.get("traceback")
        database_error = parsed_sections.get("db_error")
        http_status_str = parsed_sections.get("http_status")
        if http_status_str:
            http_status = DataCleaner.clean_int(http_status_str)
        server_log = parsed_sections.get("server_log")
        exception_type = parsed_sections.get("exception")
        resolution_hint = parsed_sections.get("resolution_hint")

        # Stable log ID
        log_id = f"{filename}_evt_{event_idx:04d}"

        return {
            "log_id": log_id,
            "log_file": filename,
            "log_record_index": event_idx,
            "defect_id": bug_id,
            "execution_id": exec_id,
            "test_case_id": tc_id,
            "timestamp": timestamp,
            "module": module,
            "user_action": user_action,
            "traceback": traceback_text,
            "database_error": database_error,
            "http_status": http_status,
            "server_log": server_log,
            "exception_type": exception_type,
            "resolution_hint": resolution_hint,
            "line_start": line_start,
            "line_end": line_end,
            "raw_block": text,
        }


class ScreenshotExtractor:
    """Extracts OCR and screenshot record fields from JSON files."""

    @classmethod
    def extract_from_records(
        cls, records: List[Dict[str, Any]], filename: str
    ) -> List[Dict[str, Any]]:
        extracted = []
        for idx, rec in enumerate(records):
            bug_id = DataCleaner.clean_id(rec.get("bug_id"))
            exec_id = DataCleaner.clean_id(rec.get("execution_id"))
            tc_id = DataCleaner.clean_id(rec.get("test_case_id"))
            screen_name = rec.get("screen_name")
            module = rec.get("module")
            captured_time = DataCleaner.parse_iso_date(rec.get("captured_time"))
            user_action = rec.get("user_action")

            buttons = rec.get("buttons") or []
            labels = rec.get("labels") or []
            menus = rec.get("menus") or []
            dialog_boxes = rec.get("dialog_boxes") or []
            tables = rec.get("tables") or []
            warning_messages = rec.get("warning_messages") or []
            error_messages = rec.get("error_messages") or []
            field_values = rec.get("field_values") or {}

            has_error = bool(error_messages and any(m.strip() for m in error_messages))
            has_warning = bool(warning_messages and any(m.strip() for m in warning_messages))

            screenshot_id = f"{filename}_rec_{idx:04d}"

            extracted.append({
                "screenshot_id": screenshot_id,
                "screenshot_file": filename,
                "record_index": idx,
                "defect_id": bug_id,
                "execution_id": exec_id,
                "test_case_id": tc_id,
                "screen_name": screen_name,
                "module": module,
                "captured_time": captured_time,
                "user_action": user_action,
                "buttons": buttons,
                "labels": labels,
                "menus": menus,
                "dialog_boxes": dialog_boxes,
                "tables": tables,
                "warning_messages": warning_messages,
                "error_messages": error_messages,
                "field_values": field_values,
                "has_error_message": has_error,
                "has_warning_message": has_warning,
                "raw_record": rec,
            })
        return extracted


class SRSExtractor:
    """Extracts structured sections, requirements, and tables from SRS PDF."""

    REQ_PATTERN = re.compile(r"((?:FR-[A-Z]+-\d+)|(?:REQ-\d+))")
    SECTION_HEADER_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z][^\n]+)$", re.MULTILINE)

    @classmethod
    def extract_from_file(cls, pdf_path: Path) -> List[Dict[str, Any]]:
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        sections: List[Dict[str, Any]] = []

        reader = PdfReader(str(pdf_path))
        num_pages = len(reader.pages)

        # Baseline approval status from page 1 / document control if detectable
        approval_status = "For stakeholder review and approval"

        for page_idx in range(num_pages):
            page_num = page_idx + 1
            page = reader.pages[page_idx]
            text = page.extract_text() or ""
            text = DataCleaner.normalize_text(text)

            # Look for requirements on this page
            # Many pages have specific FR-COM-xxx or FR-CRM-xxx requirements
            req_matches = cls.REQ_PATTERN.findall(text)
            unique_reqs = list(dict.fromkeys(req_matches))

            # Detect main section titles on page
            section_matches = cls.SECTION_HEADER_PATTERN.findall(text)
            section_title = None
            section_num = None
            if section_matches:
                section_num, section_title = section_matches[0]
                section_title = section_title.strip()

            if unique_reqs:
                # Create section document for each requirement block on this page
                # or group requirements by page
                for req_id in unique_reqs:
                    sections.append({
                        "section_title": section_title or f"Requirement {req_id}",
                        "section_number": section_num,
                        "requirement_id": req_id,
                        "page_number": page_num,
                        "page_start": page_num,
                        "page_end": page_num,
                        "approval_status": approval_status,
                        "source_file": filename,
                        "content": text,
                    })
            else:
                # General section page
                sections.append({
                    "section_title": section_title or f"SRS Page {page_num}",
                    "section_number": section_num,
                    "requirement_id": None,
                    "page_number": page_num,
                    "page_start": page_num,
                    "page_end": page_num,
                    "approval_status": approval_status,
                    "source_file": filename,
                    "content": text,
                })

        logger.info(f"Extracted {len(sections)} logical sections from SRS PDF ({num_pages} pages)")
        return sections
