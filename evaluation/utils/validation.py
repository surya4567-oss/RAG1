"""Validation and sanitization utilities for evaluation records."""

from typing import Any, Dict, List


def validate_evaluation_payload(data: Dict[str, Any]) -> bool:
    """Check that an evaluation record contains minimal required fields."""
    required = {"id", "query", "expected_answer", "category"}
    return required.issubset(data.keys())


def sanitize_text(text: Any) -> str:
    """Normalize text content for evaluation comparison."""
    if text is None:
        return ""
    return str(text).strip()

