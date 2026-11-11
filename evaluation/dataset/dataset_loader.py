"""Evaluation Dataset Loader and Schema Validator."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.ingestion.logging_config import logger


@dataclass(frozen=True)
class EvaluationQueryItem:
    """Represents a validated evaluation record."""

    id: str
    query: str
    expected_answer: str
    category: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "query": self.query,
            "expected_answer": self.expected_answer,
            "category": self.category,
        }

    def for_execution(self) -> Dict[str, str]:
        """Return payload for retrieval/generation without leaking ground-truth expected_answer."""
        return {
            "id": self.id,
            "query": self.query,
            "category": self.category,
        }


class EvaluationDataset:
    """Container and validator for evaluation queries."""

    REQUIRED_FIELDS: Set[str] = {"id", "query", "expected_answer", "category"}

    def __init__(self, items: List[EvaluationQueryItem], source_path: Path) -> None:
        self.items = items
        self.source_path = source_path
        self._id_map: Dict[str, EvaluationQueryItem] = {item.id: item for item in items}

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index: int) -> EvaluationQueryItem:
        return self.items[index]

    def get_by_id(self, query_id: str) -> Optional[EvaluationQueryItem]:
        return self._id_map.get(query_id)

    def filter_by_category(self, category: str) -> List[EvaluationQueryItem]:
        return [item for item in self.items if item.category.lower() == category.lower()]

    def filter_by_ids(self, query_ids: List[str]) -> List[EvaluationQueryItem]:
        target_ids = set(query_ids)
        return [item for item in self.items if item.id in target_ids]

    def get_categories(self) -> List[str]:
        return sorted(list({item.category for item in self.items}))

    def category_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.items:
            counts[item.category] = counts.get(item.category, 0) + 1
        return counts


def load_evaluation_dataset(dataset_path: Path) -> EvaluationDataset:
    """Load, validate, and return the evaluation dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If JSON is invalid, schema fields are missing, or duplicate IDs exist.
    """
    path = Path(dataset_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found at: {path}")

    logger.info("Loading evaluation dataset from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        try:
            raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse evaluation dataset JSON at {path}: {e}") from e

    if not isinstance(raw_data, list):
        raise ValueError(f"Dataset root must be a JSON array, got {type(raw_data).__name__}")

    seen_ids: Set[str] = set()
    validated_items: List[EvaluationQueryItem] = []

    for idx, raw_record in enumerate(raw_data):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Item at index {idx} is not a JSON object: {raw_record}")

        # Check required fields
        missing = EvaluationDataset.REQUIRED_FIELDS - set(raw_record.keys())
        if missing:
            raise ValueError(f"Item at index {idx} (ID: {raw_record.get('id', 'unknown')}) missing required fields: {missing}")

        q_id = str(raw_record["id"]).strip()
        query = str(raw_record["query"]).strip()
        expected = str(raw_record["expected_answer"]).strip()
        category = str(raw_record["category"]).strip()

        if not q_id:
            raise ValueError(f"Item at index {idx} has an empty 'id' field")
        if not query:
            raise ValueError(f"Item {q_id} has an empty 'query' field")
        if not expected:
            raise ValueError(f"Item {q_id} has an empty 'expected_answer' field")
        if not category:
            raise ValueError(f"Item {q_id} has an empty 'category' field")

        if q_id in seen_ids:
            raise ValueError(f"Duplicate query ID detected in evaluation dataset: '{q_id}' at index {idx}")

        seen_ids.add(q_id)
        validated_items.append(
            EvaluationQueryItem(
                id=q_id,
                query=query,
                expected_answer=expected,
                category=category,
            )
        )

    logger.info("Successfully loaded and validated %d queries across %d categories", len(validated_items), len(seen_ids))
    return EvaluationDataset(items=validated_items, source_path=path)

