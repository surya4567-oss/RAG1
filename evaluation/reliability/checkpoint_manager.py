"""Atomic checkpoint manager for evaluation resumption and error recovery."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict, List, Optional, Set

from src.ingestion.logging_config import logger


class QueryExecutionState(str, Enum):
    PENDING = "PENDING"
    ROUTING_COMPLETED = "ROUTING_COMPLETED"
    RETRIEVAL_COMPLETED = "RETRIEVAL_COMPLETED"
    GENERATION_COMPLETED = "GENERATION_COMPLETED"
    EVALUATION_COMPLETED = "EVALUATION_COMPLETED"
    FAILED = "FAILED"
    RATE_LIMITED = "RATE_LIMITED"


@dataclass
class QueryCheckpointRecord:
    id: str
    category: str
    query: str
    expected_answer: str
    execution_status: QueryExecutionState = QueryExecutionState.PENDING
    routing: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    reranking: Dict[str, Any] = field(default_factory=dict)
    generation: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["execution_status"] = self.execution_status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryCheckpointRecord":
        status_val = data.get("execution_status", QueryExecutionState.PENDING.value)
        try:
            status = QueryExecutionState(status_val)
        except ValueError:
            status = QueryExecutionState.FAILED

        return cls(
            id=data["id"],
            category=data["category"],
            query=data["query"],
            expected_answer=data.get("expected_answer", ""),
            execution_status=status,
            routing=data.get("routing", {}),
            retrieval=data.get("retrieval", {}),
            reranking=data.get("reranking", {}),
            generation=data.get("generation", {}),
            evaluation=data.get("evaluation", {}),
            performance=data.get("performance", {}),
            error=data.get("error"),
        )


class CheckpointManager:
    """Thread-safe, atomic checkpoint management for the evaluation pipeline."""

    def __init__(self, checkpoint_file: Path) -> None:
        self.checkpoint_file = Path(checkpoint_file).resolve()
        self._lock = threading.Lock()
        self._records: Dict[str, QueryCheckpointRecord] = {}
        self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        """Load previously saved checkpoint records from disk if present."""
        if not self.checkpoint_file.exists():
            return

        logger.info("Loading checkpoints from %s", self.checkpoint_file)
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record_dict = json.loads(line)
                        rec = QueryCheckpointRecord.from_dict(record_dict)
                        self._records[rec.id] = rec
                    except Exception as e:
                        logger.warning("Corrupted checkpoint line %d: %s", line_idx, e)
            logger.info("Loaded %d existing checkpoint records.", len(self._records))
        except Exception as e:
            logger.error("Failed to read checkpoint file %s: %s", self.checkpoint_file, e)

    def save_record(self, record: QueryCheckpointRecord) -> None:
        """Atomically update memory and append/persist to checkpoint file."""
        with self._lock:
            self._records[record.id] = record
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

            # Rewrite atomically to ensure complete consistency without duplicate keys
            temp_file = self.checkpoint_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                for rec in self._records.values():
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            temp_file.replace(self.checkpoint_file)

    def get_record(self, query_id: str) -> Optional[QueryCheckpointRecord]:
        with self._lock:
            return self._records.get(query_id)

    def get_all_records(self) -> List[QueryCheckpointRecord]:
        with self._lock:
            return list(self._records.values())

    def get_completed_ids(self) -> Set[str]:
        """Return IDs of queries that completed full evaluation."""
        with self._lock:
            return {
                r.id
                for r in self._records.values()
                if r.execution_status == QueryExecutionState.EVALUATION_COMPLETED
            }

    def get_failed_ids(self) -> Set[str]:
        """Return IDs of queries that failed or encountered rate limits."""
        with self._lock:
            return {
                r.id
                for r in self._records.values()
                if r.execution_status in (QueryExecutionState.FAILED, QueryExecutionState.RATE_LIMITED)
            }

    def get_status_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        with self._lock:
            for r in self._records.values():
                status = r.execution_status.value
                counts[status] = counts.get(status, 0) + 1
        return counts

