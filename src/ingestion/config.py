
"""Configuration settings for the Vanilla RAG Ingestion Pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env",
    override=True,
)


def get_embedding_device() -> str:
    """Use CUDA when available; otherwise fall back to CPU."""
    requested_device = os.getenv("EMBEDDING_DEVICE")

    if requested_device:
        return requested_device

    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class PipelineConfig:
    # Base paths
    base_dir: Path = field(
        default_factory=lambda: Path(os.getenv("BASE_DIR", "."))
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "data"))
    )
    processed_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PROCESSED_DATA_DIR", "data/processed")
        )
    )
    quarantine_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("QUARANTINE_DATA_DIR", "data/quarantine")
        )
    )
    reports_dir: Path = field(
        default_factory=lambda: Path(os.getenv("REPORTS_DIR", "reports"))
    )

    # Expected source files/directories
    testcases_file: Optional[Path] = None
    test_execution_file: Optional[Path] = None
    defects_file: Optional[Path] = None
    srs_file: Optional[Path] = None
    failure_logs_dir: Optional[Path] = None
    screenshots_dir: Optional[Path] = None

    # Qdrant configuration
    qdrant_url: Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_URL") or None
    )
    qdrant_path: str = field(
        default_factory=lambda: os.getenv("QDRANT_PATH", "data/qdrant_db")
    )
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv(
            "QDRANT_COLLECTION", "qa_knowledge"
        )
    )
    qdrant_sparse_collection: str = field(
        default_factory=lambda: os.getenv(
            "QDRANT_SPARSE_COLLECTION", "qa_knowledge_sparse"
        )
    )
    qdrant_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY") or None
    )

    # Embedding configuration
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"
        )
    )
    embedding_device: str = field(default_factory=get_embedding_device)
    embedding_batch_size: int = field(
        default_factory=lambda: int(
            os.getenv("EMBEDDING_BATCH_SIZE", "16")
        )
    )
    embedding_dimension: int = 768

    # Sparse embedding configuration
    sparse_model_name: str = field(
        default_factory=lambda: os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")
    )
    sparse_vector_name: str = field(
        default_factory=lambda: os.getenv("SPARSE_VECTOR_NAME", "bm25")
    )
    sparse_batch_size: int = field(
        default_factory=lambda: int(os.getenv("SPARSE_BATCH_SIZE", "128"))
    )
    sparse_checkpoint_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("SPARSE_CHECKPOINT_DIR", "data/sparse_checkpoints")
        )
    )

    # Chunking configuration
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150"))
    )

    # Indexing configuration
    index_batch_size: int = field(
        default_factory=lambda: int(os.getenv("INDEX_BATCH_SIZE", "32"))
    )

    # Versions
    dataset_version: str = field(
        default_factory=lambda: os.getenv("DATASET_VERSION", "1.0.0")
    )
    schema_version: str = field(
        default_factory=lambda: os.getenv("SCHEMA_VERSION", "1.0.0")
    )

    # Operational flags
    dry_run: bool = False
    source_filter: Optional[str] = None
    sample_limit: Optional[int] = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.processed_dir = Path(self.processed_dir)
        self.quarantine_dir = Path(self.quarantine_dir)
        self.reports_dir = Path(self.reports_dir)
        self.sparse_checkpoint_dir = Path(self.sparse_checkpoint_dir)

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_checkpoint_dir.mkdir(parents=True, exist_ok=True)


default_config = PipelineConfig()
