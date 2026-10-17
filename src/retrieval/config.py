"""Configuration settings for the Advanced Retrieval Pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env",
    override=True,
)


@dataclass
class RetrievalConfig:
    """Configurable runtime parameters for routing, retrieval, reranking, and generation."""

    # Groq LLM Configuration
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    groq_temperature: float = field(
        default_factory=lambda: float(os.getenv("GROQ_TEMPERATURE", "0.0"))
    )

    # Qdrant Database
    qdrant_url: Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_URL") or None
    )
    qdrant_path: str = field(
        default_factory=lambda: os.getenv("QDRANT_PATH", "data/data1/qdrant_db")
    )
    qdrant_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY") or None
    )
    dense_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "qa_knowledge")
    )
    sparse_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_SPARSE_COLLECTION", "qa_knowledge_sparse")
    )

    # Vector & Model Names
    dense_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    )
    sparse_model_name: str = field(
        default_factory=lambda: os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm25")
    )
    sparse_vector_name: str = field(
        default_factory=lambda: os.getenv("SPARSE_VECTOR_NAME", "bm25")
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    )

    # Retrieval & Reranker Tunables
    dense_top_k: int = field(
        default_factory=lambda: int(os.getenv("DENSE_TOP_K", "30"))
    )
    sparse_top_k: int = field(
        default_factory=lambda: int(os.getenv("SPARSE_TOP_K", "30"))
    )
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    reranker_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANKER_TOP_K", "50"))
    )
    final_evidence_top_k: int = field(
        default_factory=lambda: int(os.getenv("FINAL_EVIDENCE_TOP_K", "5"))
    )
    reranker_batch_size: int = field(
        default_factory=lambda: int(os.getenv("RERANKER_BATCH_SIZE", "16"))
    )

    # Allowed Schema Filter Keys (strictly bounded to prevent hallucinated filters)
    allowed_filter_fields: Set[str] = field(
        default_factory=lambda: {
            "test_case_id",
            "defect_id",
            "execution_id",
            "requirement_id",
            "document_type",
            "module",
            "environment",
            "priority",
            "severity",
            "status",
            "build_number",
            "source_file",
            "source_type",
        }
    )


default_retrieval_config = RetrievalConfig()
