"""Configuration settings for the Advanced Retrieval Pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

from dotenv import load_dotenv

# Load environment variables, overriding existing environment to ensure dynamic updates
load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[2] / ".env",
    override=True,
)

# Route HuggingFace and PyTorch model caches to Drive E to avoid Drive C low disk space
_cache_dir = str(Path(__file__).resolve().parents[2] / ".cache" / "huggingface")
os.environ["HF_HOME"] = _cache_dir
os.environ["HF_HUB_CACHE"] = _cache_dir
os.environ["TRANSFORMERS_CACHE"] = _cache_dir
os.environ["SENTENCE_TRANSFORMERS_HOME"] = _cache_dir


@dataclass
class RetrievalConfig:
    """Configurable runtime parameters for routing, retrieval, reranking, and generation."""

    # Provider Selection: 'groq' | 'colab' | 'kaggle' | 'remote'
    llm_provider: str = field(
        default_factory=lambda: (os.getenv("LLM_PROVIDER") or "groq").lower().strip()
    )
    router_llm_provider: Optional[str] = field(
        default_factory=lambda: (os.getenv("ROUTER_LLM_PROVIDER") or None)
    )
    generator_llm_provider: Optional[str] = field(
        default_factory=lambda: (os.getenv("GENERATOR_LLM_PROVIDER") or None)
    )
    judge_llm_provider: Optional[str] = field(
        default_factory=lambda: (os.getenv("JUDGE_LLM_PROVIDER") or None)
    )

    # Remote / Kaggle / Colab (OpenAI-compatible) Configuration
    colab_llm_base_url: str = field(
        default_factory=lambda: (
            os.getenv("KAGGLE_LLM_BASE_URL")
            or os.getenv("COLAB_LLM_BASE_URL")
            or os.getenv("REMOTE_LLM_BASE_URL")
            or ""
        ).strip()
    )
    colab_llm_api_key: str = field(
        default_factory=lambda: (
            os.getenv("KAGGLE_LLM_API_KEY")
            or os.getenv("COLAB_LLM_API_KEY")
            or os.getenv("REMOTE_LLM_API_KEY")
            or "EMPTY"
        ).strip()
    )
    colab_llm_model: str = field(
        default_factory=lambda: (
            os.getenv("KAGGLE_LLM_MODEL")
            or os.getenv("COLAB_LLM_MODEL")
            or os.getenv("REMOTE_LLM_MODEL")
            or "Qwen/Qwen2.5-7B-Instruct"
        ).strip()
    )

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

    def get_provider(self, role: str = "general") -> str:
        """Resolve the active LLM provider for a given role dynamically."""
        role_provider = getattr(self, f"{role}_llm_provider", None)
        return (role_provider or self.llm_provider or "groq").lower().strip()

    def get_model_name(self, role: str = "general") -> str:
        """Resolve the active model name dynamically based on role provider."""
        provider = self.get_provider(role)
        if provider in ("colab", "kaggle", "remote", "openai"):
            return self.colab_llm_model
        return self.groq_model

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
        default_factory=lambda: int(os.getenv("RERANKER_TOP_K", "25"))
    )
    final_evidence_top_k: int = field(
        default_factory=lambda: int(os.getenv("FINAL_EVIDENCE_TOP_K", "8"))
    )
    expansion_top_k: int = field(
        default_factory=lambda: int(os.getenv("EXPANSION_TOP_K", "15"))
    )
    reranker_batch_size: int = field(
        default_factory=lambda: int(os.getenv("RERANKER_BATCH_SIZE", "16"))
    )

    # Allowed Schema Filter Keys (strictly bounded to prevent hallucinated filters)
    allowed_filter_fields: Set[str] = field(
        default_factory=lambda: {
            # Primary Entity & Hierarchy Identifiers
            "test_case_id",
            "defect_id",
            "execution_id",
            "requirement_id",
            "record_id",
            "epic",
            "story",
            # Structural & Modality Categorical
            "document_type",
            "source_type",
            "source_file",
            "source",
            "module",
            "submodule",
            "environment",
            # Status, Triage & Classification
            "status",
            "priority",
            "severity",
            "bug_type",
            "regression",
            "automation_status",
            "automation_framework",
            "assigned_to",
            "developer",
            "reported_by",
            "reproducibility",
            # Execution Environment & Platforms
            "build_number",
            "found_in_build",
            "fixed_build",
            "fix_version",
            "browser",
            "operating_system",
            "tester",
            # Failure Logs & Technical Errors
            "error_code",
            "exception_type",
            "http_status",
            "log_id",
            "log_file",
            "log_level",
            # Screenshot & UI Evidence
            "screenshot_id",
            "screenshot_file",
            "screen_name",
            "has_error_message",
            "has_warning_message",
            # SRS & Requirement Specifics
            "section_title",
            "section_number",
            "approval_status",
            "page_number",
        }
    )


default_retrieval_config = RetrievalConfig()
