"""Pydantic data models and validation schemas for the Advanced Retrieval Pipeline."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalRoute(str, Enum):
    """Supported retrieval strategies chosen by the Query Router."""
    KEYWORD = "keyword"
    DENSE = "dense"
    HYBRID = "hybrid"


class RetrievalStatus(str, Enum):
    """Evidence grounding status of the generated answer."""
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


class RouterOutput(BaseModel):
    """Structured decision output produced by the LLM Query Router."""
    model_config = ConfigDict(extra="ignore")

    intent: str = Field(description="Concise description of the user intent")
    strategy: RetrievalRoute = Field(description="Selected retrieval route: keyword, dense, or hybrid")
    extracted_identifiers: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Extracted entities such as test_case_id, defect_id, execution_id, error_code, build_number",
    )
    validated_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validated metadata filters matching existing collection schema",
    )
    reasoning: str = Field(description="Short rationale for strategy selection and filter extraction")


class CandidateChunk(BaseModel):
    """Normalized internal representation for a retrieved document chunk."""
    model_config = ConfigDict(extra="allow")

    chunk_id: str
    text: str
    score: float = 0.0
    original_rank: int = 1
    retrieval_route: str = "dense"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class Evidence(BaseModel):
    """Single evidence citation strictly linked to a retrieved chunk."""
    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    document_type: Optional[str] = None
    source_file: Optional[str] = None
    record_id: Optional[str] = None
    supporting_text: str

    @field_validator("chunk_id", mode="before")
    @classmethod
    def strip_chunk_id(cls, v: Any) -> str:
        return str(v).strip()


class RetrievalResponse(BaseModel):
    """Machine-readable, validated end-to-end response model."""
    model_config = ConfigDict(extra="allow")

    question: str
    answer: str
    status: RetrievalStatus
    retrieval_route: RetrievalRoute
    evidence: List[Evidence] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
