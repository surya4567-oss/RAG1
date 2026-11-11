"""Pydantic data models and validation schemas for the Advanced Retrieval Pipeline."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalRoute(str, Enum):
    """Supported retrieval strategies chosen by the Query Router."""
    KEYWORD = "keyword"
    DENSE = "dense"
    HYBRID = "hybrid"


class RetrievalMode(str, Enum):
    """Retrieval mode determining whether query targets direct entity or cross-document expansion."""
    DIRECT_ENTITY = "direct_entity"
    ENTITY_EXPANSION = "entity_expansion"
    CONCEPTUAL = "conceptual"


class RetrievalStatus(str, Enum):
    """Evidence grounding status of the generated answer."""
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    GENERATION_FAILED = "generation_failed"
    RETRIEVAL_ONLY = "retrieval_only"


class RouterOutput(BaseModel):
    """Structured decision output produced by the LLM Query Router."""
    model_config = ConfigDict(extra="ignore")

    intent: str = Field(description="Concise description of the user intent")
    strategy: RetrievalRoute = Field(description="Selected retrieval route: keyword, dense, or hybrid")
    fallback_strategy: RetrievalRoute = Field(
        default=RetrievalRoute.HYBRID,
        description="Search strategy to directly switch to if metadata filtering returns no results",
    )
    retrieval_mode: RetrievalMode = Field(
        default=RetrievalMode.CONCEPTUAL,
        description="Retrieval mode: direct_entity, entity_expansion, or conceptual",
    )
    primary_entity_type: Optional[str] = Field(
        default=None,
        description="Type of the primary entity (e.g. defect, test_case, test_execution, requirement)",
    )
    primary_entity_id: Optional[str] = Field(
        default=None,
        description="Value of the primary entity ID (e.g. BUG-009235, TC-000001, EXEC-000001)",
    )
    requested_attribute: Optional[str] = Field(
        default=None,
        description="The specific attribute or information requested (e.g. screen_name, status, test_cases, steps)",
    )
    target_document_types: List[str] = Field(
        default_factory=list,
        description="Document types likely to contain the requested evidence (e.g. ['screenshot_evidence'])",
    )
    extracted_identifiers: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Extracted entities such as test_case_id, defect_id, execution_id, error_code, build_number",
    )
    validated_filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Validated metadata filters matching existing collection schema",
    )
    routing_source: str = Field(
        default="llm",
        description="Source of routing decision: 'llm', 'heuristic', or 'cache'",
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
    is_supported: bool = True
    support_confidence: float = 1.0

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
