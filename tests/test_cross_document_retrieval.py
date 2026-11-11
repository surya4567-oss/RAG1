"""Test suite verifying the 6 core scenarios for cross-document metadata filtering and evidence grounding."""

from unittest.mock import MagicMock
import pytest

from src.retrieval.answer_generator import EvidenceGroundedAnswerGenerator
from src.retrieval.config import RetrievalConfig
from src.retrieval.pipeline import AdvancedRetrievalPipeline
from src.retrieval.router import LLMQueryRouter
from src.retrieval.schemas import (
    CandidateChunk,
    Evidence,
    RetrievalMode,
    RetrievalResponse,
    RetrievalRoute,
    RetrievalStatus,
    RouterOutput,
)


# -------------------------------------------------------------------------
# Test 1: Direct Defect Lookup
# Query: "What is the status of BUG-009235?"
# Expected: DIRECT_ENTITY mode, exact defect_id filter, answers from retrieved chunk.
# -------------------------------------------------------------------------
def test_case_1_direct_defect_lookup():
    router = LLMQueryRouter(config=RetrievalConfig(groq_api_key=""))
    q = "What is the status of BUG-009235?"
    route_out = router.route(q)

    assert route_out.retrieval_mode == RetrievalMode.DIRECT_ENTITY
    assert route_out.primary_entity_id == "BUG-009235"
    assert route_out.primary_entity_type == "defect"
    assert route_out.validated_filters.get("defect_id") == "BUG-009235"
    assert route_out.strategy == RetrievalRoute.KEYWORD


# -------------------------------------------------------------------------
# Test 2: Cross-Document Evidence Lookup
# Query: "What is screen name of bug id BUG-009235?"
# Expected: ENTITY_EXPANSION mode, discovers EXEC-014086, retrieves screenshot evidence,
# and aggregates cross-document candidates.
# -------------------------------------------------------------------------
def test_case_2_cross_document_evidence_lookup():
    router = LLMQueryRouter(config=RetrievalConfig(groq_api_key=""))
    q = "What is screen name of bug id BUG-009235?"
    route_out = router.route(q)

    assert route_out.retrieval_mode == RetrievalMode.ENTITY_EXPANSION
    assert route_out.primary_entity_id == "BUG-009235"
    assert route_out.requested_attribute == "screen_name"
    assert "screenshot_evidence" in route_out.target_document_types

    # Mock pipeline execution showing two-phase expansion
    seed_defect = CandidateChunk(
        chunk_id="doc_defect_BUG-009235_c0",
        text="Defect BUG-009235 linked to execution EXEC-014086 and test case TC-007738.",
        score=10.0,
        original_rank=1,
        metadata={
            "document_type": "defect",
            "defect_id": "BUG-009235",
            "execution_id": "EXEC-014086",
            "test_case_id": "TC-007738",
            "status": "Closed",
        },
    )

    mock_client = MagicMock()
    # Mock scroll returning screenshot evidence linked to EXEC-014086
    mock_point = MagicMock()
    mock_point.id = "pt-screen-1"
    mock_point.payload = {
        "chunk_id": "ocr_batch_004.json_rec_0328",
        "document_type": "screenshot_evidence",
        "screen_name": "POS Dashboard",
        "execution_id": "EXEC-014086",
        "defect_id": "BUG-002759",
        "page_content": "Screenshot OCR: POS Dashboard showing error banner.",
    }
    mock_client.scroll.return_value = ([mock_point], None)

    pipeline = AdvancedRetrievalPipeline(
        router=router,
        qdrant_client=mock_client,
    )

    expanded = pipeline._expand_relationships([seed_defect], route_out)
    assert len(expanded) > 0
    assert any(c.metadata.get("screen_name") == "POS Dashboard" for c in expanded)


# -------------------------------------------------------------------------
# Test 3: Multiple Evidence Values
# When two screenshots associated with the same defect contain different screen names,
# both distinct values must be synthesized and cited without dropping either.
# -------------------------------------------------------------------------
def test_case_3_multiple_evidence_values_synthesis():
    generator = EvidenceGroundedAnswerGenerator(config=RetrievalConfig(groq_api_key=""))

    chunk1 = CandidateChunk(
        chunk_id="ocr_batch_003_c0",
        text="Screenshot showing Subscriptions Dashboard with active warning dialog.",
        score=0.9,
        original_rank=1,
        metadata={
            "document_type": "screenshot_evidence",
            "screen_name": "Subscriptions Dashboard",
            "defect_id": "BUG-009235",
            "execution_id": "EXEC-010835",
            "source_file": "ocr_batch_003.json",
        },
    )
    chunk2 = CandidateChunk(
        chunk_id="ocr_batch_007_c0",
        text="Screenshot showing eCommerce Dashboard with permission error popup.",
        score=0.85,
        original_rank=2,
        metadata={
            "document_type": "screenshot_evidence",
            "screen_name": "eCommerce Dashboard",
            "defect_id": "BUG-009235",
            "execution_id": "EXEC-011159",
            "source_file": "ocr_batch_007.json",
        },
    )

    # Validate evidence formatting includes all distinct metadata
    context_str = generator._format_evidence_context([chunk1, chunk2])
    assert "SCREEN_NAME: Subscriptions Dashboard" in context_str
    assert "SCREEN_NAME: eCommerce Dashboard" in context_str
    assert "DEFECT_ID: BUG-009235" in context_str

    # Validate citation validator accepts both chunks simultaneously
    raw_citations = [
        {"chunk_id": "ocr_batch_003_c0", "supporting_text": "Subscriptions Dashboard"},
        {"chunk_id": "ocr_batch_007_c0", "supporting_text": "eCommerce Dashboard"},
    ]
    chunk_map = {c.chunk_id: c for c in [chunk1, chunk2]}
    validated = generator._validate_evidence_citations(raw_citations, chunk_map)

    assert len(validated) == 2
    cited_ids = {e.chunk_id for e in validated}
    assert "ocr_batch_003_c0" in cited_ids
    assert "ocr_batch_007_c0" in cited_ids


# -------------------------------------------------------------------------
# Test 4: Conceptual Question
# Query: "What are the common causes of test execution failures?"
# Expected: CONCEPTUAL mode, dense/hybrid strategy, no forced entity ID filter.
# -------------------------------------------------------------------------
def test_case_4_conceptual_question_routing():
    router = LLMQueryRouter(config=RetrievalConfig(groq_api_key=""))
    q = "What are the common causes of test execution failures?"
    route_out = router.route(q)

    assert route_out.retrieval_mode == RetrievalMode.CONCEPTUAL
    assert route_out.strategy in (RetrievalRoute.DENSE, RetrievalRoute.HYBRID)
    assert route_out.primary_entity_id is None
    assert "test_case_id" not in route_out.validated_filters
    assert "defect_id" not in route_out.validated_filters


# -------------------------------------------------------------------------
# Test 5: Missing Evidence
# Query: "What is the screen name of BUG-999999?"
# Expected: Non-existent ID returns INSUFFICIENT_EVIDENCE without hallucination.
# -------------------------------------------------------------------------
def test_case_5_missing_evidence_returns_insufficient():
    generator = EvidenceGroundedAnswerGenerator(config=RetrievalConfig(groq_api_key=""))
    response = generator.generate(
        question="What is the screen name of BUG-999999?",
        candidates=[],  # No matching records retrieved
        retrieval_route=RetrievalRoute.KEYWORD,
    )

    assert response.status == RetrievalStatus.INSUFFICIENT_EVIDENCE
    assert len(response.evidence) == 0
    assert "No relevant evidence" in response.answer


# -------------------------------------------------------------------------
# Test 6: Related Entity Query
# Query: "Which test cases are associated with defect BUG-009235?"
# Expected: ENTITY_EXPANSION mode targeting test_case and test_execution evidence.
# -------------------------------------------------------------------------
def test_case_6_related_entity_query():
    router = LLMQueryRouter(config=RetrievalConfig(groq_api_key=""))
    q = "Which test cases are associated with defect BUG-009235?"
    route_out = router.route(q)

    assert route_out.retrieval_mode == RetrievalMode.ENTITY_EXPANSION
    assert route_out.primary_entity_id == "BUG-009235"
    assert route_out.requested_attribute == "test_cases"
    assert "test_case" in route_out.target_document_types

