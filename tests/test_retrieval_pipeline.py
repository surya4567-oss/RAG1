"""Comprehensive unit tests for the Advanced Retrieval Pipeline."""

from unittest.mock import MagicMock, patch
import pytest

from src.retrieval.answer_generator import EvidenceGroundedAnswerGenerator
from src.retrieval.config import RetrievalConfig
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.pipeline import AdvancedRetrievalPipeline
from src.retrieval.reranker import LocalCrossEncoderReranker
from src.retrieval.router import LLMQueryRouter
from src.retrieval.schemas import (
    CandidateChunk,
    Evidence,
    RetrievalResponse,
    RetrievalRoute,
    RetrievalStatus,
    RouterOutput,
)


# -------------------------------------------------------------------------
# Test 1: Router Identifier Extraction & Schema Filter Validation
# -------------------------------------------------------------------------
def test_router_identifier_extraction():
    router = LLMQueryRouter(config=RetrievalConfig())

    q1 = "What is the precondition for TC-000001 in build 22?"
    ids1 = router.extract_regex_identifiers(q1)
    assert ids1.get("test_case_id") == "TC-000001"
    assert ids1.get("build_number") == "22"

    q2 = "Find defect BUG-000962 and related execution EXEC-00045"
    ids2 = router.extract_regex_identifiers(q2)
    assert ids2.get("defect_id") == "BUG-000962"
    assert ids2.get("execution_id") == "EXEC-00045"

    q3 = "Why did we encounter AccessError in requirement REQ-1924?"
    ids3 = router.extract_regex_identifiers(q3)
    assert ids3.get("requirement_id") == "REQ-1924"
    assert ids3.get("error_code") == "ACCESSERROR"

    q4 = "Show test cases under EPIC-22 and STORY-1618 found in B-5521"
    ids4 = router.extract_regex_identifiers(q4)
    assert ids4.get("epic") == "EPIC-22"
    assert ids4.get("story") == "STORY-1618"
    assert ids4.get("build_number") == "B-5521"

    # Test SRS Functional Requirement ID formats
    q5 = "Find details on requirement FR-PUR-003 and FR-COM-001"
    ids5 = router.extract_regex_identifiers(q5)
    assert ids5.get("requirement_id") == "FR-PUR-003"


def test_router_document_type_normalization_and_modality_detection():
    router = LLMQueryRouter(config=RetrievalConfig())

    # Modality detection from question
    q_screen = "Show screenshots for TC-014555"
    ids_screen = router.extract_regex_identifiers(q_screen)
    assert ids_screen.get("test_case_id") == "TC-014555"
    assert ids_screen.get("document_type") == "screenshot_evidence"

    q_log = "Retrieve failure logs for defect BUG-003596"
    ids_log = router.extract_regex_identifiers(q_log)
    assert ids_log.get("defect_id") == "BUG-003596"
    assert ids_log.get("document_type") == "failure_log"

    # Normalization of synonyms and plurals in proposed_filters
    proposed = {
        "document_type": "screenshots",
        "has_error_message": "true",
        "http_status": "404",
        "regression": "false",
    }
    sanitized = router._validate_and_sanitize_filters(proposed, {})
    assert sanitized["document_type"] == "screenshot_evidence"
    assert sanitized["has_error_message"] is True
    assert sanitized["http_status"] == 404
    assert sanitized["regression"] is False

    # Failure log synonym
    proposed_log = {"document_type": "error_logs"}
    sanitized_log = router._validate_and_sanitize_filters(proposed_log, {})
    assert sanitized_log["document_type"] == "failure_log"


def test_router_fallback_extracts_modality():
    router = LLMQueryRouter(config=RetrievalConfig(groq_api_key=""))

    out = router.route("Show screenshot for TC-014555")
    assert out.validated_filters.get("test_case_id") == "TC-014555"
    assert out.validated_filters.get("document_type") == "screenshot_evidence"

    out_log = router.route("Find failure logs for BUG-003596")
    assert out_log.validated_filters.get("defect_id") == "BUG-003596"
    assert out_log.validated_filters.get("document_type") == "failure_log"


def test_router_filter_sanitization_rejects_hallucinated_fields():
    router = LLMQueryRouter(config=RetrievalConfig())

    # Propose invalid/hallucinated fields alongside valid ones
    proposed = {
        "fake_column": "some_value",
        "invented_metric": 123,
        "module": "POS",
        "environment": "DEV",
    }
    regex_ids = {"test_case_id": "TC-000001"}

    sanitized = router._validate_and_sanitize_filters(proposed, regex_ids)
    assert "fake_column" not in sanitized
    assert "invented_metric" not in sanitized
    assert sanitized["module"] == "POS"
    assert sanitized["environment"] == "DEV"
    assert sanitized["test_case_id"] == "TC-000001"


def test_router_fallback_behavior():
    router = LLMQueryRouter(config=RetrievalConfig(groq_api_key=""))

    # Exact ID query should fallback to keyword
    out_keyword = router.route("Show test case TC-000001")
    assert out_keyword.strategy == RetrievalRoute.KEYWORD
    assert out_keyword.validated_filters.get("test_case_id") == "TC-000001"

    # Conceptual query should fallback to dense
    out_dense = router.route("Explain the payment processing workflow")
    assert out_dense.strategy == RetrievalRoute.DENSE

    # Mixed query should fallback to hybrid
    out_hybrid = router.route("Explain why TC-000001 failed in DEV build 22")
    assert out_hybrid.strategy == RetrievalRoute.HYBRID


# -------------------------------------------------------------------------
# Test 2: Reciprocal Rank Fusion (RRF) & Deduplication
# -------------------------------------------------------------------------
def test_reciprocal_rank_fusion():
    dense = [
        CandidateChunk(chunk_id="chunk-A", text="Chunk A text", score=0.9, original_rank=1),
        CandidateChunk(chunk_id="chunk-B", text="Chunk B text", score=0.8, original_rank=2),
        CandidateChunk(chunk_id="chunk-C", text="Chunk C text", score=0.7, original_rank=3),
    ]
    sparse = [
        CandidateChunk(chunk_id="chunk-B", text="Chunk B text", score=12.5, original_rank=1),
        CandidateChunk(chunk_id="chunk-D", text="Chunk D text", score=10.0, original_rank=2),
        CandidateChunk(chunk_id="chunk-A", text="Chunk A text", score=8.0, original_rank=3),
    ]

    fused = reciprocal_rank_fusion(dense, sparse, k=60, top_n=10)

    # Chunk B is in both lists (dense rank 2, sparse rank 1):
    # RRF(B) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # Chunk A is in both lists (dense rank 1, sparse rank 3):
    # RRF(A) = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
    assert len(fused) == 4  # A, B, C, D (deduplicated)
    assert fused[0].chunk_id == "chunk-B"
    assert fused[1].chunk_id == "chunk-A"
    assert fused[0].dense_rank == 2
    assert fused[0].sparse_rank == 1
    assert fused[0].rrf_score is not None
    assert fused[0].retrieval_route == "hybrid"


# -------------------------------------------------------------------------
# Test 3: Reranker Ordering & Fallback
# -------------------------------------------------------------------------
def test_reranker_ordering():
    reranker = LocalCrossEncoderReranker(config=RetrievalConfig())

    # Mock CrossEncoder predict
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.12, 0.95, -0.45]
    reranker._load_model = MagicMock(return_value=mock_model)

    candidates = [
        CandidateChunk(chunk_id="c1", text="text 1", score=0.5, original_rank=1),
        CandidateChunk(chunk_id="c2", text="text 2", score=0.4, original_rank=2),
        CandidateChunk(chunk_id="c3", text="text 3", score=0.3, original_rank=3),
    ]

    reranked = reranker.rerank("test query", candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c2"  # score 0.95
    assert reranked[1].chunk_id == "c1"  # score 0.12
    assert reranked[0].rerank_score == 0.95


# -------------------------------------------------------------------------
# Test 4: Evidence Grounding & Citation Validation
# -------------------------------------------------------------------------
def test_citation_validation_rejects_hallucinations():
    generator = EvidenceGroundedAnswerGenerator(config=RetrievalConfig())

    candidates = [
        CandidateChunk(
            chunk_id="valid-chunk-001",
            text="The accountant confirmed the invoice successfully.",
            metadata={"document_type": "test_case", "source_file": "testcases.csv", "record_id": "TC-001"},
        )
    ]
    chunk_map = {c.chunk_id: c for c in candidates}

    raw_evidence = [
        {"chunk_id": "valid-chunk-001", "supporting_text": "confirmed invoice"},
        {"chunk_id": "fake-hallucinated-id-999", "supporting_text": "invented fact"},
    ]

    validated = generator._validate_evidence_citations(raw_evidence, chunk_map)

    assert len(validated) == 1
    assert validated[0].chunk_id == "valid-chunk-001"
    assert validated[0].document_type == "test_case"
    assert validated[0].source_file == "testcases.csv"


def test_empty_retrieval_returns_insufficient_evidence():
    generator = EvidenceGroundedAnswerGenerator(config=RetrievalConfig())

    response = generator.generate(
        question="What is the root cause of BUG-999?",
        candidates=[],
        retrieval_route=RetrievalRoute.KEYWORD,
    )

    assert response.status == RetrievalStatus.INSUFFICIENT_EVIDENCE
    assert len(response.evidence) == 0
    assert "No relevant evidence" in response.answer


# -------------------------------------------------------------------------
# Test 5: End-to-End Pipeline Orchestration (Mocked LLM & Qdrant)
# -------------------------------------------------------------------------
def test_end_to_end_pipeline_mocked():
    mock_router = MagicMock()
    mock_router.route.return_value = RouterOutput(
        intent="Lookup TC-001",
        strategy=RetrievalRoute.KEYWORD,
        extracted_identifiers={"test_case_id": "TC-001"},
        validated_filters={"test_case_id": "TC-001"},
        reasoning="Exact ID match",
    )

    mock_keyword = MagicMock()
    mock_keyword.retrieve.return_value = [
        CandidateChunk(
            chunk_id="chunk-tc-001",
            text="Test Case TC-001 preconditions: user logged in.",
            score=5.5,
            original_rank=1,
            metadata={"document_type": "test_case", "record_id": "TC-001"},
        )
    ]

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query=None, candidates=None, top_k=None, **kwargs: candidates or []

    mock_generator = MagicMock()
    mock_generator.generate.return_value = RetrievalResponse(
        question="Details of TC-001",
        answer="TC-001 requires user to be logged in.",
        status=RetrievalStatus.ANSWERED,
        retrieval_route=RetrievalRoute.KEYWORD,
        evidence=[
            Evidence(
                chunk_id="chunk-tc-001",
                document_type="test_case",
                record_id="TC-001",
                supporting_text="user logged in",
            )
        ],
        limitations=[],
    )

    pipeline = AdvancedRetrievalPipeline(
        router=mock_router,
        keyword_retriever=mock_keyword,
        reranker=mock_reranker,
        generator=mock_generator,
        qdrant_client=MagicMock(),
    )

    result = pipeline.run("What is the precondition for TC-001?")

    assert result.status == RetrievalStatus.ANSWERED
    assert result.retrieval_route == RetrievalRoute.KEYWORD
    assert len(result.evidence) == 1
    assert result.evidence[0].chunk_id == "chunk-tc-001"
    assert "total_duration_seconds" in result.metadata


# -------------------------------------------------------------------------
# Test 6: Metadata Filter Failure Automatically Switches Strategy
# -------------------------------------------------------------------------
def test_router_filter_failure_fallback():
    router = LLMQueryRouter(config=RetrievalConfig(groq_api_key=""))

    # When filter fails on a conceptual question, router switches to DENSE without filters
    route_dense = router.route_on_filter_failure(
        question="Explain how the invoice confirmation workflow operates",
        failed_filters={"module": "Account"},
    )
    assert route_dense.strategy == RetrievalRoute.DENSE
    assert route_dense.validated_filters == {}

    # When filter fails on a mixed query, router switches to HYBRID without filters
    route_hybrid = router.route_on_filter_failure(
        question="Why did TC-000001 fail with AccessError?",
        failed_filters={"test_case_id": "TC-000001"},
    )
    assert route_hybrid.strategy == RetrievalRoute.HYBRID
    assert route_hybrid.validated_filters == {}


def test_pipeline_metadata_filter_failure_switches_strategy():
    mock_router = LLMQueryRouter(config=RetrievalConfig(groq_api_key=""))

    # Keyword retriever: returns empty when filtered (filter failed), returns candidate when unfiltered
    mock_keyword = MagicMock()
    mock_keyword.retrieve.side_effect = lambda query=None, top_k=None, filters=None, **kwargs: (
        [] if filters else [
            CandidateChunk(
                chunk_id="fallback-chunk-01",
                text="Invoice workflow overview found via full-text search.",
                score=8.0,
                original_rank=1,
                metadata={"document_type": "srs_section"},
            )
        ]
    )

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query=None, candidates=None, top_k=None, **kwargs: candidates or []

    mock_generator = MagicMock()
    mock_generator.generate.return_value = RetrievalResponse(
        question="Show TC-000001 invoice details",
        answer="Found invoice details via unfiltered query fallback.",
        status=RetrievalStatus.ANSWERED,
        retrieval_route=RetrievalRoute.KEYWORD,
        evidence=[
            Evidence(
                chunk_id="fallback-chunk-01",
                document_type="srs_section",
                supporting_text="Invoice workflow overview",
            )
        ],
        limitations=[],
    )

    pipeline = AdvancedRetrievalPipeline(
        router=mock_router,
        keyword_retriever=mock_keyword,
        reranker=mock_reranker,
        generator=mock_generator,
        qdrant_client=MagicMock(),
    )

    # Initial query has TC-000001 which sets a filter.
    # The filter fails (returns []), so pipeline should invoke route_on_filter_failure,
    # switch to unfiltered retrieval, find fallback-chunk-01, and return ANSWERED.
    result = pipeline.run("Show TC-000001 invoice details")

    assert result.status == RetrievalStatus.ANSWERED
    assert len(result.evidence) == 1
    assert result.evidence[0].chunk_id == "fallback-chunk-01"
    assert result.metadata.get("filter_fallback_applied") is True

