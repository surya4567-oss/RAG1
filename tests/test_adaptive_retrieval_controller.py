"""Comprehensive test suite for the Generic Query-Adaptive Retrieval Controller.

Validates the 8 core requirements from Section 11:
1. Exact defect queries.
2. Defect-to-screenshot queries (generic relationship expansion).
3. Execution-to-failure-log queries (cross-document log expansion).
4. Requirement-to-test-case queries.
5. Conceptual questions (unrestricted semantic search).
6. Queries with missing identifiers (controlled handling & insufficient evidence).
7. Queries where metadata fields are absent.
8. Queries with multiple conflicting evidence values.
"""

from unittest.mock import MagicMock
import pytest

from src.retrieval.answer_generator import EvidenceGroundedAnswerGenerator
from src.retrieval.config import RetrievalConfig
from src.retrieval.controller import (
    ATTRIBUTE_DOCUMENT_TYPE_MAP,
    QueryAdaptiveRetrievalController,
    RetrievalQualityAssessment,
)
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


@pytest.fixture
def mock_qdrant_client():
    client = MagicMock()
    return client


@pytest.fixture
def adaptive_controller(mock_qdrant_client):
    cfg = RetrievalConfig(groq_api_key="", colab_llm_base_url="")
    mock_dense = MagicMock()
    mock_keyword = MagicMock()
    mock_hybrid = MagicMock()
    mock_reranker = MagicMock()

    mock_reranker.rerank.side_effect = lambda query=None, candidates=None, top_k=None, **kwargs: (candidates or [])[:top_k]

    return QueryAdaptiveRetrievalController(
        config=cfg,
        client=mock_qdrant_client,
        dense_retriever=mock_dense,
        keyword_retriever=mock_keyword,
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
    )


def _make_fallback_router() -> LLMQueryRouter:
    return LLMQueryRouter(config=RetrievalConfig(groq_api_key="", colab_llm_base_url=""))


# -------------------------------------------------------------------------
# Test 1: Exact Defect Queries
# -------------------------------------------------------------------------
def test_1_exact_defect_query(adaptive_controller):
    router = _make_fallback_router()
    q = "What is the status of defect BUG-000001?"
    r_out = router.route(q)

    assert r_out.primary_entity_id == "BUG-000001"
    assert r_out.primary_entity_type == "defect"
    assert r_out.validated_filters.get("defect_id") == "BUG-000001"

    # Mock initial retrieval returning the defect record
    mock_chunk = CandidateChunk(
        chunk_id="doc_defect_BUG-000001_c0",
        text="Defect BUG-000001: Severity Major, Status Closed, Module Project.",
        score=10.0,
        original_rank=1,
        metadata={"document_type": "defect", "defect_id": "BUG-000001", "severity": "Major", "status": "Closed"},
    )
    adaptive_controller.keyword_retriever.retrieve.return_value = [mock_chunk]

    evidence, assessment, telemetry = adaptive_controller.execute_adaptive_retrieval(q, r_out)

    assert len(evidence) == 1
    assert assessment.is_sufficient is True
    assert assessment.quality_status == "sufficient"
    assert assessment.attribute_values_found.get("status") == ["Closed"]


# -------------------------------------------------------------------------
# Test 2: Defect-to-Screenshot Queries (Generic Foreign-Key Expansion)
# -------------------------------------------------------------------------
def test_2_defect_to_screenshot_expansion(adaptive_controller, mock_qdrant_client):
    router = _make_fallback_router()
    q = "What is screen name of bug id BUG-009235?"
    r_out = router.route(q)

    assert r_out.retrieval_mode == RetrievalMode.ENTITY_EXPANSION
    assert r_out.primary_entity_id == "BUG-009235"
    assert r_out.requested_attribute == "screen_name"

    # Seed defect contains linked execution_id EXEC-014086
    seed_defect = CandidateChunk(
        chunk_id="doc_defect_BUG-009235_c0",
        text="Defect BUG-009235 in Documents module. Linked execution EXEC-014086.",
        score=10.0,
        original_rank=1,
        metadata={"document_type": "defect", "defect_id": "BUG-009235", "execution_id": "EXEC-014086"},
    )
    adaptive_controller.hybrid_retriever.retrieve.return_value = [seed_defect]

    # Mock Qdrant scroll returning screenshot records matching discovered execution_id via MatchAny
    screen_pt = MagicMock()
    screen_pt.id = "screen-pt-1"
    screen_pt.payload = {
        "chunk_id": "ocr_batch_004_c0",
        "document_type": "screenshot_evidence",
        "screen_name": "POS Dashboard",
        "execution_id": "EXEC-014086",
        "defect_id": "BUG-002759",
        "page_content": "Screenshot POS Dashboard showing error banner.",
    }
    mock_qdrant_client.scroll.return_value = ([screen_pt], None)

    evidence, assessment, telemetry = adaptive_controller.execute_adaptive_retrieval(q, r_out)

    assert any(c.metadata.get("screen_name") == "POS Dashboard" for c in evidence)
    assert telemetry["expanded_candidates_count"] > 0
    assert assessment.has_requested_attribute is True
    assert "POS Dashboard" in assessment.attribute_values_found.get("screen_name", [])


# -------------------------------------------------------------------------
# Test 3: Execution-to-Failure-Log Queries (Cross-Document Log Expansion)
# -------------------------------------------------------------------------
def test_3_execution_to_failure_log_expansion(adaptive_controller, mock_qdrant_client):
    router = _make_fallback_router()
    q = "Show failure logs and traceback for test execution EXEC-016067"
    r_out = router.route(q)

    assert r_out.primary_entity_id == "EXEC-016067"
    assert "failure_log" in r_out.target_document_types

    seed_exec = CandidateChunk(
        chunk_id="doc_test_execution_EXEC-016067_c0",
        text="Execution EXEC-016067: Failed on Edge Android.",
        score=8.0,
        original_rank=1,
        metadata={"document_type": "test_execution", "execution_id": "EXEC-016067", "status": "Failed"},
    )
    adaptive_controller.hybrid_retriever.retrieve.return_value = [seed_exec]

    log_pt = MagicMock()
    log_pt.id = "log-pt-1"
    log_pt.payload = {
        "chunk_id": "doc_failure_log_logs_001_c0",
        "document_type": "failure_log",
        "execution_id": "EXEC-016067",
        "exception_type": "AccessError",
        "http_status": 404,
        "page_content": "Failure Log: AccessError: You do not have the required permissions.",
    }
    mock_qdrant_client.scroll.return_value = ([log_pt], None)

    evidence, assessment, telemetry = adaptive_controller.execute_adaptive_retrieval(q, r_out)

    assert any(c.metadata.get("document_type") == "failure_log" for c in evidence)
    assert any(c.metadata.get("exception_type") == "AccessError" for c in evidence)


# -------------------------------------------------------------------------
# Test 4: Requirement-to-Test-Case Queries
# -------------------------------------------------------------------------
def test_4_requirement_to_test_case_expansion(adaptive_controller, mock_qdrant_client):
    router = _make_fallback_router()
    q = "Which test cases verify requirement FR-PUR-003?"
    r_out = router.route(q)

    assert r_out.primary_entity_id == "FR-PUR-003"
    assert "test_case" in r_out.target_document_types

    seed_req = CandidateChunk(
        chunk_id="doc_srs_section_FR-PUR-003_c0",
        text="Requirement FR-PUR-003: Purchase approval workflows.",
        score=9.0,
        original_rank=1,
        metadata={"document_type": "srs_section", "requirement_id": "FR-PUR-003"},
    )
    adaptive_controller.hybrid_retriever.retrieve.return_value = [seed_req]

    tc_pt = MagicMock()
    tc_pt.id = "tc-pt-1"
    tc_pt.payload = {
        "chunk_id": "doc_test_case_TC-004455_c0",
        "document_type": "test_case",
        "test_case_id": "TC-004455",
        "requirement_id": "FR-PUR-003",
        "page_content": "Test Case TC-004455: Verify Purchase order approval limits.",
    }
    mock_qdrant_client.scroll.return_value = ([tc_pt], None)

    evidence, assessment, telemetry = adaptive_controller.execute_adaptive_retrieval(q, r_out)

    assert any(c.metadata.get("test_case_id") == "TC-004455" for c in evidence)


# -------------------------------------------------------------------------
# Test 5: Conceptual Questions (Unrestricted Semantic Search)
# -------------------------------------------------------------------------
def test_5_conceptual_question_routing(adaptive_controller):
    router = _make_fallback_router()
    q = "What are the common causes of test execution failures?"
    r_out = router.route(q)

    assert r_out.retrieval_mode == RetrievalMode.CONCEPTUAL
    assert r_out.primary_entity_id is None
    assert len(r_out.validated_filters) == 0

    mock_dense_chunk = CandidateChunk(
        chunk_id="chunk-dense-1",
        text="Common causes of execution failures in Odoo ERP include permission errors and timeout exceptions.",
        score=0.88,
        original_rank=1,
        metadata={"document_type": "srs_section"},
    )
    adaptive_controller.dense_retriever.retrieve.return_value = [mock_dense_chunk]

    evidence, assessment, telemetry = adaptive_controller.execute_adaptive_retrieval(q, r_out)
    assert len(evidence) == 1
    assert telemetry["expanded_candidates_count"] == 0  # No unnecessary expansion for conceptual query


# -------------------------------------------------------------------------
# Test 6: Queries with Missing / Non-Existent Identifiers
# -------------------------------------------------------------------------
def test_6_missing_identifier_insufficient_evidence(adaptive_controller):
    router = _make_fallback_router()
    q = "What is the screen name of BUG-999999?"
    r_out = router.route(q)

    # Empty initial retrieval
    adaptive_controller.keyword_retriever.retrieve.return_value = []
    adaptive_controller.hybrid_retriever.retrieve.return_value = []
    adaptive_controller.dense_retriever.retrieve.return_value = []

    evidence, assessment, telemetry = adaptive_controller.execute_adaptive_retrieval(q, r_out)

    # Must assess as insufficient
    assert assessment.is_sufficient is False
    assert assessment.quality_status in ("empty", "insufficient_attribute")
    assert len(evidence) == 0

    # Generator returns INSUFFICIENT_EVIDENCE
    generator = EvidenceGroundedAnswerGenerator(config=RetrievalConfig(groq_api_key=""))
    resp = generator.generate(q, evidence)
    assert resp.status == RetrievalStatus.INSUFFICIENT_EVIDENCE


# -------------------------------------------------------------------------
# Test 7: Queries Where Requested Metadata Fields Are Absent
# -------------------------------------------------------------------------
def test_7_metadata_field_absent_in_records(adaptive_controller):
    q = "What is the quantum_flux_coefficient of TC-000001?"
    r_out = RouterOutput(
        intent="Query for missing field",
        strategy=RetrievalRoute.KEYWORD,
        retrieval_mode=RetrievalMode.DIRECT_ENTITY,
        primary_entity_id="TC-000001",
        primary_entity_type="test_case",
        requested_attribute="quantum_flux_coefficient",
        validated_filters={"test_case_id": "TC-000001"},
        reasoning="Test query for non-existent field",
    )

    tc_chunk = CandidateChunk(
        chunk_id="doc_test_case_TC-000001_c0",
        text="Test Case TC-000001: Preconditions and steps for POS order.",
        score=5.0,
        original_rank=1,
        metadata={"document_type": "test_case", "test_case_id": "TC-000001", "status": "Review"},
    )
    adaptive_controller.keyword_retriever.retrieve.return_value = [tc_chunk]

    evidence, assessment, telemetry = adaptive_controller.execute_adaptive_retrieval(q, r_out)

    assert assessment.has_requested_attribute is False
    assert assessment.quality_status == "partial"
    assert "quantum_flux_coefficient" not in assessment.attribute_values_found


# -------------------------------------------------------------------------
# Test 8: Queries with Multiple Conflicting Evidence Values
# -------------------------------------------------------------------------
def test_8_conflicting_evidence_detection(adaptive_controller):
    q = "What is the status of defect BUG-001000?"
    r_out = RouterOutput(
        intent="Query status of defect",
        strategy=RetrievalRoute.KEYWORD,
        retrieval_mode=RetrievalMode.DIRECT_ENTITY,
        primary_entity_id="BUG-001000",
        requested_attribute="status",
        validated_filters={"defect_id": "BUG-001000"},
        reasoning="Status check",
    )

    chunk_a = CandidateChunk(
        chunk_id="chunk-defect-a",
        text="Defect BUG-001000 status is marked Closed.",
        score=6.0,
        original_rank=1,
        metadata={"document_type": "defect", "defect_id": "BUG-001000", "status": "Closed"},
    )
    chunk_b = CandidateChunk(
        chunk_id="chunk-defect-b",
        text="Defect BUG-001000 status is marked Reopened in latest execution.",
        score=5.8,
        original_rank=2,
        metadata={"document_type": "defect", "defect_id": "BUG-001000", "status": "Reopened"},
    )
    adaptive_controller.keyword_retriever.retrieve.return_value = [chunk_a, chunk_b]

    evidence, assessment, telemetry = adaptive_controller.execute_adaptive_retrieval(q, r_out)

    # Controller detects conflicting values for the status attribute
    assert assessment.conflicting_values is True
    assert set(assessment.attribute_values_found.get("status", [])) == {"Closed", "Reopened"}


# -------------------------------------------------------------------------
# Test 9: Exact Query Test - Server Log of Execution ID EXEC-001138
# -------------------------------------------------------------------------
def test_exact_query_server_log_execution_id(adaptive_controller, mock_qdrant_client):
    """Test exact user query: 'What is server log of execution id EXEC-001138?'

    Verifies:
    1. KeywordRetriever returns 0 results on sparse collection without crashing (no Record.score).
    2. Controller catches 0 BM25 results, executes controlled metadata scroll on primary collection.
    3. Controller harvests EXEC-001138, dynamically targets 'failure_log' modality.
    4. Controller executes batched evidence expansion via MatchAny to find failure logs.
    5. Deduplicates candidates and sends them to reranker and answer generator.
    """
    router = _make_fallback_router()
    q = "What is server log of execution id EXEC-001138?"
    r_out = router.route(q)

    assert r_out.primary_entity_id == "EXEC-001138"
    assert "failure_log" in r_out.target_document_types

    # Stage 1: Initial retrieval (hybrid or keyword) returns 0 results
    adaptive_controller.hybrid_retriever.retrieve.return_value = []
    adaptive_controller.keyword_retriever.retrieve.return_value = []
    adaptive_controller.dense_retriever.retrieve.return_value = []

    # Stage 2: Metadata scroll on primary collection returns execution record
    # spec explicitly excludes 'score' attribute to simulate actual Qdrant Record objects
    exec_rec = MagicMock(spec=["id", "payload"])
    exec_rec.id = "pt-exec-1138"
    exec_rec.payload = {
        "chunk_id": "doc_test_execution_EXEC-001138_c0",
        "document_type": "test_execution",
        "execution_id": "EXEC-001138",
        "status": "Failed",
        "page_content": "Execution EXEC-001138 failed during POS checkout.",
    }

    # Stage 3: Expansion scroll returns failure log record (also a Record object without score)
    log_rec = MagicMock(spec=["id", "payload"])
    log_rec.id = "pt-log-1138"
    log_rec.payload = {
        "chunk_id": "doc_failure_log_server_001_c0",
        "document_type": "failure_log",
        "execution_id": "EXEC-001138",
        "exception_type": "DatabaseConnectionTimeout",
        "http_status": 500,
        "error_code": "500",
        "page_content": "Server Log: DatabaseConnectionTimeout: pool exhausted at port 5432.",
    }

    def scroll_side_effect(collection_name, scroll_filter, **kwargs):
        must = scroll_filter.must if scroll_filter else []
        is_failure_log = any(
            getattr(cond, "key", "") == "document_type"
            and getattr(getattr(cond, "match", None), "value", "") == "failure_log"
            for cond in must
        )
        if is_failure_log:
            return ([log_rec], None)
        return ([exec_rec], None)

    mock_qdrant_client.scroll.side_effect = scroll_side_effect

    evidence, assessment, telemetry = adaptive_controller.execute_adaptive_retrieval(q, r_out)

    # Report candidate counts from each stage
    print("\nCandidate telemetry for 'What is server log of execution id EXEC-001138?':", telemetry)
    assert telemetry["initial_candidates_count"] == 0  # BM25 returned 0
    assert telemetry["filter_fallback_applied"] is True  # Metadata scroll fallback succeeded
    assert telemetry["expanded_candidates_count"] >= 0
    assert telemetry["deduplicated_candidates_count"] >= 1  # Deduplicated unique candidates
    assert telemetry["final_evidence_count"] >= 1

    # Verify failure-log chunks reach final evidence without AttributeErrors
    failure_logs = [c for c in evidence if c.metadata.get("document_type") == "failure_log"]
    assert len(failure_logs) == 1
    assert failure_logs[0].metadata.get("exception_type") == "DatabaseConnectionTimeout"
    assert failure_logs[0].score == 0.0 or failure_logs[0].rerank_score is not None
    assert failure_logs[0].sparse_rank is None  # Not assigned fabricated sparse rank


def test_exact_query_server_log_pipeline_e2e(mock_qdrant_client):
    """Verify end-to-end pipeline execution for 'What is server log of execution id EXEC-001138?'

    Confirms that retrieved failure-log chunks pass through the controller, reach the reranker,
    and are provided to the answer generator for citation-backed synthesis.
    """
    router = _make_fallback_router()
    mock_keyword = MagicMock()
    mock_keyword.retrieve.return_value = []  # Sparse search yields 0 results
    mock_dense = MagicMock()
    mock_dense.retrieve.return_value = []
    mock_hybrid = MagicMock()
    mock_hybrid.retrieve.return_value = []

    # Record objects without 'score'
    exec_rec = MagicMock(spec=["id", "payload"])
    exec_rec.id = "pt-exec-1138"
    exec_rec.payload = {
        "chunk_id": "doc_test_execution_EXEC-001138_c0",
        "document_type": "test_execution",
        "execution_id": "EXEC-001138",
        "status": "Failed",
        "page_content": "Execution EXEC-001138 failed.",
    }

    log_rec = MagicMock(spec=["id", "payload"])
    log_rec.id = "pt-log-1138"
    log_rec.payload = {
        "chunk_id": "doc_failure_log_server_001_c0",
        "document_type": "failure_log",
        "execution_id": "EXEC-001138",
        "exception_type": "DatabaseConnectionTimeout",
        "http_status": 500,
        "error_code": "500",
        "page_content": "Server Log: DatabaseConnectionTimeout: pool exhausted at port 5432.",
    }

    def scroll_side_effect(collection_name, scroll_filter, **kwargs):
        must = scroll_filter.must if scroll_filter else []
        is_failure_log = any(
            getattr(cond, "key", "") == "document_type"
            and getattr(getattr(cond, "match", None), "value", "") == "failure_log"
            for cond in must
        )
        if is_failure_log:
            return ([log_rec], None)
        return ([exec_rec], None)

    mock_qdrant_client.scroll.side_effect = scroll_side_effect

    mock_reranker = MagicMock()
    mock_reranker.rerank.side_effect = lambda query=None, candidates=None, top_k=None, **kwargs: candidates or []

    mock_generator = MagicMock()
    mock_generator.generate.side_effect = lambda question=None, candidates=None, retrieval_route=None, **kwargs: RetrievalResponse(
        question=question or "",
        answer="The server log for EXEC-001138 indicates DatabaseConnectionTimeout (HTTP 500) [CHUNK_2].",
        status=RetrievalStatus.ANSWERED,
        retrieval_route=retrieval_route or RetrievalRoute.KEYWORD,
        evidence=[
            Evidence(
                chunk_id=c.chunk_id,
                document_type=c.metadata.get("document_type", "unknown"),
                supporting_text=c.text,
            )
            for c in (candidates or [])
        ],
        limitations=[],
    )

    pipeline = AdvancedRetrievalPipeline(
        router=router,
        keyword_retriever=mock_keyword,
        dense_retriever=mock_dense,
        hybrid_retriever=mock_hybrid,
        reranker=mock_reranker,
        generator=mock_generator,
        qdrant_client=mock_qdrant_client,
    )

    q = "What is server log of execution id EXEC-001138?"
    result = pipeline.run(q)

    assert result.status == RetrievalStatus.ANSWERED
    # Assert failure log reached the answer generator evidence
    assert any(e.chunk_id == "doc_failure_log_server_001_c0" for e in result.evidence)
    assert any(e.document_type == "failure_log" for e in result.evidence)
    assert result.metadata.get("final_evidence_count") >= 1
    assert result.metadata.get("filter_fallback_applied") is True
