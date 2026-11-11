"""Unit and integration tests for RAG-01 through RAG-08 fixes."""

import sqlite3
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.retrieval.answer_generator import EvidenceGroundedAnswerGenerator
from src.retrieval.config import RetrievalConfig
from src.retrieval.controller import QueryAdaptiveRetrievalController
from src.retrieval.keyword_retriever import KeywordRetriever
from src.retrieval.pipeline import AdvancedRetrievalPipeline
from src.retrieval.schemas import (
    CandidateChunk,
    Evidence,
    RetrievalMode,
    RetrievalResponse,
    RetrievalRoute,
    RetrievalStatus,
    RouterOutput,
)


# ---------------------------------------------------------------------------
# RAG-01 & RAG-07: Sparse Collection Metadata & Chunk ID Normalization
# ---------------------------------------------------------------------------

def test_sparse_collection_has_full_metadata():
    """Verify points in qa_knowledge_sparse have full metadata payload and BM25 vectors."""
    db_path = Path("data/data1/qdrant_db/collection/qa_knowledge_sparse/storage.sqlite")
    if not db_path.exists():
        pytest.skip("Local Qdrant sparse SQLite database not present.")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, point FROM points LIMIT 10")
    rows = cur.fetchall()
    conn.close()

    assert len(rows) == 10, "Sparse collection should contain points."
    for pid, blob in rows:
        pt = pickle.loads(blob)
        assert pt.payload is not None, "Payload must not be empty"
        assert "chunk_id" in pt.payload, "chunk_id must be in sparse payload"
        assert "document_type" in pt.payload, "document_type must be in sparse payload"
        assert pt.payload["document_type"] != "unknown", "document_type must be known"
        assert "bm25" in pt.vector, "Sparse vector bm25 must be intact"
        assert len(pt.vector["bm25"].indices) > 0, "Sparse indices must not be empty"


def test_sparse_candidates_have_chunk_ids_not_uuids():
    """Verify CandidateChunk objects created from sparse points have readable chunk_ids, not raw UUIDs."""
    retriever = KeywordRetriever(config=RetrievalConfig())
    mock_point = MagicMock()
    mock_point.id = "fa349b09-ae55-5199-8c66-80a0add46288"
    mock_point.score = 3.45
    mock_point.payload = {
        "chunk_id": "doc_test_case_TC-000001_c0",
        "document_type": "test_case",
        "page_content": "Verify login workflow.",
    }

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = [mock_point]
    mock_client.query_points.return_value = mock_response

    with patch.object(retriever, "_get_client", return_value=mock_client), \
         patch.object(retriever, "_check_collection", return_value=None), \
         patch.object(retriever, "_get_sparse_embedder") as mock_embedder_factory:

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = MagicMock(indices=[1, 2], values=[1.0, 1.0])
        mock_embedder_factory.return_value = mock_embedder

        candidates = retriever.retrieve("Verify login", top_k=1)
        assert len(candidates) == 1
        cand = candidates[0]
        assert cand.chunk_id == "doc_test_case_TC-000001_c0"
        assert cand.chunk_id != "fa349b09-ae55-5199-8c66-80a0add46288"
        assert cand.metadata.get("document_type") == "test_case"
        assert cand.score == 3.45


# ---------------------------------------------------------------------------
# RAG-02: Metadata Filtered Sparse Retrieval
# ---------------------------------------------------------------------------

def test_metadata_filtered_sparse_retrieval_builds_correct_filter():
    """Verify KeywordRetriever constructs proper Qdrant Filter for document_type and entity IDs."""
    retriever = KeywordRetriever(config=RetrievalConfig())
    filters = {"document_type": "test_case", "test_case_id": "TC-000001"}
    q_filter = retriever._build_filter(filters)

    assert q_filter is not None
    assert len(q_filter.must) == 2
    keys = {cond.key for cond in q_filter.must}
    assert keys == {"document_type", "test_case_id"}


# ---------------------------------------------------------------------------
# RAG-03: Citation Resolution (Normalized / Canonical Match)
# ---------------------------------------------------------------------------

def test_citation_resolution_canonical_chunk_id():
    """Verify that truncated, indexed, or slightly mismatched chunk IDs resolve to canonical ID."""
    generator = EvidenceGroundedAnswerGenerator(config=RetrievalConfig())

    canonical_id = "doc_screenshot_evidence_ocr_batch_003.json_rec_0000_c0"
    cand = CandidateChunk(
        chunk_id=canonical_id,
        text="Screen showing Subscriptions dashboard with warning dialog.",
        score=0.9,
        metadata={
            "document_type": "screenshot_evidence",
            "record_id": "rec_0000_c0",
            "source_file": "screenshot_evidence_ocr_batch_003.json",
        },
    )
    chunk_map = {canonical_id: cand}
    candidates = [cand]

    # Test 1: Model omitted 'batch_' -> 'doc_screenshot_evidence_ocr_003.json_rec_0000_c0'
    res1 = generator._resolve_citation_chunk(
        "doc_screenshot_evidence_ocr_003.json_rec_0000_c0", chunk_map, candidates
    )
    assert res1 is not None
    assert res1[0] == canonical_id

    # Test 2: Model cited index '[CHUNK_1]'
    res2 = generator._resolve_citation_chunk("[CHUNK_1]", chunk_map, candidates)
    assert res2 is not None
    assert res2[0] == canonical_id

    # Test 3: Model cited 'CHUNK_1'
    res3 = generator._resolve_citation_chunk("CHUNK_1", chunk_map, candidates)
    assert res3 is not None
    assert res3[0] == canonical_id

    # Test 4: Model cited by record_id 'rec_0000_c0'
    res4 = generator._resolve_citation_chunk("rec_0000_c0", chunk_map, candidates)
    assert res4 is not None
    assert res4[0] == canonical_id


# ---------------------------------------------------------------------------
# RAG-04: Claim-Evidence Factual Support Verification
# ---------------------------------------------------------------------------

def test_factual_support_verification_grounded_vs_hallucinated():
    """Verify that claims with evidence overlap pass and completely hallucinated claims fail."""
    generator = EvidenceGroundedAnswerGenerator(config=RetrievalConfig())

    chunk_text = "Defect BUG-009235 in Subscriptions module causes a 403 AccessError on payment validation."

    # Test Grounded claim
    is_supp, conf, text = generator._verify_factual_support(
        supporting_text="Subscriptions module causes 403 AccessError",
        chunk_text=chunk_text,
        answer_text="The defect occurs in Subscriptions causing AccessError 403.",
    )
    assert is_supp is True
    assert conf >= 0.5
    assert "AccessError" in text

    # Test Hallucinated claim
    is_supp_fake, conf_fake, _ = generator._verify_factual_support(
        supporting_text="Database server crashed due to out of memory kernel panic",
        chunk_text=chunk_text,
        answer_text="A kernel panic occurred.",
    )
    assert is_supp_fake is False
    assert conf_fake == 0.0


# ---------------------------------------------------------------------------
# RAG-05: Generation Failure and RetrievalStatus Refinement
# ---------------------------------------------------------------------------

def test_missing_groq_key_returns_retrieval_only():
    """Verify that when Groq API key is not set, generator returns RETRIEVAL_ONLY (not ANSWERED)."""
    generator = EvidenceGroundedAnswerGenerator(config=RetrievalConfig(groq_api_key=""))
    cand = CandidateChunk(
        chunk_id="test_chunk_01",
        text="Test chunk content.",
        score=0.88,
        metadata={"document_type": "test_case"},
    )
    res = generator.generate("What is the test case?", [cand])
    assert res.status == RetrievalStatus.RETRIEVAL_ONLY
    assert len(res.evidence) == 1
    assert res.evidence[0].chunk_id == "test_chunk_01"


def test_groq_api_failure_returns_generation_failed():
    """Verify that when Groq API throws 429 RateLimit or other error, status is GENERATION_FAILED."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("429 Too Many Requests: Rate limit reached")

    generator = EvidenceGroundedAnswerGenerator(config=RetrievalConfig(groq_api_key="mock_key"), llm=mock_llm)
    cand = CandidateChunk(
        chunk_id="chunk_recovered_01",
        text="Payment module validation details.",
        score=0.95,
        metadata={"document_type": "defect"},
    )

    res = generator.generate("What is payment defect?", [cand])
    assert res.status == RetrievalStatus.GENERATION_FAILED
    assert "429" in res.metadata.get("error", "")
    assert len(res.evidence) >= 1
    assert res.evidence[0].chunk_id == "chunk_recovered_01"


def test_empty_candidates_returns_insufficient_evidence():
    """Verify that when candidates is empty, status is INSUFFICIENT_EVIDENCE."""
    generator = EvidenceGroundedAnswerGenerator(config=RetrievalConfig())
    res = generator.generate("Any question?", [])
    assert res.status == RetrievalStatus.INSUFFICIENT_EVIDENCE
    assert len(res.evidence) == 0


# ---------------------------------------------------------------------------
# RAG-06: Metadata Filtering and Controlled Fallback Precision
# ---------------------------------------------------------------------------

def test_metadata_filter_fallback_telemetry_and_logging():
    """Verify fallback logs reason and populates filter_fallback_applied and fallback_type."""
    mock_keyword = MagicMock()
    # Filtered call returns 0
    mock_keyword.retrieve.side_effect = lambda query, top_k=None, filters=None: (
        [] if filters else [
            CandidateChunk(
                chunk_id="fallback_chunk_01",
                text="Unfiltered fallback evidence content.",
                score=0.75,
                metadata={"document_type": "test_case"},
            )
        ]
    )

    mock_client = MagicMock()
    mock_client.scroll.return_value = ([], None)

    controller = QueryAdaptiveRetrievalController(
        config=RetrievalConfig(),
        client=mock_client,
        keyword_retriever=mock_keyword,
        reranker=None,
    )

    router_output = RouterOutput(
        intent="Find test case TC-000001",
        strategy=RetrievalRoute.KEYWORD,
        fallback_strategy=RetrievalRoute.KEYWORD,
        retrieval_mode=RetrievalMode.DIRECT_ENTITY,
        extracted_identifiers={"test_case_id": "TC-000001"},
        validated_filters={"test_case_id": "TC-000001"},
        reasoning="Test case lookup",
    )

    evidence, assessment, telemetry = controller.execute_adaptive_retrieval(
        query="Show TC-000001",
        router_output=router_output,
    )

    assert telemetry.get("filter_fallback_applied") is True
    assert "unfiltered" in telemetry.get("fallback_type", "")
    assert "fallback_reason" in telemetry
    assert len(evidence) == 1
    assert evidence[0].chunk_id == "fallback_chunk_01"


# ---------------------------------------------------------------------------
# RAG-08: Standardized Latency Metadata
# ---------------------------------------------------------------------------

def test_pipeline_standardized_latency_metadata():
    """Verify pipeline records all latency metrics and aliases without any None values."""
    mock_router = MagicMock()
    mock_router.route.return_value = RouterOutput(
        intent="Check defect",
        strategy=RetrievalRoute.KEYWORD,
        fallback_strategy=RetrievalRoute.HYBRID,
        retrieval_mode=RetrievalMode.CONCEPTUAL,
        extracted_identifiers={},
        validated_filters={},
        reasoning="Simple query",
    )

    mock_keyword = MagicMock()
    cand = CandidateChunk(
        chunk_id="cand_1",
        text="Defect description",
        score=0.9,
        metadata={"document_type": "defect"},
    )
    mock_keyword.retrieve.return_value = [cand]

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [cand]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = RetrievalResponse(
        question="What is defect?",
        answer="A defect is a bug.",
        status=RetrievalStatus.ANSWERED,
        retrieval_route=RetrievalRoute.KEYWORD,
        evidence=[Evidence(chunk_id="cand_1", supporting_text="Defect description")],
        metadata={},
    )

    pipeline = AdvancedRetrievalPipeline(
        router=mock_router,
        keyword_retriever=mock_keyword,
        reranker=mock_reranker,
        generator=mock_generator,
        qdrant_client=MagicMock(),
    )

    res = pipeline.run("What is defect?")
    m = res.metadata

    # Check that both canonical names and aliases exist and are numeric
    assert isinstance(m.get("total_duration_seconds"), float)
    assert isinstance(m.get("total_latency"), float)
    assert isinstance(m.get("routing_duration_seconds"), float)
    assert isinstance(m.get("routing_latency"), float)
    assert isinstance(m.get("retrieval_duration_seconds"), float)
    assert isinstance(m.get("retrieval_latency"), float)
    assert isinstance(m.get("generation_duration_seconds"), float)
    assert isinstance(m.get("generation_latency"), float)
    assert isinstance(m.get("reranker_duration_seconds"), float)
    assert isinstance(m.get("rerank_latency"), float)
    assert m["total_latency"] is not None


# ---------------------------------------------------------------------------
# RAG-09: Router Token Compactness & Rate Limit Backoff
# ---------------------------------------------------------------------------

def test_router_max_tokens_and_backoff():
    """Verify router initializes with bounded max_tokens and handles 429 with retry."""
    from src.retrieval.router import LLMQueryRouter

    cfg = RetrievalConfig(groq_api_key="gsk_dummy_test_key")
    router = LLMQueryRouter(config=cfg)

    with patch("langchain_groq.ChatGroq") as mock_groq_class:
        mock_llm_instance = MagicMock()
        mock_groq_class.return_value = mock_llm_instance

        llm = router._get_llm()
        assert llm is not None
        # Verify max_tokens=350 was passed to ChatGroq
        _, kwargs = mock_groq_class.call_args
        assert kwargs.get("max_tokens") == 350

        # Simulate 429 on first call, success on second
        from langchain_core.messages import AIMessage
        mock_llm_instance.invoke.side_effect = [
            Exception("Error code: 429 - Rate limit reached: please try again in 0.1s"),
            AIMessage(content='{"strategy": "hybrid", "retrieval_mode": "conceptual", "reasoning": "ok"}'),
        ]

        output = router.route("Explain ERP testing strategy")
        assert output.routing_source == "llm"
        assert output.strategy == RetrievalRoute.HYBRID
        assert mock_llm_instance.invoke.call_count == 2


# ---------------------------------------------------------------------------
# RAG-10: Exact Record Lookup Prevents Zero Candidates
# ---------------------------------------------------------------------------

def test_exact_record_lookup_prevents_zero_candidates():
    """Verify exact-ID queries guarantee target record presence even if sparse vector search returns 0."""
    mock_client = MagicMock()
    # Mock scroll returning point for defect BUG-000002
    mock_point = MagicMock()
    mock_point.id = "bug_point_002"
    mock_point.payload = {
        "chunk_id": "doc_defect_BUG-000002_c0",
        "document_type": "defect",
        "defect_id": "BUG-000002",
        "page_content": "Defect BUG-000002: Invoicing calculation error.",
    }
    mock_client.scroll.return_value = ([mock_point], None)

    mock_keyword = MagicMock()
    # Sparse BM25 returns 0 candidates
    mock_keyword.retrieve.return_value = []

    controller = QueryAdaptiveRetrievalController(
        config=RetrievalConfig(),
        client=mock_client,
        keyword_retriever=mock_keyword,
        reranker=None,
    )

    router_output = RouterOutput(
        intent="Check BUG-000002",
        strategy=RetrievalRoute.KEYWORD,
        retrieval_mode=RetrievalMode.DIRECT_ENTITY,
        primary_entity_type="defect",
        primary_entity_id="BUG-000002",
        extracted_identifiers={"defect_id": "BUG-000002"},
        validated_filters={"defect_id": "BUG-000002"},
        reasoning="Defect lookup",
    )

    evidence, assessment, telemetry = controller.execute_adaptive_retrieval(
        query="What is BUG-000002?",
        router_output=router_output,
    )

    assert len(evidence) >= 1
    assert evidence[0].chunk_id == "doc_defect_BUG-000002_c0"
    assert telemetry.get("filter_fallback_applied") is True
    assert telemetry.get("fallback_type") == "exact_metadata_scroll"


# ---------------------------------------------------------------------------
# RAG-12: Schema Filter Justification & Assignee Role Mapping
# ---------------------------------------------------------------------------

def test_router_drops_unjustified_filters_and_remaps_roles():
    """Verify router drops hallucinated priority/status filters and correctly maps defect assignee."""
    from src.retrieval.router import LLMQueryRouter

    router = LLMQueryRouter(config=RetrievalConfig())

    # Case 1: Query asks for priority, does not specify P3 -> P3 must be dropped!
    query = "What is the priority of defect BUG-000001?"
    raw_filters = {"priority": "P3", "status": "In Progress"}
    regex_ids = {"defect_id": "BUG-000001"}
    sanitized = router._validate_and_sanitize_filters(raw_filters, regex_ids, question=query)

    assert "priority" not in sanitized
    assert "status" not in sanitized
    assert sanitized.get("defect_id") == "BUG-000001"

    # Case 2: Query asks for failed test cases -> 'failed' is in query, status should be kept!
    query2 = "List all failed test cases for auth module"
    raw_filters2 = {"status": "Failed", "module": "auth"}
    regex_ids2 = {}
    sanitized2 = router._validate_and_sanitize_filters(raw_filters2, regex_ids2, question=query2)
    assert sanitized2.get("status") == "Failed"
    assert sanitized2.get("module") == "auth"

    # Case 3: Query asks about developer for a defect, but LLM outputs 'tester'
    query3 = "Who is the assigned developer for defect BUG-000002?"
    raw_filters3 = {"tester": "Melissa Fletcher"}  # Name not in query!
    sanitized3 = router._validate_and_sanitize_filters(raw_filters3, {"defect_id": "BUG-000002"}, question=query3)
    assert "tester" not in sanitized3
    assert "assigned_to" not in sanitized3

