"""Unit tests for embeddings, deduplication, and Qdrant indexing logic."""

import pytest
from src.ingestion.embeddings import MockDeterministicEmbeddings
from src.ingestion.indexer import QdrantIndexer
from src.ingestion.deduplicator import Deduplicator
from langchain_core.documents import Document


def test_mock_embeddings_dimension_and_finite():
    emb = MockDeterministicEmbeddings(dimension=768)
    vec = emb.embed_query("Test query for QA traceability")
    assert len(vec) == 768
    for v in vec:
        assert isinstance(v, float)
        assert -1.0 <= v <= 1.0


def test_stable_uuid_derivation():
    chunk_id = "doc_test_case_TC-000001_c0"
    uuid1 = QdrantIndexer.get_point_uuid(chunk_id)
    uuid2 = QdrantIndexer.get_point_uuid(chunk_id)
    assert uuid1 == uuid2
    assert len(uuid1) == 36  # Standard UUID format


def test_deduplicator_identical_content_vs_different_record():
    dedup = Deduplicator()

    # Two records with identical content but different record IDs should BOTH be kept!
    doc1 = Document(
        page_content="Common error message",
        metadata={"document_type": "failure_log", "record_id": "log_1", "content_hash": "hash_abc"},
    )
    doc2 = Document(
        page_content="Common error message",
        metadata={"document_type": "failure_log", "record_id": "log_2", "content_hash": "hash_abc"},
    )
    # Third record is exact duplicate of doc1
    doc3 = Document(
        page_content="Common error message",
        metadata={"document_type": "failure_log", "record_id": "log_1", "content_hash": "hash_abc"},
    )

    filtered, stats = dedup.filter_documents([doc1, doc2, doc3])
    assert len(filtered) == 2
    assert stats.exact_duplicates_skipped == 1
    recs = [d.metadata["record_id"] for d in filtered]
    assert "log_1" in recs
    assert "log_2" in recs
