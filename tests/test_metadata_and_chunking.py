"""Unit tests for metadata schema, document builder, and chunking."""

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import SourceAwareChunker
from src.ingestion.config import PipelineConfig
from src.ingestion.document_builder import DocumentBuilder
from src.ingestion.metadata_enricher import MetadataEnricher
from src.ingestion.metadata_schema import sanitize_payload


def test_sanitize_payload_no_literal_none():
    raw = {
        "valid_str": "hello",
        "none_val": None,
        "literal_none": "None",
        "literal_null": "null",
        "literal_nan": "NaN",
        "empty_str": "",
        "valid_num": 42,
        "valid_bool": True,
        "list_with_nan": ["valid", "None", "good"],
    }
    clean = sanitize_payload(raw)
    assert "literal_none" not in clean
    assert "literal_null" not in clean
    assert "literal_nan" not in clean
    assert "none_val" not in clean
    assert clean["valid_str"] == "hello"
    assert clean["valid_num"] == 42
    assert clean["valid_bool"] is True
    assert clean["list_with_nan"] == ["valid", "good"]


def test_document_builder_and_enrichment():
    config = PipelineConfig()
    enricher = MetadataEnricher(config)

    raw_tc = {
        "test_case_id": "TC-000123",
        "requirement_id": "REQ-456",
        "feature": "User Checkout",
        "module": "eCommerce",
        "priority": "High",
        "severity": "Critical",
        "status": "Review",
        "precondition": "Cart has items",
        "test_steps": "Click checkout",
        "expected_result": "Payment screen opens",
    }
    doc = DocumentBuilder.build_test_case_doc(raw_tc, source_file="testcases.csv")
    assert "TC-000123" in doc.page_content
    assert doc.metadata["document_type"] == "test_case"

    enriched = enricher.enrich_document(doc)
    assert enriched.metadata["document_id"] == "doc_test_case_TC-000123"
    assert enriched.metadata["test_case_id"] == "TC-000123"
    assert enriched.metadata["requirement_id"] == "REQ-456"
    assert "content_hash" in enriched.metadata
    assert enriched.metadata["is_cross_consistent"] is True


def test_chunker_small_and_oversized():
    config = PipelineConfig(chunk_size=100, chunk_overlap=20)
    chunker = SourceAwareChunker(config)

    # Small doc
    small_doc = Document(
        page_content="Short text within chunk limit.",
        metadata={"document_id": "doc_001", "document_type": "test_case", "record_id": "TC-1"},
    )
    small_chunks = chunker.chunk_single_document(small_doc)
    assert len(small_chunks) == 1
    assert small_chunks[0].metadata["chunk_id"] == "doc_001_c0"
    assert small_chunks[0].metadata["parent_document_id"] == "doc_001"

    # Oversized doc
    long_text = "This is a sentence. " * 20  # ~400 chars > 100
    long_doc = Document(
        page_content=long_text,
        metadata={"document_id": "doc_002", "document_type": "defect", "record_id": "BUG-2"},
    )
    long_chunks = chunker.chunk_single_document(long_doc)
    assert len(long_chunks) > 1
    for idx, ch in enumerate(long_chunks):
        assert ch.metadata["parent_document_id"] == "doc_002"
        assert ch.metadata["chunk_index"] == idx
        assert ch.metadata["chunk_id"] == f"doc_002_c{idx}"
