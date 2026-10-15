"""Integration tests for persistent local Qdrant indexing, retrieval, and re-ingestion."""

import pytest
import shutil
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion.config import PipelineConfig
from src.ingestion.embeddings import MockDeterministicEmbeddings
from src.ingestion.indexer import QdrantIndexer
from src.ingestion.vector_store import QdrantManager


@pytest.fixture
def temp_qdrant_env(tmp_path):
    db_path = tmp_path / "test_qdrant_db"
    config = PipelineConfig(
        data_dir=tmp_path,
        processed_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
        qdrant_url=None,  # Use local embedded
        qdrant_path=str(db_path),
        qdrant_collection="test_qa_knowledge",
        embedding_dimension=768,
        index_batch_size=4,
    )
    embeddings = MockDeterministicEmbeddings(dimension=768)
    vector_store = QdrantManager(config)
    indexer = QdrantIndexer(config=config, vector_store=vector_store, embeddings=embeddings)

    yield config, vector_store, indexer

    # Cleanup
    vector_store.close()
    if db_path.exists():
        shutil.rmtree(db_path, ignore_errors=True)


def test_qdrant_lifecycle_and_indexing(temp_qdrant_env):
    config, vector_store, indexer = temp_qdrant_env

    # 1. Ensure collection creation
    created = vector_store.ensure_collection()
    assert created is True
    assert vector_store.get_point_count() == 0

    # 2. Test incompatibility check: wrong dimension must raise ValueError
    # Close vector_store first to release local file lock
    vector_store.close()

    bad_config = PipelineConfig(
        qdrant_url=None,
        qdrant_path=config.qdrant_path,
        qdrant_collection="test_qa_knowledge",
        embedding_dimension=384,  # mismatched
    )
    bad_store = QdrantManager(bad_config)
    with pytest.raises(ValueError, match="incompatible vector size"):
        bad_store.ensure_collection()
    bad_store.close()

    # Re-open original store and indexer
    vector_store = QdrantManager(config)
    vector_store.ensure_collection()
    indexer.vector_store = vector_store

    # 3. Create sample chunks
    chunks = [
        Document(
            page_content="Database connection timeout on port 5432 during invoice processing.",
            metadata={
                "chunk_id": "chunk_db_timeout_001",
                "document_id": "doc_log_001",
                "document_type": "failure_log",
                "record_id": "BUG-000100",
                "defect_id": "BUG-000100",
                "source": "logs_001.txt",
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="User permissions verification failed with AccessError in module HR.",
            metadata={
                "chunk_id": "chunk_access_denied_002",
                "document_id": "doc_tc_002",
                "document_type": "test_case",
                "record_id": "TC-000200",
                "test_case_id": "TC-000200",
                "source": "testcases.csv",
                "chunk_index": 0,
            },
        ),
    ]

    # 4. Index chunks
    result = indexer.index_chunks(chunks)
    assert result.points_inserted == 2
    assert result.points_failed == 0
    assert result.collection_final_count == 2
    assert result.smoke_test_passed is True

    # 5. Idempotent re-indexing (same chunks)
    re_result = indexer.index_chunks(chunks)
    assert re_result.points_inserted == 2
    # Collection point count should remain 2 (upserted in-place via deterministic UUIDs)
    assert vector_store.get_point_count() == 2

    # 6. Source-specific stale point removal
    indexer.remove_stale_source_points("logs_001.txt")
    # Remaining point should be the test_case from testcases.csv
    assert vector_store.get_point_count() == 1
