"""Unit and integration tests for sparse BM25 embeddings, checkpointing, and Qdrant updates."""

import json
import pytest
from pathlib import Path
from qdrant_client import QdrantClient, models

from src.ingestion.config import PipelineConfig
from src.ingestion.sparse_embeddings import (
    LocalBM25SparseEmbeddings,
    MockBM25SparseEmbeddings,
    SparseVectorResult,
)
from src.ingestion.vector_store import QdrantManager
from scripts.generate_sparse_vectors import CheckpointManager


def test_sparse_vector_result_and_validation():
    # Valid result
    res = SparseVectorResult(indices=[1, 42, 105], values=[0.1, 1.5, 0.88])
    d = res.to_dict()
    assert d["indices"] == [1, 42, 105]
    assert d["values"] == [0.1, 1.5, 0.88]

    # Validation succeeds
    LocalBM25SparseEmbeddings.validate_sparse_vector(res.indices, res.values)

    # Length mismatch
    with pytest.raises(ValueError, match="mismatch"):
        LocalBM25SparseEmbeddings.validate_sparse_vector([1, 2], [0.5])

    # Negative index
    with pytest.raises(ValueError, match="negative"):
        LocalBM25SparseEmbeddings.validate_sparse_vector([-1], [0.5])

    # Non-finite value
    with pytest.raises(ValueError, match="non-finite"):
        LocalBM25SparseEmbeddings.validate_sparse_vector([1], [float("nan")])


def test_mock_bm25_sparse_embeddings():
    embedder = MockBM25SparseEmbeddings(vocab_size=1000)
    docs = [
        "Database connection error during transaction.",
        "User authentication failed due to expired token.",
    ]
    results = embedder.embed_texts(docs)
    assert len(results) == 2

    for r in results:
        assert len(r.indices) > 0
        assert len(r.indices) == len(r.values)
        for idx in r.indices:
            assert isinstance(idx, int)
            assert 0 <= idx < 1000
        for val in r.values:
            assert isinstance(val, float)
            assert val > 0.0

    # Query embedding
    q_res = embedder.embed_query("transaction error")
    assert len(q_res.indices) > 0
    assert len(q_res.indices) == len(q_res.values)


def test_checkpoint_manager_save_and_resume(tmp_path):
    cp_dir = tmp_path / "checkpoints"
    collection = "test_col"
    sparse_name = "bm25"

    mgr = CheckpointManager(cp_dir, collection, sparse_name)
    assert len(mgr.processed_ids) == 0

    batch = [
        {"id": "pt-001", "indices": [10, 20], "values": [0.5, 1.2]},
        {"id": "pt-002", "indices": [30, 40], "values": [0.8, 0.3]},
    ]
    mgr.save_batch(batch, last_offset="offset_1", total_in_collection=10)

    assert mgr.is_processed("pt-001") is True
    assert mgr.is_processed("pt-002") is True
    assert mgr.is_processed("pt-003") is False

    # Simulate resuming in a new instance
    resumed_mgr = CheckpointManager(cp_dir, collection, sparse_name)
    assert len(resumed_mgr.processed_ids) == 2
    assert resumed_mgr.is_processed("pt-001") is True
    assert resumed_mgr.is_processed("pt-002") is True
    assert resumed_mgr.is_processed("pt-003") is False


def test_qdrant_sparse_schema_migration_and_dense_preservation(tmp_path):
    """Verify that adding a sparse vector preserves existing dense vectors, point IDs, and payloads."""
    db_path = tmp_path / "test_qdrant_db"
    config = PipelineConfig(
        qdrant_url=None,
        qdrant_path=str(db_path),
        qdrant_collection="test_migration",
        embedding_dimension=4,
    )

    manager = QdrantManager(config)
    client = manager.client

    # 1. Create collection with unnamed dense vector
    client.create_collection(
        collection_name="test_migration",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
    )

    # 2. Insert point with dense vector and payload
    dense_original = [0.1, 0.2, 0.3, 0.4]
    client.upsert(
        collection_name="test_migration",
        points=[
            models.PointStruct(
                id=101,
                vector=dense_original,
                payload={"chunk_id": "chunk_tc_101", "text": "Test sample text for QA."},
            )
        ],
    )

    # 3. Safely ensure sparse vector schema exists
    added = manager.ensure_sparse_vector("bm25")
    assert added is True

    # Idempotent re-check
    added_again = manager.ensure_sparse_vector("bm25")
    assert added_again is False

    # 4. Update point with sparse vector
    sparse_indices = [5, 42]
    sparse_values = [0.75, 1.25]
    manager.update_point_sparse_vectors(
        updates=[(101, sparse_indices, sparse_values)],
        vector_name="bm25",
    )

    # 5. Retrieve point and verify 100% preservation
    points = client.retrieve(
        collection_name="test_migration",
        ids=[101],
        with_payload=True,
        with_vectors=True,
    )
    assert len(points) == 1
    pt = points[0]

    # Verify ID
    assert pt.id == 101

    # Verify payload intact
    assert pt.payload["chunk_id"] == "chunk_tc_101"
    assert pt.payload["text"] == "Test sample text for QA."

    # Verify dense vector preserved (Cosine normalization applied by Qdrant or exact floats)
    assert isinstance(pt.vector, dict)
    dense_retrieved = pt.vector.get("")
    assert dense_retrieved is not None
    assert len(dense_retrieved) == 4

    # Verify sparse vector stored
    sparse_retrieved = pt.vector.get("bm25")
    assert sparse_retrieved is not None
    assert sparse_retrieved.indices == sparse_indices
    assert [round(v, 2) for v in sparse_retrieved.values] == [0.75, 1.25]

    manager.close()
