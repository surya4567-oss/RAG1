"""Keyword and sparse BM25 retrieval using FastEmbed and Qdrant sparse vectors."""

from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.ingestion.logging_config import logger
from src.ingestion.sparse_embeddings import LocalBM25SparseEmbeddings
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.schemas import CandidateChunk


class KeywordRetriever:
    """Keyword / sparse retriever using FastEmbed BM25 and Qdrant."""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        client: Optional[QdrantClient] = None,
        sparse_embedder: Optional[LocalBM25SparseEmbeddings] = None,
    ):
        self.config = config or default_retrieval_config
        self._client = client
        self._sparse_embedder = sparse_embedder

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            if self.config.qdrant_url:
                self._client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key,
                )
            else:
                self._client = QdrantClient(path=self.config.qdrant_path)
        return self._client

    def _get_sparse_embedder(self) -> LocalBM25SparseEmbeddings:
        if self._sparse_embedder is None:
            self._sparse_embedder = LocalBM25SparseEmbeddings(
                model_name=self.config.sparse_model_name,
            )
        return self._sparse_embedder

    def _build_filter(self, metadata_filters: Optional[Dict[str, Any]]) -> Optional[models.Filter]:
        """Convert validated dictionary filters into Qdrant Filter models."""
        if not metadata_filters:
            return None

        conditions = []
        for key, value in metadata_filters.items():
            if value is None or key not in self.config.allowed_filter_fields:
                continue
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )

        if not conditions:
            return None
        return models.Filter(must=conditions)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateChunk]:
        """Execute BM25 sparse vector retrieval against Qdrant."""
        limit = top_k or self.config.sparse_top_k
        if not query.strip():
            return []

        embedder = self._get_sparse_embedder()
        client = self._get_client()

        # Compute BM25 sparse vector for the query text
        sparse_res = embedder.embed_query(query)
        if not sparse_res.indices:
            logger.warning(f"Sparse embedder returned 0 tokens for query: '{query}'")
            return []

        qdrant_sparse_vector = models.SparseVector(
            indices=sparse_res.indices,
            values=sparse_res.values,
        )

        qdrant_filter = self._build_filter(filters)

        # Check target collection (prefer dedicated sparse collection if configured)
        target_collection = self.config.sparse_collection

        try:
            response = client.query_points(
                collection_name=target_collection,
                query=qdrant_sparse_vector,
                using=self.config.sparse_vector_name,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False,
            )
            points = response.points if hasattr(response, "points") else response
        except Exception as e:
            # Fallback to search API with named sparse vector
            try:
                points = client.search(
                    collection_name=target_collection,
                    query_vector=models.NamedSparseVector(
                        name=self.config.sparse_vector_name,
                        vector=qdrant_sparse_vector,
                    ),
                    limit=limit,
                    query_filter=qdrant_filter,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as search_err:
                logger.error(
                    f"Sparse retrieval failed on collection '{target_collection}' "
                    f"with vector '{self.config.sparse_vector_name}': {search_err}"
                )
                raise search_err from e

        candidates: List[CandidateChunk] = []
        for rank, pt in enumerate(points, start=1):
            payload = pt.payload or {}
            chunk_id = str(payload.get("chunk_id") or pt.id)
            text = str(payload.get("text") or payload.get("page_content") or "")

            candidates.append(
                CandidateChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(pt.score) if pt.score is not None else 0.0,
                    original_rank=rank,
                    retrieval_route="keyword",
                    metadata=dict(payload),
                    dense_rank=None,
                    sparse_rank=rank,
                    rrf_score=None,
                    rerank_score=None,
                )
            )

        logger.info(f"Keyword retrieval returned {len(candidates)} candidates for query '{query[:40]}...'")
        return candidates
