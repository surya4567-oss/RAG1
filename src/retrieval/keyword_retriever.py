
"""Keyword and sparse BM25 retrieval using FastEmbed and Qdrant."""

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.ingestion.logging_config import logger
from src.ingestion.sparse_embeddings import LocalBM25SparseEmbeddings
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.schemas import CandidateChunk


class KeywordRetriever:
    """Generic sparse BM25 retriever for Odoo ERP QA documents."""

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
        """Create or reuse the configured Qdrant client."""

        if self._client is None:
            if self.config.qdrant_url:
                self._client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key,
                )
            else:
                self._client = QdrantClient(
                    path=self.config.qdrant_path
                )

        return self._client

    def _get_sparse_embedder(self) -> LocalBM25SparseEmbeddings:
        """Create or reuse the local BM25 sparse embedder."""

        if self._sparse_embedder is None:
            self._sparse_embedder = LocalBM25SparseEmbeddings(
                model_name=self.config.sparse_model_name
            )

        return self._sparse_embedder

    def _build_filter(
        self,
        metadata_filters: Optional[Dict[str, Any]],
    ) -> Optional[models.Filter]:
        """Build a Qdrant filter from allowed metadata fields."""

        if not metadata_filters:
            return None

        allowed_fields = set(self.config.allowed_filter_fields)
        conditions = []

        for key, value in metadata_filters.items():

            if key not in allowed_fields or value is None:
                continue

            if isinstance(value, str) and not value.strip():
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

    def _check_collection(self, client: QdrantClient) -> None:
        """Verify sparse collection and named vector configuration."""
        if getattr(self, "_collection_verified", False):
            return

        collections = [
            collection.name
            for collection in client.get_collections().collections
        ]

        collection_name = self.config.sparse_collection

        if collection_name not in collections:
            raise ValueError(
                f"Sparse collection '{collection_name}' not found. "
                f"Available collections: {collections}. "
                f"Check the Qdrant storage path or run ingestion."
            )

        collection_info = client.get_collection(
            collection_name=collection_name
        )

        sparse_vectors = collection_info.config.params.sparse_vectors or {}

        vector_name = self.config.sparse_vector_name

        if vector_name not in sparse_vectors:
            raise ValueError(
                f"Sparse vector '{vector_name}' not found in "
                f"collection '{collection_name}'. "
                f"Available sparse vectors: {list(sparse_vectors.keys())}"
            )

        self._collection_verified = True

    def _search(
        self,
        client: QdrantClient,
        query_vector: models.SparseVector,
        limit: int,
        query_filter: Optional[models.Filter],
    ) -> List[Any]:
        """Execute sparse-vector search using the current Qdrant API."""

        response = client.query_points(
            collection_name=self.config.sparse_collection,
            query=query_vector,
            using=self.config.sparse_vector_name,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
        )

        return response.points

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[CandidateChunk]:
        """
        Retrieve relevant QA chunks using sparse BM25 search.

        Performs genuine sparse-vector retrieval.
        Fallback and evidence expansion are managed by the
        adaptive retrieval controller.
        """

        query = (query or "").strip()

        if not query:
            logger.warning(
                "Empty query supplied to keyword retriever."
            )
            return []

        limit = top_k or self.config.sparse_top_k

        client = self._get_client()
        embedder = self._get_sparse_embedder()

        try:
            # Verify collection before executing retrieval.
            self._check_collection(client)

            # Generate sparse query embedding.
            sparse_result = embedder.embed_query(query)

            if not sparse_result.indices:
                logger.warning(
                    "Sparse embedder returned no tokens "
                    "for query: %r",
                    query,
                )
                return []

            sparse_vector = models.SparseVector(
                indices=sparse_result.indices,
                values=sparse_result.values,
            )

            # Build metadata filter.
            query_filter = self._build_filter(filters)

            logger.info(
                "Starting sparse retrieval: collection=%s, "
                "vector=%s, top_k=%s, filters=%s",
                self.config.sparse_collection,
                self.config.sparse_vector_name,
                limit,
                filters,
            )

            # Execute sparse retrieval.
            points = self._search(
                client=client,
                query_vector=sparse_vector,
                limit=limit,
                query_filter=query_filter,
            )

        except Exception:
            logger.exception(
                "Sparse retrieval failed for collection '%s' "
                "using vector '%s'.",
                self.config.sparse_collection,
                self.config.sparse_vector_name,
            )
            return []

        # Convert Qdrant results into CandidateChunk objects.
        candidates: List[CandidateChunk] = []

        for rank, point in enumerate(points, start=1):

            payload = point.payload or {}

            chunk_id = str(
                payload.get("chunk_id")
                or payload.get("document_id")
                or payload.get("record_id")
                or point.id
            )

            text_content = (
                payload.get("page_content")
                or payload.get("text")
                or ""
            )

            if not isinstance(text_content, str):
                text_content = str(text_content)

            point_score = getattr(point, "score", None)

            score = (
                float(point_score)
                if point_score is not None
                else 0.0
            )

            candidate = CandidateChunk(
                chunk_id=chunk_id,
                text=text_content,
                score=score,
                original_rank=rank,
                retrieval_route="keyword",
                metadata=dict(payload),
                dense_rank=None,
                sparse_rank=rank,
                rrf_score=None,
                rerank_score=None,
            )

            candidates.append(candidate)

        logger.info(
            "Keyword retrieval returned %d candidates "
            "from %s for query: %r",
            len(candidates),
            self.config.sparse_collection,
            query[:100],
        )

        return candidates
