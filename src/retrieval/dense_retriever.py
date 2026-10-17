"""Dense semantic retrieval using local BGE embeddings and Qdrant vector search."""

from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.ingestion.embeddings import LocalBGEEmbeddings
from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.schemas import CandidateChunk


class DenseRetriever:
    """Semantic vector retriever using BAAI/bge-base-en-v1.5 and Qdrant."""

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        client: Optional[QdrantClient] = None,
        embedder: Optional[LocalBGEEmbeddings] = None,
    ):
        self.config = config or default_retrieval_config
        self._client = client
        self._embedder = embedder

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

    def _get_embedder(self) -> LocalBGEEmbeddings:
        if self._embedder is None:
            self._embedder = LocalBGEEmbeddings()
        return self._embedder

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
        """Execute dense vector similarity retrieval against Qdrant."""
        limit = top_k or self.config.dense_top_k
        if not query.strip():
            return []

        embedder = self._get_embedder()
        client = self._get_client()

        # Embed query text
        query_vector = embedder._model.embed_query(query) if hasattr(embedder, "_model") and embedder._model else None
        if query_vector is None:
            embedder._load_model()
            query_vector = embedder._model.embed_query(query)

        qdrant_filter = self._build_filter(filters)

        try:
            # Query Qdrant dense collection
            response = client.query_points(
                collection_name=self.config.dense_collection,
                query=query_vector,
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False,
            )
            points = response.points if hasattr(response, "points") else response
        except Exception as e:
            # Fallback to search API if query_points signature differs
            try:
                points = client.search(
                    collection_name=self.config.dense_collection,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=qdrant_filter,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as search_err:
                logger.error(f"Dense retrieval failed on collection '{self.config.dense_collection}': {search_err}")
                raise search_err from e

        candidates: List[CandidateChunk] = []
        for rank, pt in enumerate(points, start=1):
            payload = pt.payload or {}
            chunk_id = str(payload.get("chunk_id") or pt.id)
            text = str(payload.get("page_content") or payload.get("text") or "")

            candidates.append(
                CandidateChunk(
                    chunk_id=chunk_id,
                    text=text,
                    score=float(pt.score) if pt.score is not None else 0.0,
                    original_rank=rank,
                    retrieval_route="dense",
                    metadata=dict(payload),
                    dense_rank=rank,
                    sparse_rank=None,
                    rrf_score=None,
                    rerank_score=None,
                )
            )

        logger.info(f"Dense retrieval returned {len(candidates)} candidates for query '{query[:40]}...'")
        return candidates
