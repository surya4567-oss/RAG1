"""Local BM25 sparse embedding generator with FastEmbed integration and validation."""

import gc
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from src.ingestion.config import PipelineConfig, default_config
from src.ingestion.logging_config import logger


@dataclass
class SparseVectorResult:
    """Represents a sparse vector with parallel token indices and numerical weights."""
    indices: List[int]
    values: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dictionary format suitable for JSON serialization."""
        return {
            "indices": self.indices,
            "values": self.values,
        }

    def to_qdrant_model(self):
        """Convert to Qdrant SparseVector model if qdrant-client is available."""
        from qdrant_client.http import models
        return models.SparseVector(
            indices=self.indices,
            values=self.values,
        )


class LocalBM25SparseEmbeddings:
    """Local BM25 sparse embedding generator using FastEmbed.
    
    Generates sparse vector representations (token indices and weights)
    compatible with Qdrant sparse vector search.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        threads: Optional[int] = None,
    ):
        self.config = config or default_config
        self.model_name = model_name or self.config.sparse_model_name
        self.batch_size = max(1, batch_size or self.config.sparse_batch_size)
        self.threads = threads
        self._model = None

    def _load_model(self):
        """Lazy load the FastEmbed SparseTextEmbedding model."""
        if self._model is not None:
            return

        try:
            from fastembed import SparseTextEmbedding
        except ImportError as e:
            raise ImportError(
                "fastembed is required for sparse BM25 embedding generation. "
                "Install it using: pip install fastembed>=0.4.0"
            ) from e

        logger.info(
            f"Loading sparse embedding model '{self.model_name}' "
            f"with batch_size={self.batch_size}, threads={self.threads}"
        )

        self._model = SparseTextEmbedding(
            model_name=self.model_name,
            threads=self.threads,
        )

        logger.info(f"FastEmbed sparse model '{self.model_name}' loaded successfully.")

    def embed_texts(
        self,
        texts: Sequence[str],
        show_progress: bool = False,
    ) -> List[SparseVectorResult]:
        """Generate sparse embeddings for a sequence of document texts in batches.
        
        Args:
            texts: List of document strings to embed.
            show_progress: Whether to log intermediate batch progress.
            
        Returns:
            List of SparseVectorResult containing validated indices and values.
        """
        if not texts:
            return []

        self._load_model()
        results: List[SparseVectorResult] = []
        total_texts = len(texts)

        # Process in batches to control memory usage and facilitate cleanup
        for start_idx in range(0, total_texts, self.batch_size):
            batch_texts = list(texts[start_idx:start_idx + self.batch_size])
            batch_embeddings = list(
                self._model.embed(
                    batch_texts,
                    batch_size=min(len(batch_texts), self.batch_size),
                )
            )

            for raw_emb in batch_embeddings:
                # FastEmbed returns an object with .indices and .values (numpy arrays)
                raw_indices = raw_emb.indices
                raw_values = raw_emb.values

                indices = [int(idx) for idx in raw_indices]
                values = [float(val) for val in raw_values]

                self.validate_sparse_vector(indices, values)
                results.append(SparseVectorResult(indices=indices, values=values))

            del batch_embeddings

            if (start_idx // self.batch_size) % 10 == 0:
                gc.collect()

        return results

    def embed_query(self, query: str) -> SparseVectorResult:
        """Generate a sparse embedding for a search query.
        
        Args:
            query: The search query string.
            
        Returns:
            SparseVectorResult with query token indices and weights.
        """
        self._load_model()
        # FastEmbed uses query_embed for search queries
        query_embeddings = list(self._model.query_embed([query]))
        if not query_embeddings:
            return SparseVectorResult(indices=[], values=[])

        raw_emb = query_embeddings[0]
        indices = [int(idx) for idx in raw_emb.indices]
        values = [float(val) for val in raw_emb.values]

        self.validate_sparse_vector(indices, values)
        return SparseVectorResult(indices=indices, values=values)

    @staticmethod
    def validate_sparse_vector(indices: List[int], values: List[float]):
        """Validate indices and values for sparse vector consistency.
        
        Raises:
            ValueError: If lengths mismatch, indices are negative, or values non-finite.
        """
        if len(indices) != len(values):
            raise ValueError(
                f"Sparse vector mismatch: {len(indices)} indices vs {len(values)} values."
            )

        for idx in indices:
            if idx < 0:
                raise ValueError(f"Sparse index cannot be negative: {idx}")

        for val in values:
            if not math.isfinite(val):
                raise ValueError(
                    f"Sparse vector weight contains a non-finite value (NaN or Inf): {val}"
                )


class MockBM25SparseEmbeddings:
    """Deterministic sparse pseudo-embeddings for fast unit testing without model downloads."""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size

    def _tokenize_to_sparse(self, text: str) -> SparseVectorResult:
        # Tokenize by alphanumeric words
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return SparseVectorResult(indices=[1], values=[1.0])

        counts: Dict[int, int] = {}
        for token in tokens:
            # Deterministic hash to token ID
            import hashlib
            token_id = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % self.vocab_size
            counts[token_id] = counts.get(token_id, 0) + 1

        sorted_items = sorted(counts.items(), key=lambda x: x[0])
        indices = [item[0] for item in sorted_items]
        # Log-tf weighting
        values = [round(1.0 + math.log(item[1]), 4) for item in sorted_items]

        return SparseVectorResult(indices=indices, values=values)

    def embed_texts(self, texts: Sequence[str], **kwargs) -> List[SparseVectorResult]:
        return [self._tokenize_to_sparse(t) for t in texts]

    def embed_query(self, query: str) -> SparseVectorResult:
        return self._tokenize_to_sparse(query)
