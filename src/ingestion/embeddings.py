
"""Local BGE embedding generator with vector validation and batching."""

import gc
import math
from typing import List, Optional

from src.ingestion.config import PipelineConfig, default_config
from src.ingestion.logging_config import logger


class LocalBGEEmbeddings:
    """Local BGE embeddings with CUDA support and configurable batching."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config
        self.model_name = self.config.embedding_model
        self.device = self.config.embedding_device
        self.batch_size = max(1, self.config.embedding_batch_size)
        self.expected_dim = self.config.embedding_dimension
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from sentence_transformers import SentenceTransformer

        # Avoid silently falling back to CPU when CUDA was requested.
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was configured but is unavailable. "
                "Enable a GPU runtime in Google Colab."
            )

        logger.info(
            f"Loading embedding model '{self.model_name}' "
            f"on device '{self.device}', batch size {self.batch_size}"
        )

        st = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
        st.max_seq_length = 512

        class _STWrapper:
            def __init__(self, model, batch_size):
                self.st = model
                self.batch_size = batch_size

            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                if not texts:
                    return []

                with torch.no_grad():
                    vectors = self.st.encode(
                        texts,
                        normalize_embeddings=True,
                        batch_size=min(len(texts), self.batch_size),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )

                return vectors.tolist()

            def embed_query(self, text: str) -> List[float]:
                with torch.no_grad():
                    vector = self.st.encode(
                        text,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )

                return vector.tolist()

        self._model = _STWrapper(st, self.batch_size)

        logger.info(
            f"SentenceTransformer loaded successfully on "
            f"{self.device}; max sequence length=512"
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts in batches and validate each vector."""
        if not texts:
            return []

        self._load_model()

        all_embeddings: List[List[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]

            batch_embeddings = self._model.embed_documents(batch)

            for vector in batch_embeddings:
                self.validate_vector(vector)
                all_embeddings.append(vector)

            del batch_embeddings

            if start % (self.batch_size * 8) == 0:
                gc.collect()

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Generate and validate a single query embedding."""
        self._load_model()

        vector = self._model.embed_query(query)
        self.validate_vector(vector)

        return vector

    def validate_vector(self, vector: List[float]):
        """Validate vector dimensions and numerical values."""
        if len(vector) != self.expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected "
                f"{self.expected_dim}, got {len(vector)}"
            )

        if not all(math.isfinite(value) for value in vector):
            raise ValueError(
                "Embedding contains a non-finite numerical value (NaN or Inf)"
            )


class MockDeterministicEmbeddings:
    """Deterministic pseudo-embeddings for testing only."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw_values = [(byte / 255.0) * 2.0 - 1.0 for byte in digest]

        repeated = (
            raw_values * (self.dimension // len(raw_values) + 1)
        )[:self.dimension]

        norm = math.sqrt(sum(value * value for value in repeated)) or 1.0

        return [value / norm for value in repeated]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)
