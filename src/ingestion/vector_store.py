"""Qdrant vector store connection, schema verification, and index management."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.ingestion.config import PipelineConfig, default_config
from src.ingestion.logging_config import logger


class QdrantManager:
    """Manages collection lifecycle, connection, and indexing in Qdrant."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config
        self.collection_name = self.config.qdrant_collection
        self.vector_dim = self.config.embedding_dimension
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._connect()
        return self._client

    def _connect(self):
        """Connect to Qdrant via URL or local persistent storage."""
        if self.config.qdrant_url:
            logger.info(f"Connecting to remote Qdrant service at {self.config.qdrant_url}")
            self._client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout=30.0,
            )
        else:
            db_path = Path(self.config.qdrant_path)
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using local persistent embedded Qdrant storage at {db_path}")
            self._client = QdrantClient(path=str(db_path))

    def close(self):
        """Close connection and release file locks if local."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def ensure_collection(self) -> bool:
        """Create collection if not exists, or verify configuration compatibility.
        
        Raises:
            ValueError: If vector dimension or distance metric is incompatible.
        """
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            logger.info(
                f"Creating collection '{self.collection_name}' "
                f"(dim={self.vector_dim}, distance=Cosine)"
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE,
                ),
            )
            self._create_payload_indexes()
            return True

        # Collection exists: inspect vector parameters
        info = self.client.get_collection(self.collection_name)
        params = info.config.params.vectors

        # In qdrant-client, params can be VectorParams or dict of VectorParams
        dim = getattr(params, "size", None)
        dist = getattr(params, "distance", None)

        if dim is not None and dim != self.vector_dim:
            raise ValueError(
                f"Collection '{self.collection_name}' exists with incompatible vector size "
                f"{dim} (pipeline expects {self.vector_dim}). Aborting to avoid data corruption."
            )
        if dist is not None and dist != models.Distance.COSINE:
            raise ValueError(
                f"Collection '{self.collection_name}' exists with incompatible distance metric "
                f"{dist} (pipeline expects Cosine). Aborting."
            )

        logger.info(
            f"Collection '{self.collection_name}' verified. Existing points: {info.points_count}"
        )
        self._create_payload_indexes()
        return False

    def _create_payload_indexes(self):
        """Create keyword/text indexes on key filtering fields."""
        fields_to_index = [
            ("document_type", models.PayloadSchemaType.KEYWORD),
            ("test_case_id", models.PayloadSchemaType.KEYWORD),
            ("execution_id", models.PayloadSchemaType.KEYWORD),
            ("defect_id", models.PayloadSchemaType.KEYWORD),
            ("requirement_id", models.PayloadSchemaType.KEYWORD),
            ("module", models.PayloadSchemaType.KEYWORD),
            ("status", models.PayloadSchemaType.KEYWORD),
            ("source", models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in fields_to_index:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception as e:
                # Payload index might already exist
                logger.debug(f"Payload index on '{field_name}': {e}")

    def get_point_count(self) -> int:
        """Get the current point count in the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def reset_collection(self, confirm: bool = False):
        """Reset the collection if explicitly confirmed."""
        if not confirm:
            raise PermissionError("Explicit confirmation required to reset Qdrant collection.")
        logger.warning(f"Resetting collection '{self.collection_name}'")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")
        self.ensure_collection()
