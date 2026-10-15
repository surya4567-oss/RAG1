
"""Batch vector indexer with timing, progress reporting, and manifest tracking."""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from qdrant_client.http import models

from src.ingestion.config import PipelineConfig, default_config
from src.ingestion.embeddings import LocalBGEEmbeddings
from src.ingestion.logging_config import logger
from src.ingestion.vector_store import QdrantManager


@dataclass
class IndexingResult:
    total_chunks: int = 0
    points_inserted: int = 0
    points_updated: int = 0
    points_failed: int = 0
    points_skipped: int = 0
    collection_final_count: int = 0
    smoke_test_passed: bool = False
    smoke_test_details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class QdrantIndexer:
    """Index chunks using deterministic IDs, with timing and progress logs."""

    UUID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        vector_store: Optional[QdrantManager] = None,
        embeddings: Optional[Any] = None,
    ):
        self.config = config or default_config
        self.vector_store = vector_store or QdrantManager(self.config)
        self.embeddings = embeddings or LocalBGEEmbeddings(self.config)
        self.manifest_path = self.config.processed_dir / "ingestion_manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                if not isinstance(manifest, dict):
                    raise ValueError("Manifest root must be a JSON object")
                manifest.setdefault("sources", {})
                return manifest
            except Exception as e:
                logger.warning(f"Could not load manifest: {e}")
        return {"sources": {}}

    def _save_manifest(self):
        """Save manifest atomically to reduce corruption risk."""
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.manifest_path.with_suffix(".tmp")

            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._manifest, f, indent=2)
                f.flush()

            temp_path.replace(self.manifest_path)
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    @classmethod
    def get_point_uuid(cls, chunk_id: str) -> str:
        """Derive a reproducible Qdrant ID from a stable chunk ID."""
        return str(uuid.uuid5(cls.UUID_NAMESPACE, chunk_id))

    def index_chunks(self, chunks: List[Document]) -> IndexingResult:
        """Embed and upsert batches while reporting elapsed and estimated time."""
        result = IndexingResult(total_chunks=len(chunks))
        overall_start = time.perf_counter()

        if not chunks:
            logger.warning("No chunks to index.")
            result.collection_final_count = self.vector_store.get_point_count()
            return result

        self.vector_store.ensure_collection()

        # This is the number of chunks sent to index_chunks' embedding call
        # per batch. The embedder may subdivide these further.
        batch_size = max(1, self.config.index_batch_size)
        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size

        logger.info("=" * 70)
        logger.info(f"Indexing started: {total_chunks:,} chunks")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Total batches: {total_batches:,}")
        logger.info(f"Embedding device: {self.config.embedding_device}")
        logger.info(f"Embedding batch size: {self.config.embedding_batch_size}")
        logger.info("=" * 70)

        source_chunk_map: Dict[str, List[str]] = {}
        processed_chunks = 0
        successful_chunks = 0
        failed_chunks = 0
        last_log_time = overall_start

        try:
            for batch_number, start in enumerate(
                range(0, total_chunks, batch_size), start=1
            ):
                batch_start = time.perf_counter()
                batch = chunks[start:start + batch_size]
                texts = [doc.page_content for doc in batch]

                # Stable IDs are essential for safe repeated ingestion.
                chunk_ids = []
                for doc in batch:
                    chunk_id = doc.metadata.get("chunk_id")
                    if not chunk_id:
                        raise ValueError(
                            "A chunk is missing metadata['chunk_id']; "
                            "refusing to generate a random ID because it "
                            "would not be reproducible across runs."
                        )
                    chunk_ids.append(str(chunk_id))

                try:
                    vectors = self.embeddings.embed_texts(texts)

                    if len(vectors) != len(batch):
                        raise ValueError(
                            f"Embedding count mismatch: expected {len(batch)}, "
                            f"received {len(vectors)}"
                        )

                    points: List[models.PointStruct] = []

                    for doc, chunk_id, vector in zip(batch, chunk_ids, vectors):
                        point_id = self.get_point_uuid(chunk_id)
                        source = str(doc.metadata.get("source", "unknown"))
                        source_chunk_map.setdefault(source, []).append(point_id)

                        payload = dict(doc.metadata)
                        payload["chunk_id"] = chunk_id
                        payload["page_content"] = doc.page_content
                        payload["text"] = doc.page_content

                        points.append(
                            models.PointStruct(
                                id=point_id,
                                vector=vector,
                                payload=payload,
                            )
                        )

                    self.vector_store.client.upsert(
                        collection_name=self.vector_store.collection_name,
                        points=points,
                        wait=True,
                    )

                    successful_chunks += len(points)
                    result.points_inserted += len(points)

                except Exception as e:
                    failed_chunks += len(batch)
                    result.points_failed += len(batch)
                    result.errors.append(f"Batch {batch_number}: {e}")
                    logger.exception(
                        f"Batch {batch_number}/{total_batches} failed"
                    )

                processed_chunks += len(batch)
                batch_elapsed = time.perf_counter() - batch_start
                elapsed = time.perf_counter() - overall_start

                rate = processed_chunks / elapsed if elapsed > 0 else 0.0
                remaining = max(0, total_chunks - processed_chunks)
                eta_seconds = remaining / rate if rate > 0 else 0.0

                # Log every batch so progress is visible during long runs.
                logger.info(
                    f"Batch {batch_number:,}/{total_batches:,} | "
                    f"Processed {processed_chunks:,}/{total_chunks:,} "
                    f"({processed_chunks / total_chunks * 100:.1f}%) | "
                    f"Successful: {successful_chunks:,} | "
                    f"Failed: {failed_chunks:,} | "
                    f"Batch time: {batch_elapsed:.1f}s | "
                    f"Rate: {rate:.1f} chunks/s | "
                    f"Elapsed: {elapsed / 60:.1f} min | "
                    f"ETA: {eta_seconds / 60:.1f} min"
                )

                # Persist progress after every successful upsert batch.
                # This records IDs, but does not by itself implement resume.
                for source, point_ids in source_chunk_map.items():
                    existing = self._manifest["sources"].get(source, {})
                    known_ids = set(existing.get("point_ids", []))
                    known_ids.update(point_ids)

                    self._manifest["sources"][source] = {
                        "point_ids": sorted(known_ids),
                        "point_count": len(known_ids),
                        "timestamp": (
                            str(Path(source).stat().st_mtime)
                            if Path(source).exists()
                            else None
                        ),
                    }

                self._save_manifest()
                source_chunk_map.clear()
                last_log_time = time.perf_counter()

        except KeyboardInterrupt:
            elapsed = time.perf_counter() - overall_start
            logger.warning("Indexing interrupted by user.")
            logger.warning(
                f"Progress before interruption: {processed_chunks:,}/"
                f"{total_chunks:,} chunks ({processed_chunks / total_chunks * 100:.1f}%)"
            )
            logger.warning(f"Elapsed: {elapsed / 60:.2f} minutes")
            raise

        total_elapsed = time.perf_counter() - overall_start

        result.collection_final_count = self.vector_store.get_point_count()

        logger.info("=" * 70)
        logger.info("INDEXING TIMING SUMMARY")
        logger.info(f"Total chunks: {total_chunks:,}")
        logger.info(f"Processed: {processed_chunks:,}")
        logger.info(f"Successful: {successful_chunks:,}")
        logger.info(f"Failed: {failed_chunks:,}")
        logger.info(f"Elapsed: {total_elapsed / 60:.2f} minutes")
        logger.info(f"Average throughput: {processed_chunks / total_elapsed:.2f} chunks/s")
        logger.info(f"Collection points: {result.collection_final_count:,}")
        logger.info("=" * 70)

        if result.points_failed == 0:
            result.smoke_test_passed, result.smoke_test_details = (
                self.run_smoke_test()
            )
        else:
            logger.warning(
                "Skipping smoke test because one or more indexing batches failed."
            )

        return result

    def run_smoke_test(
        self, query: str = "permission error access denied"
    ) -> Tuple[bool, Dict[str, Any]]:
        """Verify vector retrieval on the populated collection."""
        try:
            logger.info(f"Running retrieval smoke-test with query: '{query}'")
            query_vector = self.embeddings.embed_query(query)

            if hasattr(self.vector_store.client, "query_points"):
                response = self.vector_store.client.query_points(
                    collection_name=self.vector_store.collection_name,
                    query=query_vector,
                    limit=3,
                    with_payload=True,
                )
                search_result = response.points
            elif hasattr(self.vector_store.client, "search"):
                search_result = self.vector_store.client.search(
                    collection_name=self.vector_store.collection_name,
                    query_vector=query_vector,
                    limit=3,
                    with_payload=True,
                )
            else:
                search_result = []

            results_summary = []
            for hit in search_result:
                payload = hit.payload or {}
                results_summary.append({
                    "id": hit.id,
                    "score": round(hit.score, 4),
                    "document_type": payload.get("document_type"),
                    "record_id": payload.get("record_id"),
                    "source": payload.get("source"),
                    "snippet": (payload.get("page_content") or "")[:150],
                })

            passed = len(results_summary) > 0
            logger.info(
                f"Smoke-test {'PASSED' if passed else 'FAILED'} "
                f"(hits: {len(results_summary)})"
            )
            return passed, {
                "query": query,
                "hits_count": len(results_summary),
                "top_hits": results_summary,
            }

        except Exception as e:
            logger.error(f"Smoke-test failed with exception: {e}")
            return False, {"query": query, "error": str(e)}

    def remove_stale_source_points(self, source_filename: str):
        """Remove previously indexed points belonging to a changed source."""
        if source_filename not in self._manifest["sources"]:
            return

        old_point_ids = self._manifest["sources"][source_filename].get(
            "point_ids", []
        )

        if old_point_ids:
            logger.info(
                f"Removing {len(old_point_ids)} stale points "
                f"for source '{source_filename}'"
            )
            try:
                self.vector_store.client.delete(
                    collection_name=self.vector_store.collection_name,
                    points_selector=models.PointIdsList(points=old_point_ids),
                )
            except Exception as e:
                logger.warning(f"Error removing stale points: {e}")

        del self._manifest["sources"][source_filename]
        self._save_manifest()
