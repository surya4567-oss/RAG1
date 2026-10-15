"""Deduplication and stable ID generator."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from langchain_core.documents import Document

from src.ingestion.logging_config import logger


@dataclass
class DeduplicationStats:
    total_evaluated: int = 0
    unique_records_kept: int = 0
    exact_duplicates_skipped: int = 0
    duplicate_record_ids: List[str] = field(default_factory=list)


class Deduplicator:
    """Manages idempotent re-ingestion and deduplication while respecting record identity."""

    def __init__(self):
        # Maps (document_type, record_id) -> content_hash
        self.seen_record_hashes: Dict[Tuple[str, str], str] = {}
        # Set of chunk_ids
        self.seen_chunk_ids: Set[str] = set()

    def filter_documents(self, documents: List[Document]) -> Tuple[List[Document], DeduplicationStats]:
        """Filter exact duplicate documents having the same record identity and content."""
        stats = DeduplicationStats(total_evaluated=len(documents))
        unique_docs: List[Document] = []

        for doc in documents:
            doc_type = doc.metadata.get("document_type", "unknown")
            rec_id = doc.metadata.get("record_id", "unknown")
            content_hash = doc.metadata.get("content_hash", "")

            key = (doc_type, rec_id)
            if key in self.seen_record_hashes:
                # If exact same record ID and same content, skip duplicate
                if self.seen_record_hashes[key] == content_hash:
                    stats.exact_duplicates_skipped += 1
                    stats.duplicate_record_ids.append(f"{doc_type}:{rec_id}")
                    continue
                else:
                    # Content updated for existing record: replace hash and keep
                    self.seen_record_hashes[key] = content_hash
                    unique_docs.append(doc)
            else:
                self.seen_record_hashes[key] = content_hash
                unique_docs.append(doc)

        stats.unique_records_kept = len(unique_docs)
        logger.info(
            f"Deduplication complete: {stats.unique_records_kept} unique documents retained, "
            f"{stats.exact_duplicates_skipped} exact duplicates skipped."
        )
        return unique_docs, stats

    def filter_chunks(self, chunks: List[Document]) -> List[Document]:
        """Ensure chunk ID uniqueness."""
        unique_chunks: List[Document] = []
        for ch in chunks:
            chunk_id = ch.metadata.get("chunk_id")
            if chunk_id and chunk_id in self.seen_chunk_ids:
                # Disambiguate duplicate chunk_id if collision occurs
                idx = 1
                while f"{chunk_id}_dup{idx}" in self.seen_chunk_ids:
                    idx += 1
                new_id = f"{chunk_id}_dup{idx}"
                ch.metadata["chunk_id"] = new_id
                chunk_id = new_id

            if chunk_id:
                self.seen_chunk_ids.add(chunk_id)
            unique_chunks.append(ch)

        return unique_chunks
