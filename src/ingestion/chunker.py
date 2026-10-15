"""Source-aware document chunking preserving provenance and QA identifiers."""

import hashlib
from typing import List, Optional
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Lightweight fallback character splitter
    class RecursiveCharacterTextSplitter:  # type: ignore
        def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150, separators: Optional[List[str]] = None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.separators = separators or ["\n\n", "\n", " ", ""]

        def split_text(self, text: str) -> List[str]:
            if len(text) <= self.chunk_size:
                return [text]
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunks.append(text[start:end])
                if end == len(text):
                    break
                start = end - self.chunk_overlap
            return chunks

from src.ingestion.config import PipelineConfig, default_config
from src.ingestion.logging_config import logger


class SourceAwareChunker:
    """Applies modality-appropriate chunking rules while maintaining stable IDs and metadata."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", "; ", ". ", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Process a list of enriched logical documents into indexed chunks."""
        chunks: List[Document] = []
        for doc in documents:
            chunks.extend(self.chunk_single_document(doc))
        logger.info(f"Chunked {len(documents)} logical documents into {len(chunks)} chunks")
        return chunks

    def chunk_single_document(self, doc: Document) -> List[Document]:
        """Split a single logical document based on its type and size."""
        text = doc.page_content
        parent_id = doc.metadata.get("document_id") or f"doc_{doc.metadata.get('record_id', 'unknown')}"
        doc_type = doc.metadata.get("document_type")

        # If document text is within chunk_size, keep as a single chunk
        if len(text) <= self.config.chunk_size:
            chunk_meta = dict(doc.metadata)
            chunk_meta["parent_document_id"] = parent_id
            chunk_meta["chunk_index"] = 0
            chunk_meta["chunk_id"] = f"{parent_id}_c0"
            chunk_meta["content_hash"] = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
            return [Document(page_content=text, metadata=chunk_meta)]

        # Otherwise, split oversized document
        split_texts = self.splitter.split_text(text)
        doc_chunks: List[Document] = []

        # Context header for split chunks to retain primary identifiers
        header_context = self._get_context_header(doc)

        for idx, chunk_text in enumerate(split_texts):
            # Prepend context header to subsequent chunks if needed
            final_text = chunk_text
            if idx > 0 and header_context and not chunk_text.startswith(header_context):
                final_text = f"{header_context}\n\n[Continued Section {idx+1}]\n{chunk_text}"

            chunk_meta = dict(doc.metadata)
            chunk_meta["parent_document_id"] = parent_id
            chunk_meta["chunk_index"] = idx
            chunk_meta["chunk_id"] = f"{parent_id}_c{idx}"
            chunk_meta["content_hash"] = hashlib.sha256(final_text.strip().encode("utf-8")).hexdigest()

            doc_chunks.append(Document(page_content=final_text, metadata=chunk_meta))

        return doc_chunks

    def _get_context_header(self, doc: Document) -> str:
        """Create a concise header to maintain retrieval context in split chunks."""
        doc_type = doc.metadata.get("document_type")
        rec_id = doc.metadata.get("record_id", "")
        module = doc.metadata.get("module", "")

        parts = [f"Context: {doc_type} {rec_id}"]
        if module:
            parts.append(f"Module: {module}")
        return " | ".join(parts)
