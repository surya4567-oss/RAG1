
"""Main ingestion orchestrator executing the 15-step pipeline."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.ingestion.chunker import SourceAwareChunker
from src.ingestion.cleaners import DataCleaner, QuarantineManager
from src.ingestion.config import PipelineConfig, default_config
from src.ingestion.deduplicator import Deduplicator
from src.ingestion.discovery import DatasetDiscoverer, InventoryReport
from src.ingestion.document_builder import DocumentBuilder
from src.ingestion.indexer import IndexingResult, QdrantIndexer
from src.ingestion.loaders import DatasetLoader
from src.ingestion.logging_config import logger
from src.ingestion.metadata_enricher import MetadataEnricher
from src.ingestion.reporting import IngestionReporter
from src.ingestion.validators import CrossFileValidator


class IngestionPipeline:
    """End-to-end Vanilla RAG ingestion pipeline with timing."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config
        self.discoverer = DatasetDiscoverer(self.config)
        self.enricher = MetadataEnricher(self.config)
        self.chunker = SourceAwareChunker(self.config)
        self.deduplicator = Deduplicator()
        self.quarantine_manager = QuarantineManager(
            self.config.quarantine_dir
        )
        self.reporter = IngestionReporter(self.config)
        self.indexer: Optional[QdrantIndexer] = None

    def run(self) -> Tuple[Path, Path]:
        """Execute ingestion and report timings for each major stage."""
        pipeline_start = time.perf_counter()
        logger.info("=" * 80)
        logger.info("Starting Vanilla RAG Ingestion Pipeline")
        logger.info(
            f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(
            f"Target collection: '{self.config.qdrant_collection}' | "
            f"Dry run: {self.config.dry_run}"
        )
        logger.info("=" * 80)

        errors: List[str] = []
        warnings: List[str] = []

        # Steps 1-2: Discover and validate sources
        preprocessing_start = time.perf_counter()
        logger.info("[Step 1/15] Discovering source files...")
        inventory: InventoryReport = self.discoverer.discover()

        if inventory.missing_sources:
            msg = f"Missing source files/directories: {inventory.missing_sources}"
            warnings.append(msg)
            logger.warning(msg)

        logger.info("[Step 2/15] Validating discovered sources and schemas...")
        valid_sources = [
            source for source in inventory.discovered_sources
            if source.is_valid
        ]

        # Steps 3-4: Load and normalize records
        logger.info("[Step 3-4/15] Loading and normalizing source records...")
        test_cases_raw = []
        executions_raw = []
        defects_raw = []
        srs_sections_raw = []
        failure_logs_raw = []
        screenshots_raw = []

        sample_limit = self.config.sample_limit

        for src in valid_sources:
            if (
                self.config.source_filter
                and self.config.source_filter.lower()
                not in src.category.lower()
            ):
                continue

            try:
                if src.category == "testcases":
                    test_cases_raw.extend(
                        DatasetLoader.load_csv(src.path, sample_limit)
                    )
                elif src.category == "test_execution":
                    executions_raw.extend(
                        DatasetLoader.load_csv(src.path, sample_limit)
                    )
                elif src.category == "defects":
                    defects_raw.extend(
                        DatasetLoader.load_csv(src.path, sample_limit)
                    )
                elif src.category == "srs":
                    srs_sections_raw.extend(
                        DatasetLoader.load_srs_pdf(src.path)
                    )
            except Exception as exc:
                message = f"Failed to load {src.relative_path}: {exc}"
                errors.append(message)
                logger.exception(message)

        log_sources = [
            src.path for src in valid_sources
            if src.category == "failure_logs"
            and (
                not self.config.source_filter
                or "failure" in self.config.source_filter.lower()
            )
        ]
        if log_sources:
            failure_logs_raw = DatasetLoader.load_failure_logs(
                log_sources, sample_limit
            )

        shot_sources = [
            src.path for src in valid_sources
            if src.category == "screenshots"
            and (
                not self.config.source_filter
                or "screen" in self.config.source_filter.lower()
            )
        ]
        if shot_sources:
            screenshots_raw = DatasetLoader.load_screenshot_jsons(
                shot_sources, sample_limit
            )

        # Step 5: Validate cross-dataset relationships
        logger.info("[Step 5/15] Validating cross-dataset relationships...")
        validator = CrossFileValidator()
        validator.register_test_cases(test_cases_raw)
        validator.register_executions(executions_raw)
        validator.register_defects(defects_raw)
        validator.register_srs_sections(srs_sections_raw)

        relationship_report = validator.validate_all(
            test_cases=test_cases_raw,
            executions=executions_raw,
            defects=defects_raw,
            failure_logs=failure_logs_raw,
            screenshots=screenshots_raw,
        )

        # Step 6: Build logical documents
        logger.info("[Step 6/15] Building logical documents...")
        logical_documents: List[Document] = []

        for tc in test_cases_raw:
            logical_documents.append(
                DocumentBuilder.build_test_case_doc(
                    tc, source_file="testcases.csv"
                )
            )

        for execution in executions_raw:
            logical_documents.append(
                DocumentBuilder.build_test_execution_doc(
                    execution, source_file="test_execution.csv"
                )
            )

        for defect in defects_raw:
            logical_documents.append(
                DocumentBuilder.build_defect_doc(
                    defect, source_file="defects_rows.csv"
                )
            )

        for section in srs_sections_raw:
            logical_documents.append(
                DocumentBuilder.build_srs_doc(
                    section,
                    source_file=section.get("source_file", "SRS.pdf"),
                )
            )

        for failure in failure_logs_raw:
            logical_documents.append(
                DocumentBuilder.build_failure_log_doc(
                    failure,
                    source_file=failure.get(
                        "log_file", "failure_log.txt"
                    ),
                )
            )

        for screenshot in screenshots_raw:
            logical_documents.append(
                DocumentBuilder.build_screenshot_doc(
                    screenshot,
                    source_file=screenshot.get(
                        "screenshot_file", "screenshot.json"
                    ),
                )
            )

        doc_counts_by_type: Dict[str, int] = {}
        for doc in logical_documents:
            doc_type = doc.metadata.get("document_type", "unknown")
            doc_counts_by_type[doc_type] = (
                doc_counts_by_type.get(doc_type, 0) + 1
            )

        logger.info(
            f"Logical documents: {len(logical_documents):,} | "
            f"Types: {doc_counts_by_type}"
        )

        # Step 7: Enrich metadata
        logger.info("[Step 7/15] Enriching metadata and assigning IDs...")
        enriched_documents = self.enricher.enrich_batch(logical_documents)

        # Step 8: Chunk documents
        logger.info("[Step 8/15] Chunking documents...")
        chunks = self.chunker.chunk_documents(enriched_documents)

        # Steps 9-10: Deduplicate
        logger.info("[Step 9-10/15] Deduplicating and resolving chunk IDs...")
        filtered_docs, dedup_stats = self.deduplicator.filter_documents(
            enriched_documents
        )
        final_chunks = self.deduplicator.filter_chunks(chunks)

        chunk_sizes = [len(chunk.page_content) for chunk in final_chunks]
        chunk_stats = {
            "total_chunks": len(final_chunks),
            "min_size": min(chunk_sizes) if chunk_sizes else 0,
            "max_size": max(chunk_sizes) if chunk_sizes else 0,
            "avg_size": (
                round(sum(chunk_sizes) / len(chunk_sizes), 1)
                if chunk_sizes else 0
            ),
        }

        preprocessing_elapsed = time.perf_counter() - preprocessing_start
        logger.info(
            f"PREPROCESSING TIME: {preprocessing_elapsed / 60:.2f} minutes"
        )
        logger.info(f"Chunks ready for indexing: {len(final_chunks):,}")

        # Steps 11-14: Index vectors
        indexing_result = IndexingResult(
            total_chunks=len(final_chunks)
        )
        indexing_elapsed = 0.0

        if self.config.dry_run:
            logger.info("[DRY RUN] Skipping embedding and Qdrant indexing.")
            indexing_result.points_skipped = len(final_chunks)
        else:
            logger.info("[Step 11-14/15] Initializing indexer...")
            self.indexer = QdrantIndexer(self.config)

            index_start = time.perf_counter()
            logger.info(
                f"INDEXING STARTED: {len(final_chunks):,} chunks"
            )

            try:
                indexing_result = self.indexer.index_chunks(final_chunks)
            except BaseException:
                indexing_elapsed = time.perf_counter() - index_start
                total_elapsed = time.perf_counter() - pipeline_start

                logger.exception("Indexing was interrupted or failed.")
                logger.info(
                    f"TIME BEFORE INTERRUPTION — indexing: "
                    f"{indexing_elapsed / 60:.2f} minutes"
                )
                logger.info(
                    f"TIME BEFORE INTERRUPTION — total: "
                    f"{total_elapsed / 60:.2f} minutes"
                )
                raise

            indexing_elapsed = time.perf_counter() - index_start
            logger.info(
                f"INDEXING TIME: {indexing_elapsed / 60:.2f} minutes "
                f"({indexing_elapsed / 3600:.2f} hours)"
            )

        # Step 15: Generate reports
        logger.info("[Step 15/15] Generating reports...")
        duration = time.perf_counter() - pipeline_start

        json_report, markdown_report = self.reporter.generate_report(
            inventory_summary=inventory.summary(),
            relationship_summary=relationship_report.summary(),
            deduplication_summary={
                "total_evaluated": dedup_stats.total_evaluated,
                "unique_kept": dedup_stats.unique_records_kept,
                "exact_duplicates_skipped": (
                    dedup_stats.exact_duplicates_skipped
                ),
            },
            doc_counts_by_type=doc_counts_by_type,
            chunk_stats=chunk_stats,
            indexing_summary={
                "points_inserted": indexing_result.points_inserted,
                "points_updated": indexing_result.points_updated,
                "points_failed": indexing_result.points_failed,
                "points_skipped": indexing_result.points_skipped,
                "collection_final_count": (
                    indexing_result.collection_final_count
                ),
                "smoke_test_passed": indexing_result.smoke_test_passed,
                "smoke_test_details": indexing_result.smoke_test_details,
            },
            quarantine_count=len(
                self.quarantine_manager.quarantined_records
            ),
            dry_run=self.config.dry_run,
            duration_seconds=duration,
            errors=errors + indexing_result.errors,
            warnings=warnings,
        )

        logger.info("=" * 80)
        logger.info(
            f"TOTAL PIPELINE TIME: {duration / 60:.2f} minutes "
            f"({duration / 3600:.2f} hours)"
        )
        logger.info(f"Preprocessing time: {preprocessing_elapsed / 60:.2f} minutes")
        logger.info(f"Indexing time: {indexing_elapsed / 60:.2f} minutes")
        logger.info(
            f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logger.info(f"JSON report: {json_report}")
        logger.info(f"Markdown report: {markdown_report}")
        logger.info("=" * 80)

        return json_report, markdown_report
