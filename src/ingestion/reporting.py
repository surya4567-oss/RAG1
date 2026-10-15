"""Generates comprehensive JSON and Markdown ingestion reports."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.ingestion.config import PipelineConfig, default_config
from src.ingestion.logging_config import logger


class IngestionReporter:
    """Produces detailed execution reports and Markdown summaries."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or default_config
        self.reports_dir = Path(self.config.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        inventory_summary: Dict[str, Any],
        relationship_summary: Dict[str, Any],
        deduplication_summary: Dict[str, Any],
        doc_counts_by_type: Dict[str, int],
        chunk_stats: Dict[str, Any],
        indexing_summary: Dict[str, Any],
        quarantine_count: int,
        dry_run: bool,
        duration_seconds: float,
        errors: List[str],
        warnings: List[str],
    ) -> Tuple[Path, Path]:
        """Generate both JSON report and human-readable Markdown summary."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_path = self.reports_dir / f"ingestion_report_{timestamp}.json"
        md_path = self.reports_dir / f"ingestion_report_{timestamp}.md"

        report_data = {
            "execution_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(duration_seconds, 2),
                "dry_run": dry_run,
                "dataset_version": self.config.dataset_version,
                "schema_version": self.config.schema_version,
            },
            "configuration": {
                "embedding_model": self.config.embedding_model,
                "embedding_dimension": self.config.embedding_dimension,
                "embedding_device": self.config.embedding_device,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "qdrant_collection": self.config.qdrant_collection,
            },
            "source_inventory": inventory_summary,
            "document_counts": doc_counts_by_type,
            "chunk_statistics": chunk_stats,
            "deduplication": deduplication_summary,
            "relationship_validation": relationship_summary,
            "quarantined_records_count": quarantine_count,
            "indexing_results": indexing_summary,
            "errors": errors,
            "warnings": warnings,
        }

        # Write JSON report
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # Write Markdown summary
        md_content = self._render_markdown_report(report_data)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Ingestion reports generated: \n  JSON: {json_path}\n  Markdown: {md_path}")
        return json_path, md_path

    def _render_markdown_report(self, data: Dict[str, Any]) -> str:
        exec_meta = data["execution_metadata"]
        cfg = data["configuration"]
        inv = data["source_inventory"]
        docs = data["document_counts"]
        chunks = data["chunk_statistics"]
        dedup = data["deduplication"]
        rel = data["relationship_validation"]
        idx = data["indexing_results"]

        md = [
            "# Vanilla RAG Ingestion Pipeline Execution Report",
            f"**Execution Timestamp**: `{exec_meta['timestamp']}`  ",
            f"**Duration**: `{exec_meta['duration_seconds']}s` | **Mode**: `{'DRY RUN' if exec_meta['dry_run'] else 'FULL INDEX'}`  ",
            f"**Embedding Model**: `{cfg['embedding_model']}` ({cfg['embedding_dimension']} dim, {cfg['embedding_device']})  ",
            f"**Target Collection**: `{cfg['qdrant_collection']}`  ",
            "",
            "---",
            "## 1. Discovered Sources Inventory",
            f"- **Total Discovered Source Files**: `{inv.get('total_files', 0)}`",
            f"- **Total Estimated Source Records**: `{inv.get('total_records', 0)}`",
            "",
            "| Source Category | Files Found | Status |",
            "|---|---|---|",
        ]

        cats = inv.get("categories", {})
        for cat_name, files in cats.items():
            cnt = len(files)
            status = "✅ Discovered" if cnt > 0 else "❌ Missing"
            sample = f" ({files[0]} + {cnt-1} more)" if cnt > 1 else f" ({files[0]})" if cnt == 1 else ""
            md.append(f"| `{cat_name}` | {cnt}{sample} | {status} |")

        md.extend([
            "",
            "---",
            "## 2. Logical Document Formation",
            "| Document Type | Logical Documents Count |",
            "|---|---|",
        ])
        for doc_type, count in docs.items():
            md.append(f"| `{doc_type}` | {count:,} |")
        md.append(f"| **Total Documents** | **{sum(docs.values()):,}** |")

        md.extend([
            "",
            "---",
            "## 3. Chunking & Deduplication",
            f"- **Total Resulting Chunks**: `{chunks.get('total_chunks', 0):,}`",
            f"- **Min / Max / Avg Chunk Size**: `{chunks.get('min_size', 0)}` / `{chunks.get('max_size', 0)}` / `{chunks.get('avg_size', 0)}` chars",
            f"- **Exact Duplicates Skipped**: `{dedup.get('exact_duplicates_skipped', 0)}`",
            f"- **Quarantined Records**: `{data.get('quarantined_records_count', 0)}`",
            "",
            "---",
            "## 4. Cross-Dataset Referential Integrity & Validation",
            f"- **Total Records Cross-Checked**: `{rel.get('total_checked', 0):,}`",
            f"- **Consistent Records**: `{rel.get('consistent_count', 0):,}`",
            f"- **Inconsistent Records (Flagged, preserved)**: `{rel.get('inconsistent_count', 0):,}`",
            f"- **Cross-Link Mismatches (Defect/Execution)**: `{rel.get('cross_link_mismatches', 0):,}`",
            f"- **Unmatched Test Case References**: `{rel.get('missing_test_case_refs', 0):,}`",
            f"- **Unmatched Execution References**: `{rel.get('missing_execution_refs', 0):,}`",
            f"- **Unmatched Defect References**: `{rel.get('missing_defect_refs', 0):,}`",
            "",
            "---",
            "## 5. Qdrant Indexing & Verification",
            f"- **Points Inserted / Updated**: `{idx.get('points_inserted', 0):,}`",
            f"- **Points Failed**: `{idx.get('points_failed', 0)}`",
            f"- **Final Collection Point Count**: `{idx.get('collection_final_count', 0):,}`",
            f"- **Smoke Test Status**: `{'✅ PASSED' if idx.get('smoke_test_passed') else '⚠️ SKIPPED / FAILED'}`",
        ])

        smoke = idx.get("smoke_test_details", {})
        if smoke.get("top_hits"):
            md.append("\n### Smoke Test Sample Hits:")
            for hit in smoke["top_hits"]:
                md.append(
                    f"- **Score**: `{hit.get('score')}` | **Type**: `{hit.get('document_type')}` | "
                    f"**ID**: `{hit.get('record_id')}`\n  `{hit.get('snippet')}...`"
                )

        if data.get("errors"):
            md.append("\n--- \n## ⚠️ Errors Encountered:")
            for err in data["errors"]:
                md.append(f"- {err}")

        return "\n".join(md)
