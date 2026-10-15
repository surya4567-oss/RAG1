"""Inspects representative formed documents and enriched metadata before indexing."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.config import PipelineConfig
from src.ingestion.discovery import DatasetDiscoverer
from src.ingestion.document_builder import DocumentBuilder
from src.ingestion.loaders import DatasetLoader
from src.ingestion.metadata_enricher import MetadataEnricher


def main():
    parser = argparse.ArgumentParser(
        description="Inspect representative logical documents and their enriched metadata."
    )
    parser.add_argument(
        "--samples-per-source",
        type=int,
        default=2,
        help="Number of document samples to display for each source type.",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    discoverer = DatasetDiscoverer(config)
    enricher = MetadataEnricher(config)

    inventory = discoverer.discover()
    valid_sources = [s for s in inventory.discovered_sources if s.is_valid]

    print("\n" + "=" * 80)
    print("REPRESENTATIVE DOCUMENT INSPECTION")
    print("=" * 80)

    for src in valid_sources:
        print(f"\n>>> SOURCE: {src.category.upper()} ({src.relative_path})")
        docs = []

        if src.category == "testcases":
            recs = DatasetLoader.load_csv(src.path, sample_limit=args.samples_per_source)
            for r in recs:
                docs.append(DocumentBuilder.build_test_case_doc(r, src.path.name))
        elif src.category == "test_execution":
            recs = DatasetLoader.load_csv(src.path, sample_limit=args.samples_per_source)
            for r in recs:
                docs.append(DocumentBuilder.build_test_execution_doc(r, src.path.name))
        elif src.category == "defects":
            recs = DatasetLoader.load_csv(src.path, sample_limit=args.samples_per_source)
            for r in recs:
                docs.append(DocumentBuilder.build_defect_doc(r, src.path.name))
        elif src.category == "srs":
            recs = DatasetLoader.load_srs_pdf(src.path)[:args.samples_per_source]
            for r in recs:
                docs.append(DocumentBuilder.build_srs_doc(r, src.path.name))
        elif src.category == "failure_logs":
            recs = DatasetLoader.load_failure_logs([src.path], sample_limit=args.samples_per_source)
            for r in recs:
                docs.append(DocumentBuilder.build_failure_log_doc(r, src.path.name))
        elif src.category == "screenshots":
            recs = DatasetLoader.load_screenshot_jsons([src.path], sample_limit=args.samples_per_source)
            for r in recs:
                docs.append(DocumentBuilder.build_screenshot_doc(r, src.path.name))

        enriched = enricher.enrich_batch(docs)
        for i, doc in enumerate(enriched, 1):
            print(f"\n--- [Sample {i}] Document ID: {doc.metadata.get('document_id')} ---")
            print("METADATA PAYLOAD:")
            for k, v in doc.metadata.items():
                print(f"  {k}: {v}")
            print("\nPAGE CONTENT (Snippet):")
            snippet = "\n".join(doc.page_content.splitlines()[:15])
            print(snippet)
            if len(doc.page_content.splitlines()) > 15:
                print("  ...")


if __name__ == "__main__":
    main()
