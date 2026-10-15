# Vanilla RAG Ingestion Pipeline with Metadata Awareness and QA Traceability

A modular, production-ready Vanilla RAG ingestion and indexing pipeline built with LangChain, local BGE embeddings (`BAAI/bge-base-en-v1.5`), and persistent local Qdrant.

## Architecture and Workflow

The pipeline executes a deterministic 15-step ingestion workflow:

```text
1. Discover source files (CSVs, SRS PDF, 10 Failure Logs, 10 Screenshot JSONs)
2. Validate files and schemas
3. Load and extract records
4. Clean and normalize (whitespace, encoding safety, type casting)
5. Validate data quality and cross-dataset referential integrity
6. Form logical documents before chunking (readable markdown with visible IDs)
7. Enrich metadata (standardized schema + source-specific fields + provenance)
8. Apply source-aware chunking (preserving headers, stack traces, and IDs)
9. Generate stable document and chunk IDs (UUID v5 / deterministic hashes)
10. Detect duplicates (exact duplicate skipping while preserving distinct occurrences)
11. Generate local BGE embeddings (BAAI/bge-base-en-v1.5, 768-dim, no OpenAI APIs)
12. Validate Qdrant configuration (768-dim Cosine vector verification)
13. Upsert chunks into Qdrant (batch upserts with payload indexes)
14. Verify indexing and retrieval (automated smoke-test query)
15. Generate ingestion and validation reports (JSON + Markdown summaries)
```

## Dataset Modalities Handled

1. **Test Cases (`data/testcases.csv`)**: Row-level logical documents preserving preconditions, test steps, expected/actual results, module, requirement ID, and acceptance criteria.
2. **Test Executions (`data/test_execution.csv`)**: Row-level logical documents preserving execution ID, test case ID, environment, browser, operating system, status, and execution notes.
3. **Defects (`data/defects_rows.csv`)**: Row-level logical documents preserving defect ID, linked test case, linked execution, title, description, root cause analysis, severity, priority, and reproduction steps.
4. **SRS PDF (`data/SRS.pdf`)**: Page and section-level logical documents preserving section titles, section numbers, requirement IDs (e.g. `FR-COM-xxx`), and approval status.
5. **Failure Logs (`data/failure_logs/*.txt`)**: 10 log files parsed into structured blocks delimited by `====...====`, extracting Python tracebacks, database errors, HTTP status codes, server logs, resolution hints, line ranges, and linked QA identifiers.
6. **Screenshot OCR Evidence (`data/screenshots/*.json`)**: 10 JSON files parsed into rich UI state documents with screen names, buttons, labels, menus, tables, field values, error/warning messages, and timestamps.

## Project Structure

```text
VanillaRAG/
├── data/
│   ├── SRS.pdf
│   ├── defects_rows.csv
│   ├── test_execution.csv
│   ├── testcases.csv
│   ├── failure_logs/           # 10 log files
│   ├── screenshots/            # 10 JSON files
│   ├── processed/              # Manifest and staging
│   ├── quarantine/             # Invalid records quarantine
│   └── qdrant_db/              # Persistent local Qdrant database
├── src/
│   └── ingestion/
│       ├── __init__.py
│       ├── config.py           # Configuration and environment loader
│       ├── logging_config.py   # Structured logging
│       ├── discovery.py        # Dataset discovery & source inventory
│       ├── loaders.py          # Unified data loaders
│       ├── extractors.py       # Format-aware extractors (CSV, PDF, logs, OCR)
│       ├── cleaners.py         # Conservative cleaning & quarantine manager
│       ├── validators.py       # Cross-file referential integrity validator
│       ├── metadata_schema.py  # Standardized Pydantic schemas & sanitizer
│       ├── document_builder.py # Logical Document formation
│       ├── metadata_enricher.py# Metadata enrichment & stable ID generator
│       ├── chunker.py          # Source-aware chunker
│       ├── deduplicator.py     # Deduplication & stable chunk ID resolver
│       ├── embeddings.py       # Local BGE embeddings (CPU/CUDA)
│       ├── vector_store.py     # Qdrant connection & collection verification
│       ├── indexer.py          # Batch upsert, manifest tracking & smoke test
│       ├── pipeline.py         # Full 15-step pipeline orchestrator
│       └── reporting.py        # JSON & Markdown report generation
├── scripts/
│   ├── run_ingestion.py        # CLI pipeline runner
│   ├── inspect_documents.py    # Document preview & metadata inspector
│   ├── inspect_collection.py   # Qdrant status & vector query inspector
│   └── reset_collection.py     # Safe collection reset CLI (--confirm)
├── tests/                      # Full test suite with pytest
├── reports/                    # Execution reports
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Run Tests

```bash
pytest tests/ -v
```

### 3. Inspect Formed Documents (Dry Preview)

```bash
python scripts/inspect_documents.py --samples-per-source 2
```

### 4. Run Ingestion Dry-Run

```bash
python scripts/run_ingestion.py --dry-run
```

### 5. Run Full Ingestion Pipeline

```bash
# Ingest with a sample limit for testing
python scripts/run_ingestion.py --limit 100

# Full ingestion across the entire dataset
python scripts/run_ingestion.py
```

### 6. Inspect Qdrant Collection & Test Retrieval

```bash
python scripts/inspect_collection.py --query "permission error access denied"
```

### 7. Reset Collection (If needed)

```bash
python scripts/reset_collection.py --confirm
```

## Key Architectural Guarantees

- **No API Keys or OpenAI Dependencies**: Embeddings are calculated locally using `BAAI/bge-base-en-v1.5` (768 dimensions, normalized Cosine).
- **Referential Integrity Without Data Mutation**: Conflicting identifiers across screenshots, logs, defects, and test cases are flagged and tracked in `inconsistency_reasons`, preserving original source values.
- **Persistent Local Qdrant**: Storage persists in `data/qdrant_db/` without requiring Docker, while also supporting remote Qdrant via `QDRANT_URL`.
- **Idempotency & Re-ingestion**: Deterministic point UUIDs (derived from chunk IDs) prevent duplicate vector points when re-running unchanged data.
