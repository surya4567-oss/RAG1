# Complete AI Coding Prompt: Build a Metadata-Aware Vanilla RAG Ingestion Pipeline

## 1. Role and Objective

Act as a senior Python AI engineer with expertise in LangChain, document ingestion, data quality, embedding models, and Qdrant.

Build a complete, modular, runnable Vanilla RAG ingestion and indexing pipeline for my QA traceability dataset.

The pipeline must load structured and unstructured data, validate the source records, clean and normalize content, form logical documents, enrich them with dataset-specific metadata, apply source-aware chunking, generate local BGE embeddings, and index the resulting chunks into persistent local Qdrant.

Do not just generate a project template. Inspect the available dataset, implement the actual code, run tests where possible, and verify indexing results.

### Mandatory constraints

- Use Python and LangChain.
- Use `BAAI/bge-base-en-v1.5` as the default embedding model.
- Do not include OpenAI embeddings or OpenAI API calls.
- Use local Qdrant with persistent storage.
- Implement document formation explicitly before chunking.
- Implement metadata enrichment based on the actual dataset schemas.
- Preserve source provenance and cross-file identifiers.
- Do not silently repair inconsistent source data.
- Do not implement answer generation, query routing, or a complete RAG query pipeline.
- Do not claim that indexing is complete unless Qdrant has been populated and verified.

## 2. Dataset Layout and Discovery

The logical dataset contains:

- A test case CSV.
- A test execution CSV.
- A defects CSV.
- An SRS PDF.
- A folder containing 10 failure-log files.
- A separate folder containing 10 screenshot/OCR JSON files.

Expected logical layout:

```text
data/
├── raw/
│   ├── testcase.csv
│   ├── testexecution.csv
│   ├── defects.csv
│   ├── srs.pdf
│   ├── failurelogs/
│   │   ├── log_file_01.txt
│   │   ├── ...
│   │   └── log_file_10.txt
│   └── screenshots/
│       ├── screenshot_file_01.json
│       ├── ...
│       └── screenshot_file_10.json
├── processed/
└── quarantine/
```

These are logical names. Inspect the real repository before assuming the exact filenames or directory names.

### Discovery requirements

1. Recursively discover the CSV, PDF, text-log, and JSON files in their configured source directories.
2. Identify the two directories containing the 10 failure-log files and 10 screenshot JSON files.
3. Inspect the actual filenames, file extensions, encodings, headers, JSON structures, and representative records.
4. Determine whether JSON files contain arrays, objects, or JSON Lines.
5. Detect duplicate filenames, missing expected files, empty files, and unsupported formats.
6. Do not accidentally ingest generated reports, processed outputs, caches, or previous ingestion artifacts as source data.
7. Generate a source inventory report showing discovered files and their record counts.
8. Make all source paths configurable.

If the repository contains differently named files, use the discovered names and configure their roles explicitly rather than renaming or overwriting source files.

## 3. Dataset-Specific Metadata Design

Create `src/ingestion/metadata_schema.py` to define and validate the common metadata schema and source-specific mappings.

Use a common schema plus source-specific fields. Do not force every field onto every document.

### 3.1 Common metadata

Apply the following fields to every applicable logical document and its resulting chunks:

| Field | Meaning |
|---|---|
| `source` | Original filename |
| `source_path` | Relative path of the source file |
| `source_type` | `csv`, `pdf`, `text_log`, or `json` |
| `document_type` | Semantic record type, such as `test_case` or `failure_log` |
| `record_id` | Original source record identifier |
| `document_id` | Stable identifier for the logical document |
| `chunk_id` | Stable identifier for the indexed chunk |
| `parent_document_id` | ID of the logical document from which the chunk originated |
| `content_hash` | Hash of normalized source content |
| `dataset_version` | Version or manifest identifier for the source dataset |
| `schema_version` | Version of the metadata schema |
| `ingestion_timestamp` | UTC time when the document was processed |
| `chunk_index` | Zero-based chunk position within the logical document |

Do not confuse `document_id` with `chunk_id`. One document may produce multiple chunks.

Use deterministic IDs for stable document and chunk identity. Avoid generating a different ID every time unchanged data is ingested.

For Qdrant compatibility, store metadata as supported JSON payload values. Use ISO-formatted strings for dates and timestamps, numbers for numeric values, booleans for Boolean values, and lists or dictionaries only where appropriate.

Do not store missing values as the literal strings `"None"`, `"null"`, or `"NaN"`.

### 3.2 Test case metadata

Inspect the actual test case CSV. The available dataset documentation identifies fields including:

- `test_case_id`
- `test_title`
- `module`
- `feature`
- `requirement_id`
- `requirement_statement`
- `test_type`
- `srs_type`
- `preconditions`
- `test_data`
- `test_steps`
- `expected_result`
- `priority`
- `created_by`
- `source_file`
- `srs_page`
- `comments`

There may also be an `index` field or additional fields. Discover the actual schema and retain all meaningful source columns.

Use this metadata mapping:

```python
{
    "document_type": "test_case",
    "record_id": "<test_case_id>",
    "test_case_id": "<test_case_id>",
    "requirement_id": "<requirement_id>",
    "module": "<module>",
    "feature": "<feature>",
    "priority": "<priority>",
    "test_type": "<test_type>",
    "srs_type": "<srs_type>",
    "source_file": "<source_file>",
    "srs_page": "<srs_page>"
}
```

Populate optional fields only when present and valid.

Keep the test title, requirement statement, preconditions, test data, steps, expected result, and other retrieval-relevant content in the document text.

Do not put every descriptive field into Qdrant metadata when it is already adequately represented in the text payload.

### 3.3 Test execution metadata

Inspect the actual execution CSV and map its fields. The documented schema includes:

- `execution_id`
- `test_case_id`
- `build_number`
- `execution_date`
- `tester`
- `environment`
- `browser`
- `operating_system`
- `execution_time_seconds`
- `status`
- `execution_notes`

Retain any other available source columns.

Use the mapping:

```python
{
    "document_type": "test_execution",
    "record_id": "<execution_id>",
    "execution_id": "<execution_id>",
    "test_case_id": "<test_case_id>",
    "build_number": "<build_number>",
    "execution_date": "<execution_date>",
    "tester": "<tester>",
    "environment": "<environment>",
    "browser": "<browser>",
    "operating_system": "<operating_system>",
    "execution_time_seconds": "<execution_time_seconds>",
    "status": "<status>"
}
```

Do not assume that every execution is failed. Preserve the actual execution status.

### 3.4 Defect metadata

Inspect the defects CSV. The available schema documentation identifies:

- `defect_id`
- `linked_test_case`
- `linked_execution`
- `title`
- `description`
- `severity`
- `priority`
- `module`
- `status`
- `ownership`
- `root_cause`
- `bug_type`
- `reproducibility`
- `environment`
- date fields
- build fields
- `regression`
- `comments`
- `fix_version`

Map additional actual fields such as `assigned_to`, `reported_by`, `developer`, `created_date`, `resolved_date`, `found_in_build`, and `fixed_build` if they exist.

Use the mapping:

```python
{
    "document_type": "defect",
    "record_id": "<defect_id>",
    "defect_id": "<defect_id>",
    "test_case_id": "<linked_test_case>",
    "execution_id": "<linked_execution>",
    "severity": "<severity>",
    "priority": "<priority>",
    "module": "<module>",
    "status": "<status>",
    "bug_type": "<bug_type>",
    "root_cause": "<root_cause>",
    "reproducibility": "<reproducibility>",
    "environment": "<environment>",
    "fix_version": "<fix_version>"
}
```

Keep `linked_test_case` and `linked_execution` as original-source references or map them into canonical metadata fields only after validating their meanings.

Do not treat a defect's module, environment, status, or dates as correct simply because they are populated.

### 3.5 SRS PDF metadata

Form logical documents from the SRS based on its actual structure.

Use fields such as:

```python
{
    "document_type": "srs_section",
    "record_id": "<stable_section_or_requirement_id>",
    "section_title": "<section_title>",
    "section_number": "<section_number>",
    "requirement_id": "<requirement_id_if_present>",
    "page_number": 1,
    "page_start": 1,
    "page_end": 1,
    "approval_status": "<source_supported_status>"
}
```

The page numbers above are illustrative. Derive actual page numbers from the PDF loader and document builder.

Preserve requirement identifiers, acceptance criteria, section hierarchy, table context, and the original source filename.

Do not invent requirement IDs. If the document is a draft, preserve its actual approval status or mark the status unknown if it cannot be established.

### 3.6 Failure-log metadata

Each of the 10 log files must be processed separately.

The available source documentation identifies log fields such as:

- Bug ID
- Execution ID
- Test Case ID
- Timestamp
- Module
- User Action
- Python Traceback
- Database Error
- HTTP Status
- Server Log
- Exception
- Resolution Hint

Inspect the actual log format and extract only fields supported by the source.

Use the mapping:

```python
{
    "document_type": "failure_log",
    "record_id": "<stable_log_record_id>",
    "log_file": "<original_log_filename>",
    "log_record_index": 0,
    "log_id": "<stable_source_log_id>",
    "defect_id": "<validated_bug_id_if_present>",
    "execution_id": "<execution_id_if_present>",
    "test_case_id": "<test_case_id_if_present>",
    "timestamp": "<source_timestamp_if_present>",
    "module": "<module_if_present>",
    "log_level": "<level_if_detectable>",
    "error_code": "<code_if_detectable>",
    "exception_type": "<exception_type_if_detectable>",
    "http_status": "<http_status_if_present>",
    "line_start": 1,
    "line_end": 1
}
```

The line ranges are illustrative and must be computed from the source.

Preserve complete exceptions and related stack traces when practical. Do not split an exception away from its associated traceback without retaining enough context to reconstruct the original event.

If a log contains multiple events, create separate logical documents where event boundaries are reliable. Preserve file and line provenance for each event.

### 3.7 Screenshot/OCR JSON metadata

Each of the 10 JSON files must be processed separately. The expected logical content is screenshot/OCR evidence, but inspect the actual JSON schema before implementing transformations.

The available records may include:

- `bug_id`
- `execution_id`
- `test_case_id`
- `screen_name`
- `module`
- `buttons`
- `labels`
- `menus`
- `dialog_boxes`
- `tables`
- `warning_messages`
- `error_messages`
- `field_values`
- `captured_time`
- `user_action`

Use this metadata mapping:

```python
{
    "document_type": "screenshot_evidence",
    "record_id": "<stable_screenshot_record_id>",
    "screenshot_id": "<source_evidence_id_if_present>",
    "screenshot_file": "<original_json_filename>",
    "record_index": 0,
    "defect_id": "<bug_id_if_present>",
    "execution_id": "<execution_id_if_present>",
    "test_case_id": "<test_case_id_if_present>",
    "screen_name": "<screen_name_if_present>",
    "module": "<module_if_present>",
    "captured_time": "<captured_time_if_present>",
    "has_error_message": True,
    "has_warning_message": False
}
```

The Boolean values above are examples. Calculate them from the actual contents.

Preserve meaningful OCR text, error messages, warning messages, UI labels, user actions, and nested field values in `page_content`. Do not embed raw serialized JSON as the only document text.

If an actual screenshot identifier does not exist, use a stable record identifier derived from the source filename and record identity. Do not fabricate a real screenshot path or claim that the underlying image was inspected.

### 3.8 Relationship and integrity metadata

Implement a validation layer that cross-checks IDs across the datasets.

Validate:

- Every execution's `test_case_id` against the canonical test case IDs.
- Every defect's linked test case and execution IDs against the corresponding records.
- Whether a defect's linked execution references the same test case as the defect's linked test case.
- Whether log references to bug, execution, and test case IDs exist.
- Whether screenshot/OCR references to bug, execution, and test case IDs exist.
- Whether screenshot/OCR identifiers agree with the linked defect and execution records.
- Whether SRS requirement IDs referenced by test cases exist in the SRS where extraction supports this check.

Create a structured relationship-validation report.

**Critical rule:** preserve conflicting values exactly as observed. Add validation flags and provenance, but never overwrite an original identifier with a guessed value.

For example, if a screenshot's `bug_id`, `execution_id`, and `test_case_id` do not agree with the canonical defect/execution relationship, retain the three source values and flag the inconsistency. Do not create a false association.

Do not merge records simply because they share a module or have similar text.

## 4. Project Structure

Create the following modular structure, adapting it to any existing repository:

```text
vanilla-rag/
├── data/
│   ├── raw/
│   ├── processed/
│   └── quarantine/
├── src/
│   └── ingestion/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_config.py
│       ├── discovery.py
│       ├── loaders.py
│       ├── extractors.py
│       ├── cleaners.py
│       ├── validators.py
│       ├── metadata_schema.py
│       ├── document_builder.py
│       ├── metadata_enricher.py
│       ├── chunker.py
│       ├── deduplicator.py
│       ├── embeddings.py
│       ├── vector_store.py
│       ├── indexer.py
│       ├── pipeline.py
│       └── reporting.py
├── scripts/
│   ├── __init__.py
│   ├── run_ingestion.py
│   ├── inspect_documents.py
│   ├── inspect_collection.py
│   └── reset_collection.py
├── tests/
├── reports/
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 5. Loading and Extraction

Use LangChain-compatible loading for each source:

- CSV: `CSVLoader` or pandas-based row conversion.
- PDF: `PyMuPDFLoader`.
- Text logs: `TextLoader` plus a format-aware log parser.
- JSON: `JSONLoader` or a custom parser appropriate to the actual JSON structure.

Use custom parsers where required to preserve row boundaries, JSON array records, JSONL records, or log event boundaries.

Every extracted record must retain its original filename and source record identity.

Do not load generated reports or processed files as new input documents.

## 6. Cleaning and Data Quality

Implement conservative cleaning.

- Normalize whitespace and line endings.
- Handle encoding errors safely.
- Normalize metadata field names and types.
- Detect missing or malformed identifiers.
- Validate dates and numeric fields.
- Preserve meaningful punctuation, error codes, stack traces, requirement IDs, and QA identifiers.
- Do not remove repeated log events blindly.
- Do not silently delete invalid records.

Write invalid records and explanations to `data/quarantine/` when appropriate. Include them in the validation report.

Keep original source files immutable.

## 7. Document Formation

Implement `document_builder.py` to produce LangChain `Document` objects before metadata enrichment and chunking.

Use the following rules:

- Test case CSV: one logical document per row.
- Test execution CSV: one logical document per execution row.
- Defects CSV: one logical document per defect row.
- SRS PDF: one logical document per section or requirement where practical.
- Failure logs: one logical document per related event or exception group.
- Screenshot JSON: one logical document per screenshot/OCR record.

Use readable key-value text. Include all relevant source fields and keep identifiers visible in `page_content` when they are important for retrieval.

Keep record boundaries intact. Do not concatenate unrelated records merely to achieve a target chunk size.

## 8. Metadata Enrichment

Implement `metadata_enricher.py` using `metadata_schema.py`.

Requirements:

1. Apply the common metadata schema.
2. Apply the correct source-specific field mapping.
3. Preserve all original IDs and source filenames.
4. Add deterministic document IDs and content hashes.
5. Normalize supported data types.
6. Validate required metadata fields.
7. Preserve conflicting source values and attach validation flags.
8. Avoid inventing missing fields or relationships.
9. Produce a metadata validation report.
10. Pass enriched documents to the chunking stage.

Use separate functions for test cases, executions, defects, SRS sections, failure logs, and screenshot evidence.

## 9. Document-Aware Chunking

Implement `chunker.py`.

Use source-aware chunking, not one blind splitter for all data.

### Test case, execution, and defect CSVs

- Preserve each record as a logical document.
- Keep short records as a single chunk.
- Split only unusually large records.
- Keep IDs and related context in every split chunk.
- Preserve record provenance.

### SRS PDF

- Split by section, subsection, requirement ID, and table boundaries where practical.
- Keep descriptions and acceptance criteria together.
- Use `RecursiveCharacterTextSplitter` as a fallback for oversized sections.
- Preserve page and section metadata.

### Failure logs

- Split at reliable event, timestamp, and exception boundaries.
- Keep related stack traces with their error messages.
- Use recursive splitting only for oversized events.
- Preserve log filename and line ranges.

### Screenshot/OCR JSON

- Keep one screenshot/OCR record per logical document.
- Preserve screen name, user action, error messages, warning messages, and linked identifiers.
- Split oversized records only when necessary.

Default configuration:

```python
chunk_size = 1000
chunk_overlap = 150
```

These values represent characters when using the default `RecursiveCharacterTextSplitter` configuration. Make them configurable.

Avoid unnecessary overlap between independent CSV or JSON records.

Every final chunk must retain its parent document ID, source path, source record identity, and applicable QA identifiers.

## 10. Deduplication and Stable IDs

Implement `deduplicator.py`.

Requirements:

- Use deterministic IDs for logical documents and chunks.
- Detect exact duplicates using normalized content hashes.
- Distinguish identical content from identical record identity.
- Do not merge distinct test cases simply because their text is identical.
- Preserve repeated execution records and legitimate repeated log events.
- Keep the original source record ID when one exists.
- Make re-ingestion idempotent.

If content changes, the record's canonical identity should remain stable while its content hash changes. Chunk IDs may change when chunk content or chunk boundaries change.

Do not delete old indexed chunks until the replacement source version has been successfully indexed and verified.

## 11. Embedding Generation

Implement `embeddings.py` using local BGE embeddings:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 16,
    },
)
```

Requirements:

- Default to CPU for reliable execution.
- Support CUDA when explicitly configured and available.
- Make batch size configurable.
- Generate embeddings from chunk text.
- Do not include serialized metadata in the embedding input by default.
- Validate vector dimensions and finite numerical values.
- Use batch processing.
- Handle errors with useful logs.
- Do not call a hosted embedding API.
- Do not include OpenAI embeddings anywhere in the implementation.

Expected vector dimension for `BAAI/bge-base-en-v1.5` is 768.

## 12. Qdrant and Indexing

Implement `vector_store.py` and `indexer.py`.

Use a persistent local Qdrant database. Support a Docker-based Qdrant server and a clearly configured embedded local mode if appropriate.

Recommended server configuration:

```dotenv
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=qa_knowledge
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DEVICE=cpu
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
INDEX_BATCH_SIZE=64
```

Use a 768-dimensional cosine vector configuration for the default model.

The indexer must:

1. Accept final enriched chunks.
2. Generate embeddings in batches.
3. Store each chunk as a Qdrant point.
4. Store chunk text and metadata in the payload.
5. Use stable point IDs.
6. Upsert in configurable batches.
7. Report inserted, updated, skipped, and failed points.
8. Create useful payload indexes for common filters.
9. Verify collection configuration.
10. Verify indexed point count and perform a smoke-test search.

If the collection already exists with an incompatible vector configuration, stop and explain the mismatch. Do not silently delete or recreate a populated collection.

### Source-specific re-ingestion

Maintain a manifest mapping source files and source versions to their indexed document and chunk IDs.

When a source changes:

- Identify its new content hash.
- Rebuild its logical documents and chunks.
- Index and verify the replacement version.
- Remove stale chunks belonging only to that source after successful replacement.
- Leave unrelated sources untouched.

If atomic replacement is not guaranteed, implement the safest practical staging strategy and document the limitations.

## 13. Ingestion Orchestration

Implement `pipeline.py` with this order:

```text
1. Discover source files
2. Validate files and schemas
3. Load and extract records
4. Clean and normalize
5. Validate data quality and relationships
6. Form logical documents
7. Enrich metadata
8. Apply source-aware chunking
9. Generate stable document and chunk IDs
10. Detect duplicates
11. Generate local BGE embeddings
12. Validate Qdrant configuration
13. Upsert chunks into Qdrant
14. Verify indexing and retrieval
15. Generate ingestion and validation reports
```

Each stage must have a clear input/output contract and log its results.

Provide a dry-run mode that performs discovery, validation, document formation, metadata enrichment, and chunking without modifying Qdrant.

Provide an option to process a single source file or source directory.

## 14. Reports and Inspection

Create a timestamped JSON report and a readable summary.

Include:

- All discovered source filenames.
- Expected versus discovered source counts.
- Record counts by source.
- Logical document counts.
- Chunk counts and size distributions.
- Missing and invalid metadata counts.
- Relationship consistency errors.
- Conflicting identifier counts.
- Duplicate counts.
- Quarantined records.
- Embedding model and dimensions.
- Number of successfully indexed points.
- Number of failed, skipped, inserted, or updated points.
- Final Qdrant collection count.
- Smoke-test results.
- Errors and warnings.

Implement `inspect_documents.py` to display representative documents and metadata before indexing.

Implement `inspect_collection.py` to show the collection configuration, point count, representative payloads, and a smoke-test search.

Implement `reset_collection.py` with an explicit confirmation requirement. Never run it automatically during ingestion.

## 15. Tests

Use `pytest` to test:

- CSV schema discovery and row preservation.
- PDF page provenance.
- Log file discovery across all 10 files.
- Log event boundaries and line numbers.
- JSON parsing across all 10 files.
- JSON array and JSONL handling.
- Metadata mapping for every source type.
- Metadata propagation across chunks.
- Relationship consistency checks.
- Handling of conflicting screenshot/defect/execution IDs.
- Deterministic IDs and content hashes.
- Deduplication without losing legitimate repeated events.
- Embedding dimensions and numerical validity.
- Qdrant collection configuration.
- Idempotent indexing.
- Safe source-specific re-ingestion.
- Accurate reporting when a batch fails.

Use small fixtures for unit tests and a local Qdrant instance for integration tests when available. Tests must not require OpenAI credentials.

## 16. Dependencies and Environment

Create a compatible `requirements.txt` containing the necessary packages, including:

- `langchain`
- `langchain-community`
- `langchain-text-splitters`
- `langchain-huggingface`
- `langchain-qdrant`
- `qdrant-client`
- `sentence-transformers`
- `pymupdf`
- `pandas`
- `python-dotenv`
- `pytest`

Resolve version compatibility and document installation steps.

Create `.env.example` without secrets. Add `.env`, caches, generated data, temporary files, and virtual environments to `.gitignore`.

## 17. Acceptance Criteria

The implementation is complete only when:

1. All six logical source categories are discovered and processed.
2. Both 10-file folders are handled correctly.
3. The actual source schemas are inspected rather than assumed.
4. Metadata is source-specific and validated.
5. All original QA identifiers and source provenance are retained.
6. Conflicting references are flagged rather than silently corrected.
7. Logical documents are formed before chunking.
8. Chunks retain their parent document and source identifiers.
9. BGE embeddings are generated locally without OpenAI API calls.
10. Vectors and payloads are successfully indexed in Qdrant.
11. Re-running unchanged data does not create duplicate points.
12. Changed sources can be reindexed without deleting unrelated data.
13. Reports reflect actual processing and indexing results.
14. Tests pass, or any failures are explicitly reported.
15. Qdrant point counts and smoke-test retrieval are verified.

## 18. Final Execution Instructions

Follow these steps:

1. Inspect the existing repository and source files.
2. Print the actual source directory structure and all discovered filenames.
3. Inspect CSV headers and representative records.
4. Inspect the SRS structure and representative pages.
5. Inspect representative log entries from the 10 log files.
6. Inspect representative JSON objects from the 10 screenshot files.
7. Finalize the metadata mapping based on those actual structures.
8. Implement the project modules and tests.
9. Run validation and dry-run.
10. Run ingestion only when the Qdrant service is available.
11. Verify the actual collection point count and run a smoke-test search.
12. Report the files created or modified, tests executed, metadata schema, document/chunk counts, indexed point count, and any remaining issues.

Do not fabricate schema details, test results, or successful indexing. If any source file is unavailable, state exactly which file or directory is missing and continue only with supported inputs.

**Final deliverable:** a working, tested, metadata-aware Vanilla RAG ingestion pipeline using LangChain, local BGE embeddings, and persistent Qdrant, with accurate cross-file QA traceability and explicit support for 10 failure-log files and 10 screenshot/OCR JSON files.