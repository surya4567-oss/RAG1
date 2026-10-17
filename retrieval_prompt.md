# Task: Implement an Advanced Retrieval Pipeline for My Existing Vanilla RAG Project

Act as a senior Python AI engineer specializing in LangChain, Groq LLMs, Qdrant hybrid search, reranking, and evidence-grounded RAG.

Build a complete, modular, runnable retrieval pipeline by inspecting and extending my existing Vanilla RAG codebase. Implement the actual code, tests, and execution instructions—not just a template.

## 1. Existing project context

My ingestion pipeline is already implemented.

- Framework: LangChain
- Vector database: persistent local Qdrant
- Collection: `qa_knowledge`
- Dataset: approximately 123,776 chunks
- Dense embedding model: `BAAI/bge-base-en-v1.5`
- Dense vector dimension: 768
- Sparse embeddings: already stored in Qdrant
- LLM provider: Groq API
- Dataset sources: test cases CSV, test execution CSV, defects CSV, SRS PDF, failure logs, and screenshot/OCR JSON
- Metadata includes source provenance and relevant QA identifiers.

Use the existing collection and payload schema. Inspect the actual repository and Qdrant configuration before implementing retrieval.

**Do not run ingestion, regenerate embeddings, reset the collection, delete points, or modify the ingestion pipeline unnecessarily.**

## 2. Technology requirements

Use:

- LangChain for orchestration.
- `langchain-groq` for Groq LLM integration.
- `qdrant-client` for Qdrant retrieval.
- `sentence-transformers` for the local cross-encoder reranker.
- Pydantic for structured router output and final answer validation.
- The existing `BAAI/bge-base-en-v1.5` model for query embeddings.
- The existing sparse-vector representation and configured sparse model from Qdrant ingestion.

Use the existing Groq model configuration if available. Otherwise, make the model configurable through environment variables. Do not hardcode API keys.

Do not use OpenAI APIs or OpenAI embeddings.

## 3. Required architecture

Implement the following flow:

User query → Groq LLM query router → selected retrieval strategy → candidate deduplication and RRF when applicable → cross-encoder reranker → evidence selection → Groq answer generation → validated structured output.

The router must choose among:

- `keyword`
- `dense`
- `hybrid`

Do not implement a separate query-understanding module. Query understanding, intent classification, identifier extraction, and retrieval strategy selection must be handled by the LLM router.

## 4. LLM query router

Implement `src/retrieval/router.py`.

The router must:

1. Accept the original user question.
2. Identify the query intent.
3. Extract explicit identifiers when present, such as `TC-000646`, `BUG-000962`, execution IDs, build numbers, and error codes.
4. Extract supported metadata filters only when the question explicitly supports them.
5. Select `keyword`, `dense`, or `hybrid`.
6. Return a validated Pydantic model.
7. Use a controlled fallback route if Groq fails or returns invalid output.

Suggested routing behavior:

- Exact test case, defect, execution, or error-code lookup → `keyword`, with exact metadata filters where possible.
- Conceptual or explanatory question → `dense`.
- Questions involving both exact identifiers and semantic explanations, cross-source evidence, or multiple related records → `hybrid`.
- Ambiguous or complex questions → `hybrid` as the fallback.

Use structured output or JSON mode supported by the configured Groq model. If native structured output is unavailable, validate JSON with Pydantic and handle parsing failures safely.

Do not let the LLM invent field names or filters. Validate proposed fields against the actual collection schema.

## 5. Keyword retrieval

Implement `src/retrieval/keyword_retriever.py`.

Use the existing sparse vectors in Qdrant for sparse retrieval, if they are confirmed to be available and correctly configured.

Requirements:

- Use Qdrant's supported sparse-vector query API.
- Use the actual sparse vector name from the collection configuration.
- Support exact metadata filtering for known identifiers where appropriate.
- Preserve payload text and provenance.
- Return a normalized internal result format.
- Handle empty results and missing metadata.
- Do not assume sparse retrieval guarantees literal exact matching.

If true full-text keyword search is required and is not supported by the current Qdrant schema, document the gap and implement only a compatible solution. Do not silently substitute dense search and call it keyword search.

## 6. Dense retrieval

Implement `src/retrieval/dense_retriever.py`.

Requirements:

- Load the existing `BAAI/bge-base-en-v1.5` embedding model locally.
- Embed the original query without regenerating document embeddings.
- Use the existing dense vector name and 768-dimensional configuration.
- Search Qdrant for the configured number of candidates.
- Support validated metadata filters.
- Preserve payload and provenance.
- Avoid repeatedly loading the model for every query.
- Return the same result format used by keyword retrieval.

## 7. Hybrid retrieval and RRF

Implement `src/retrieval/hybrid_retriever.py` and `src/retrieval/fusion.py`.

For the `hybrid` route:

1. Run dense and sparse retrieval.
2. Apply compatible metadata filters.
3. Deduplicate results using stable point IDs or chunk IDs.
4. Combine ranked lists using Reciprocal Rank Fusion (RRF).
5. Preserve each result's original rank and source metadata.
6. Return a bounded set of candidates for reranking.

Use a configurable RRF constant, initially `k=60`.

Do not add dense similarity scores directly to sparse scores. Use rank-based fusion unless a separately evaluated score-fusion method is implemented.

For the `keyword` and `dense` routes, preserve their route-specific rankings unless an explicitly justified fusion step is needed.

## 8. Cross-encoder reranker

Implement `src/retrieval/reranker.py`.

Use the local model:

`BAAI/bge-reranker-base`

Requirements:

- Load the model once and reuse it.
- Score each `(original_query, candidate_chunk_text)` pair.
- Sort candidates by descending reranker score.
- Keep the original retrieval and RRF scores as separate fields.
- Make candidate count, batch size, and final top-k configurable.
- Default to reranking up to 50 unique candidates and returning 5–10 final chunks.
- Support CPU and CUDA where available.
- Handle empty inputs and model errors gracefully.
- Do not interpret reranker scores as calibrated probabilities.

The reranker must operate after candidate retrieval and fusion.

## 9. Evidence-grounded answer generation

Implement `src/retrieval/answer_generator.py`.

Use the configured Groq model to answer the original user question using only the retrieved evidence.

Requirements:

- Preserve the original question throughout the pipeline.
- Provide the selected chunks with their source metadata to the LLM.
- Instruct the model not to invent facts, IDs, relationships, filenames, or citations.
- Distinguish source-supported facts from uncertainty and conflicting records.
- If evidence is insufficient, return an appropriate insufficient-evidence status.
- Include source evidence for every substantive factual claim.
- Do not expose hidden chain-of-thought. Return concise explanations and evidence-based conclusions only.

Do not let hypothetical or inferred relationships become verified facts.

## 10. Validated structured output

Implement `src/retrieval/schemas.py` using Pydantic.

Use a response schema similar to:

```python
class Evidence(BaseModel):
    chunk_id: str
    document_type: str | None = None
    source_file: str | None = None
    record_id: str | None = None
    supporting_text: str

class RetrievalResponse(BaseModel):
    question: str
    answer: str
    status: str
    retrieval_route: str
    evidence: list[Evidence]
    limitations: list[str]
```

You may refine this schema to match the actual metadata and add a `conflicts` field if appropriate.

Requirements:

- `status` must use a controlled set of values, such as `answered`, `insufficient_evidence`, and `conflicting_evidence`.
- `retrieval_route` must be one of `keyword`, `dense`, or `hybrid`.
- Every evidence item must reference a chunk actually returned by retrieval.
- Validate chunk IDs and source metadata against the retrieved candidate set.
- Ensure `supporting_text` is grounded in the selected chunk, not fabricated by the LLM.
- Return a machine-readable JSON object.
- Validate the final response using Pydantic.
- Handle invalid LLM output through bounded retries or safe error handling.
- Do not silently fabricate evidence to satisfy required fields.

## 11. Retrieval orchestration

Implement `src/retrieval/pipeline.py` to orchestrate:

1. Receive the user question.
2. Call the Groq LLM router.
3. Validate the selected route and filters.
4. Execute the chosen retrieval strategy.
5. Fuse results with RRF for hybrid retrieval.
6. Deduplicate candidates.
7. Rerank candidates using `BAAI/bge-reranker-base`.
8. Select the final evidence chunks.
9. Generate the answer with Groq.
10. Validate the structured response and evidence references.
11. Return the validated response.

Keep routing, retrieval, fusion, reranking, and answer generation in separate modules.

## 12. Configuration and cost control

Extend the existing `.env.example` without exposing secrets.

Use configurable values for:

- `GROQ_API_KEY`
- `GROQ_MODEL`
- `QDRANT_COLLECTION=qa_knowledge`
- Qdrant URL or persistent local path
- Dense vector name
- Sparse vector name
- `EMBEDDING_MODEL=BAAI/bge-base-en-v1.5`
- `RERANKER_MODEL=BAAI/bge-reranker-base`
- Dense top-k
- Sparse top-k
- RRF constant
- Reranker candidate count
- Final evidence top-k
- Reranker batch size

Use one router call and one answer-generation call per normal request when possible. Do not call Groq separately for every retrieved chunk. Reuse loaded local models.

Do not invent a model name or assume the Groq account supports a specific model. Validate the configured model and document how to change it.

## 13. Logging and error handling

Log:

- Router decision and selected route.
- Extracted identifiers and validated filters.
- Number of dense and sparse candidates.
- Fusion and deduplication counts.
- Reranker duration.
- Selected evidence chunk IDs.
- Answer-generation duration.
- Token usage and latency when exposed by the API.
- Failures and fallback decisions.

Never log API keys or other secrets.

If sparse retrieval fails, use an explicit fallback and record the degradation. Never label dense-only results as hybrid results.

## 14. Testing and verification

Create tests for:

- Router output validation and fallback behavior.
- Identifier extraction.
- Filter validation against the actual schema.
- Dense retrieval.
- Sparse retrieval.
- RRF ranking and duplicate removal.
- Reranker ordering.
- Evidence citation validation.
- Structured output validation.
- Empty retrieval results.
- Conflicting source evidence.
- Groq API errors and rate limits.
- Preservation of source provenance.

Use mocked Groq responses and a test collection for unit tests. Do not modify or reset the existing populated collection during testing.

Provide example queries covering exact IDs, error codes, conceptual questions, and cross-source relationships.

## 15. Deliverables

Provide:

- Complete retrieval modules.
- Updated dependencies and `.env.example`.
- Tests and example queries.
- Commands to run the retrieval pipeline.
- Example validated JSON responses.
- A report identifying actual collection vector names and supported search modes.

First inspect the existing repository and Qdrant schema. Then implement the pipeline in the current codebase, reusing existing components wherever possible.

Do not modify the ingestion pipeline or regenerate dense/sparse embeddings. Do not claim successful retrieval or testing unless it has actually been run and verified.