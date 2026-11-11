# Task: Build a Complete RAG Evaluation Pipeline for Existing VanillaRAG Project

## 1. Project Objective

I have an existing VanillaRAG application with a complete retrieval and answer-generation pipeline.

I want to create a dedicated `evaluation/` folder inside my existing VanillaRAG project to evaluate the performance, retrieval quality, answer quality, reliability, and execution efficiency of the RAG application.

**IMPORTANT:**
- Do not rebuild the RAG application.
- Do not modify the existing retrieval architecture unnecessarily.
- Do not replace existing retrievers, rerankers, query routers, or answer generators.
- Reuse the existing project modules and functions.
- First inspect the entire project structure and understand how the existing pipeline works.
- Generate the evaluation code only. Do not execute the 200-query evaluation pipeline.

The evaluation must use my existing 200-query JSON evaluation dataset.

## 2. First: Inspect the Existing Project

Before writing any code, inspect the following:

1. Project folder structure.
2. Configuration files.
3. Query router implementation.
4. Dense retrieval implementation.
5. Sparse retrieval implementation.
6. Hybrid retrieval implementation.
7. Reranker implementation.
8. Evidence selection and context construction.
9. Answer generator implementation.
10. Qdrant collection configuration.
11. Metadata filtering and retrieval logic.
12. Existing logging and error-handling mechanisms.
13. Existing evaluation-related files, if any.

Understand the actual function names, parameters, return types, data structures, and dependencies.

Do not assume function names or invent interfaces.

Create an internal integration plan showing which existing functions will be reused by each evaluation module.

If an existing implementation has limitations, document them instead of silently changing its behavior.

## 3. Evaluation Dataset

My evaluation dataset contains 200 queries in JSON format.

Example:

{
  "id": "Q002",
  "query": "What version and status are stated on the title page of the Odoo ERP System SRS?",
  "expected_answer": "Version 1.0 — Baseline draft, with status 'For stakeholder review and approval'.",
  "category": "Basic Retrieval"
}

Dataset fields:

- `id`: Unique query identifier.
- `query`: User question.
- `expected_answer`: Ground-truth answer.
- `category`: Query classification.

Requirements:

1. Load the existing JSON dataset.
2. Validate the dataset schema.
3. Check for duplicate query IDs.
4. Check for missing questions or expected answers.
5. Preserve all 200 queries.
6. Do not modify the original dataset.
7. Do not pass expected answers to the retrieval or generation pipeline.
8. Use expected answers exclusively during evaluation.
9. Preserve the original query IDs and categories in all result files.

Support category-wise evaluation in addition to overall evaluation.

## 4. Evaluation Architecture

The evaluation must follow the actual existing VanillaRAG execution architecture.

The application uses dynamic retrieval selection based on the type and characteristics of the incoming query.

The execution flow must be:

Evaluation Dataset
    |
    v
Load Evaluation Query
    |
    v
Existing Query Router
    |
    v
Select Appropriate Retrieval Strategy
    |
    +---- Dense Retrieval
    |
    +---- Sparse Retrieval
    |
    +---- Hybrid Retrieval
    |
    v
Existing Metadata Filtering
    |
    v
Existing Retrieval Pipeline
    |
    v
Existing Reranker
    |
    v
Evidence Selection
    |
    v
Existing Answer Generator
    |
    v
Collect Evaluation Outputs
    |
    v
RAG Evaluation Framework
    |
    v
Metrics and Reports

IMPORTANT:

- The router must determine the retrieval strategy.
- Execute only the selected retrieval strategy for each query.
- Do not execute all three retrieval strategies for every query.
- Do not manually assign retrieval strategies based on the evaluation category.
- Record the actual retrieval strategy selected by the router.
- Preserve the existing fallback behavior for insufficient metadata matches and insufficient evidence.
- Do not bypass metadata filtering, reranking, or evidence selection.
- Do not introduce a new retrieval implementation.

## 5. Evaluation Metrics

Use the RAGAS framework wherever appropriate.

Separate evaluation into the following categories.

### A. Retrieval Evaluation

Calculate:

1. Context Precision
   - Measures whether retrieved contexts are relevant to the query.
   - Evaluate the relevance and ranking of retrieved evidence.

2. Context Recall
   - Measures whether retrieved contexts contain the information necessary to answer the query.
   - Compare retrieved evidence against the reference answer.

3. Retrieval Success Rate
   - Percentage of queries for which relevant evidence is retrieved.

4. Empty Retrieval Rate
   - Percentage of queries returning no usable evidence.

5. Insufficient Evidence Rate
   - Percentage of queries for which the application correctly identifies insufficient evidence.

### B. Generation Evaluation

Calculate:

1. Faithfulness
   - Determine whether the generated answer is supported by retrieved contexts.
   - Penalize unsupported claims and hallucinations.

2. Answer Relevancy
   - Determine whether the generated answer addresses the original question.

3. Answer Correctness
   - Compare generated answers against expected answers.
   - Evaluate factual correctness, not just textual similarity.

### C. Performance Evaluation

Record:

- Query routing latency.
- Metadata filtering latency.
- Retrieval latency.
- Reranking latency.
- Evidence construction latency.
- Answer generation latency.
- Evaluation latency.
- Total end-to-end latency.

Also calculate:

- Average latency.
- Median latency.
- P90 latency.
- P95 latency.
- Minimum latency.
- Maximum latency.

Do not confuse evaluation-model latency with actual RAG application latency.

## 6. LLM Rate-Limit Handling and Retry Fallback

Implement robust error handling for model API rate limits and temporary service failures.

This is particularly important because evaluation involves 200 queries and can generate many model requests.

Requirements:

1. Detect HTTP 429 rate-limit errors.
2. Detect temporary server errors such as 500, 502, 503 and 504.
3. Detect connection timeouts and temporary network failures.
4. If a rate limit is reached:
   - Read the Retry-After header if available.
   - Otherwise use an appropriate exponential backoff strategy.
   - Wait for the specified duration.
   - Retry the same request.
5. Implement configurable maximum retry attempts.
6. Add a configurable delay between retries.
7. Add exponential backoff with a maximum wait limit and optional jitter.
8. Log every retry attempt, failure reason, and waiting duration.
9. Do not repeatedly send requests while the rate limit is active.

Fallback strategy:

- First attempt: Execute using the configured primary model.
- If rate limited: Wait and retry.
- If the retry attempts are exhausted: Switch to a configured fallback model, if available.
- If the fallback model also encounters a temporary failure: Apply its own retry policy.
- If all configured models fail: Save the query state and mark it as temporarily failed.
- Never silently replace a failed LLM call with a fabricated answer.
- Do not classify a failed generation request as a successful evaluation.
- Resume failed queries later without repeating already completed queries.

Make model configuration independent of the evaluation code.

Support both remote model endpoints and locally loaded models through the existing answer-generation interface.

Do not introduce a new model-loading mechanism unless the existing architecture requires it.

## 7. Checkpointing and Resume Support

The evaluation must support all 200 queries without losing completed work.

Implement:

- Automatic checkpointing.
- Query-level completion tracking.
- Resume from the last successfully completed query.
- Failed-query tracking.
- Retry of failed queries.
- Prevention of duplicate evaluation records.
- Periodic saving of partial results.

If the runtime disconnects after 75 queries, the system should resume from the remaining queries rather than restarting all 200.

Maintain separate statuses:

- PENDING
- ROUTING_COMPLETED
- RETRIEVAL_COMPLETED
- GENERATION_COMPLETED
- EVALUATION_COMPLETED
- FAILED
- RATE_LIMITED

Persist sufficient intermediate information to resume safely.

Avoid losing retrieval outputs when only answer generation or metric evaluation fails.

## 8. Evaluation Folder Structure

Create a clean and modular folder structure.

Suggested structure:

evaluation/
│
├── __init__.py
│
├── config/
│   ├── __init__.py
│   └── evaluation_config.py
│
├── dataset/
│   ├── __init__.py
│   └── dataset_loader.py
│
├── pipeline/
│   ├── __init__.py
│   ├── evaluation_pipeline.py
│   ├── retrieval_evaluator.py
│   ├── generation_evaluator.py
│   └── performance_evaluator.py
│
├── metrics/
│   ├── __init__.py
│   ├── retrieval_metrics.py
│   ├── generation_metrics.py
│   └── aggregate_metrics.py
│
├── reliability/
│   ├── __init__.py
│   ├── retry_handler.py
│   ├── fallback_handler.py
│   └── checkpoint_manager.py
│
├── reporting/
│   ├── __init__.py
│   ├── report_generator.py
│   ├── category_analysis.py
│   └── visualization.py
│
├── utils/
│   ├── __init__.py
│   ├── logging_utils.py
│   ├── timing_utils.py
│   └── validation.py
│
├── results/
│   ├── checkpoints/
│   ├── per_query/
│   ├── metrics/
│   ├── reports/
│   └── charts/
│
├── run_evaluation.py
│
└── requirements-evaluation.txt

Modify this structure if the existing project architecture suggests a better organization.

Do not create unnecessary duplicate modules.

## 9. Per-Query Result Storage

For every evaluation query, save the following information:

{
  "id": "Q002",
  "category": "Basic Retrieval",
  "query": "What version and status are stated on the title page of the Odoo ERP System SRS?",

  "expected_answer": "Version 1.0 — Baseline draft, with status 'For stakeholder review and approval'.",

  "routing": {
    "selected_strategy": null,
    "routing_latency": null
  },

  "retrieval": {
    "candidate_count": null,
    "retrieved_chunk_ids": [],
    "retrieved_contexts": [],
    "retrieval_latency": null
  },

  "reranking": {
    "reranked_chunk_ids": [],
    "selected_contexts": [],
    "reranking_latency": null
  },

  "generation": {
    "generated_answer": null,
    "model": null,
    "generation_latency": null,
    "status": null
  },

  "evaluation": {
    "context_precision": null,
    "context_recall": null,
    "faithfulness": null,
    "answer_relevancy": null,
    "answer_correctness": null
  },

  "performance": {
    "total_latency": null
  },

  "execution_status": "PENDING",
  "error": null
}

This is a suggested schema. Adapt it to the actual project outputs.

Save per-query records in JSONL or another efficient format.

Keep detailed retrieval provenance wherever possible, including chunk IDs, source metadata, and selected evidence.

Do not discard retrieved contexts after generation.

## 10. Evaluation Execution Modes

Provide configurable execution options:

1. Run all 200 queries.
2. Run a limited number of queries for testing.
3. Run a selected list of query IDs.
4. Resume an interrupted evaluation.
5. Retry failed queries.
6. Run metric calculation on previously saved results without rerunning retrieval or generation.

Provide command-line arguments or configuration options.

Example commands:

python -m evaluation.run_evaluation --limit 5

python -m evaluation.run_evaluation --all

python -m evaluation.run_evaluation --resume

python -m evaluation.run_evaluation --retry-failed

python -m evaluation.run_evaluation --metrics-only

The commands must actually work with the implemented configuration and entry point.

## 11. Category-Wise Evaluation

The evaluation dataset contains different query categories.

Preserve the category of each query and calculate:

- Number of queries per category.
- Average Context Precision.
- Average Context Recall.
- Average Faithfulness.
- Average Answer Relevancy.
- Average Answer Correctness.
- Average retrieval latency.
- Average generation latency.

Do not assume that every category has the same number of queries.

Generate category-wise performance reports to identify weaknesses in specific query types.

## 12. Final Reports and Visualizations

Generate the following outputs:

### A. Overall Metrics

- Overall retrieval metrics.
- Overall generation metrics.
- Overall performance metrics.
- Total successful queries.
- Failed queries.
- Incomplete queries.
- Rate-limit events.
- Fallback model usage.

### B. Category-Wise Metrics

Compare the evaluation metrics across all query categories.

### C. Retrieval Strategy Analysis

Since the application dynamically chooses a retrieval strategy, calculate:

- Number of Dense queries.
- Number of Sparse queries.
- Number of Hybrid queries.
- Retrieval success rate per strategy.
- Context Precision per strategy.
- Context Recall per strategy.
- Average retrieval latency per strategy.
- Generation quality associated with each strategy.

Do not force queries into a strategy. Use the actual routing decisions recorded during execution.

### D. Visualizations

Generate:

- Metric comparison charts.
- Category-wise performance charts.
- Retrieval strategy distribution.
- Latency distribution.
- Retrieval and generation latency comparison.
- Query success and failure distribution.

Save charts as PNG files.

### E. Export Formats

Generate:

- Overall metrics JSON.
- Per-query results JSONL.
- Category-wise metrics CSV.
- Latency analysis CSV.
- Final evaluation report in Markdown.
- Charts in PNG format.

## 13. Reliability and Code Quality

Follow these requirements:

- Modular Python implementation.
- Type hints wherever practical.
- Proper exception handling.
- Structured logging.
- No hardcoded machine-specific paths.
- Configurable model settings.
- Configurable retry limits and waiting periods.
- Configurable batch sizes.
- No unnecessary concurrent generation requests.
- No accidental duplicate processing.
- No fabricated metric values.
- Handle missing retrieved contexts correctly.
- Handle empty answers correctly.
- Handle insufficient evidence correctly.
- Handle partial evaluation failures correctly.
- Ensure checkpoint files are written safely.
- Keep original project configuration unchanged unless absolutely necessary.

Use reproducible evaluation settings wherever possible.

## 14. Testing Before Actual Evaluation

Create unit tests and small integration tests.

Test:

1. Dataset loading.
2. Dataset validation.
3. Existing query-router integration.
4. Dense retrieval integration.
5. Sparse retrieval integration.
6. Hybrid retrieval integration.
7. Dynamic strategy selection.
8. Reranker integration.
9. Answer-generator integration.
10. Retry behavior.
11. Fallback behavior.
12. Checkpoint creation.
13. Resume functionality.
14. Failed-query handling.
15. Metric calculation.
16. Report generation.

Use mocked model responses where appropriate for testing retry and checkpoint behavior.

Do not launch the full 200-query evaluation as part of these tests.

## 15. Strict Implementation Instructions

Follow this development sequence:

Phase 1:
Inspect the complete existing VanillaRAG project.

Phase 2:
Document the current architecture and identify the integration points.

Phase 3:
Design the evaluation folder structure.

Phase 4:
Implement dataset loading and validation.

Phase 5:
Implement evaluation pipeline integration using existing project functions.

Phase 6:
Implement RAGAS metrics and additional performance metrics.

Phase 7:
Implement rate-limit retry, fallback, and checkpointing.

Phase 8:
Implement reporting and visualizations.

Phase 9:
Create tests and verify imports, function signatures, and integration compatibility.

Phase 10:
Provide complete execution instructions.

### Critical restrictions

1. Do not modify the existing retrieval architecture.
2. Do not introduce a separate retrieval pipeline for evaluation.
3. Do not execute Dense, Sparse and Hybrid for every query.
4. Do not use the expected answer during retrieval or generation.
5. Do not run the full evaluation automatically.
6. Do not fabricate results or evaluation metrics.
7. Do not remove existing functionality.
8. Do not silently ignore exceptions.
9. Do not lose completed query results when rate limits occur.
10. Do not generate placeholder implementations for critical modules.

## 16. Final Deliverables

After implementation, provide:

1. Complete `evaluation/` folder structure.
2. All Python source files with complete implementations.
3. Evaluation dependency file.
4. Configuration instructions.
5. Dataset location configuration.
6. Model configuration instructions.
7. Test instructions.
8. Commands for running a small evaluation.
9. Commands for running all 200 queries.
10. Resume and retry instructions.
11. Expected output folder structure.
12. Explanation of every evaluation metric.
13. A summary of which existing project files were reused.
14. A list of any required changes to existing files, with reasons.

**Final instruction:**

Generate the complete evaluation implementation and verify its integration with the existing project. Do not execute the actual 200-query evaluation. I will run the evaluation pipeline manually after reviewing the generated code.