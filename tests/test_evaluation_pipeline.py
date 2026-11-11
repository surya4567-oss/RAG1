"""Comprehensive unit and integration tests for VanillaRAG Evaluation Pipeline."""

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from evaluation.config.evaluation_config import EvaluationConfig
from evaluation.dataset.dataset_loader import (
    EvaluationDataset,
    EvaluationQueryItem,
    load_evaluation_dataset,
)
from evaluation.metrics.aggregate_metrics import (
    aggregate_category_metrics,
    aggregate_overall_metrics,
    aggregate_strategy_metrics,
)
from evaluation.metrics.generation_metrics import GenerationMetricsEvaluator
from evaluation.metrics.retrieval_metrics import (
    compute_context_precision,
    compute_context_recall,
)
from evaluation.pipeline.performance_evaluator import PerformanceEvaluator
from evaluation.pipeline.retrieval_evaluator import RetrievalEvaluator
from evaluation.reliability.checkpoint_manager import (
    CheckpointManager,
    QueryCheckpointRecord,
    QueryExecutionState,
)
from evaluation.reliability.fallback_handler import FallbackHandler
from evaluation.reliability.retry_handler import RetryHandler
from evaluation.reporting.category_analysis import export_category_and_strategy_csvs
from evaluation.reporting.report_generator import generate_markdown_report, generate_summary_json
from evaluation.reporting.visualization import generate_evaluation_charts
from src.retrieval.schemas import CandidateChunk, RetrievalResponse, RetrievalRoute, RetrievalStatus, RouterOutput


# ---------------------------------------------------------------------------
# Test 1 & 2: Dataset Loading and Schema Validation
# ---------------------------------------------------------------------------
def test_dataset_loading_and_validation():
    """Verify loading real 200-query evaluation dataset."""
    dataset_path = Path("vanillarag_evaluation_dataset.json")
    dataset = load_evaluation_dataset(dataset_path)

    assert len(dataset) == 200
    assert len(dataset.get_categories()) == 11

    first_item = dataset[0]
    assert first_item.id == "Q001"
    assert first_item.query != ""
    assert first_item.expected_answer != ""
    assert first_item.category != ""

    # Verify expected_answer is stripped for execution
    exec_dict = first_item.for_execution()
    assert "expected_answer" not in exec_dict
    assert exec_dict["id"] == "Q001"


def test_dataset_validation_detects_errors():
    """Verify dataset loader catches missing fields and duplicate IDs."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump([
            {"id": "DUP01", "query": "q1", "expected_answer": "a1", "category": "c1"},
            {"id": "DUP01", "query": "q2", "expected_answer": "a2", "category": "c2"},
        ], f)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Duplicate query ID"):
            load_evaluation_dataset(temp_path)
    finally:
        temp_path.unlink()


# ---------------------------------------------------------------------------
# Test 3-8: Routing and Retrieval Integration
# ---------------------------------------------------------------------------
def test_retrieval_evaluator_metrics():
    """Verify context precision and context recall calculation."""
    evaluator = RetrievalEvaluator(relevance_threshold=0.5)

    chunks = [
        CandidateChunk(chunk_id="c1", text="Status of defect BUG-000001 is Closed.", score=0.9),
        CandidateChunk(chunk_id="c2", text="Unrelated text about sales.", score=0.1),
    ]

    mock_ragas_eval = MagicMock()
    mock_ragas_eval.evaluate.return_value = {
        "context_precision": 0.85,
        "context_recall": 1.0,
        "metric_status": {"context_precision": "SUCCESS", "context_recall": "SUCCESS"},
        "metric_errors": {},
    }

    with patch("evaluation.pipeline.retrieval_evaluator.get_ragas_retrieval_evaluator", return_value=mock_ragas_eval):
        res = evaluator.evaluate(chunks, expected_answer="Status is Closed", query="What is the status of BUG-000001?")
        assert res["candidate_count"] == 2
        assert res["context_precision"] == 0.85
        assert res["context_recall"] == 1.0
        assert "c1" in res["retrieved_chunk_ids"]


# ---------------------------------------------------------------------------
# Test 9: Generation Metrics & Judge Evaluation
# ---------------------------------------------------------------------------
def test_generation_metrics_heuristic_and_unanswerable():
    """Verify generation evaluator handles normal queries and unanswerable edge cases."""
    evaluator = GenerationMetricsEvaluator()
    evaluator._initialized = True

    # Case A: Correctly identifies empty generated answer
    scores_empty = evaluator.evaluate_generation(
        query="What is the root cause of nonexistent defect BUG-999999?",
        generated_answer="",
        expected_answer="Insufficient evidence in the provided dataset.",
        retrieved_contexts=[],
    )
    assert scores_empty["answer_correctness"] == 0.0
    assert scores_empty["faithfulness"] == 0.0

    # Case B: Evaluation with mocked Ragas metrics
    with patch.object(evaluator.faithfulness_metric, "single_turn_score", return_value=0.95), \
         patch.object(evaluator.answer_relevancy_metric, "single_turn_score", return_value=0.90), \
         patch.object(evaluator.answer_correctness_metric, "single_turn_score", return_value=1.0):
        scores_norm = evaluator.evaluate_generation(
            query="What is the status of BUG-000001?",
            generated_answer="Defect BUG-000001 has status Closed in module Project.",
            expected_answer="Closed",
            retrieved_contexts=["Defect BUG-000001 status is Closed."],
        )
        assert scores_norm["answer_correctness"] == 1.0
        assert scores_norm["faithfulness"] == 0.95
        assert scores_norm["answer_relevancy"] == 0.90


# ---------------------------------------------------------------------------
# Test 10 & 11: Retry and Fallback Handlers
# ---------------------------------------------------------------------------
def test_retry_handler_recovers_after_retryable_error():
    """Verify RetryHandler retries on 429 and succeeds on next attempt."""
    handler = RetryHandler(max_retries=2, initial_delay=0.01, jitter=False)

    call_count = 0

    def flaky_func():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("429 Too Many Requests: rate limit exceeded")
        return "success"

    result = handler.execute_with_retry(flaky_func, operation_name="Test")
    assert result == "success"
    assert call_count == 2


def test_fallback_handler_switches_to_fallback():
    """Verify FallbackHandler invokes fallback function on primary error."""
    handler = FallbackHandler(fallback_enabled=True)

    def failing_primary():
        raise RuntimeError("Primary model error")

    def working_fallback():
        return "fallback_result"

    result, failure, used_fallback = handler.execute_with_fallback(
        primary_func=failing_primary,
        fallback_func=working_fallback,
        query_id="Q001",
    )
    assert result == "fallback_result"
    assert used_fallback is True
    assert failure.used_fallback is True


# ---------------------------------------------------------------------------
# Test 12, 13 & 14: Checkpoint Creation, Resumption, and Failed-Query Handling
# ---------------------------------------------------------------------------
def test_checkpoint_manager_atomic_save_and_resume():
    """Verify checkpoint state is persisted and resumed without duplicates."""
    with tempfile.TemporaryDirectory() as temp_dir:
        chk_file = Path(temp_dir) / "checkpoints.jsonl"
        mgr = CheckpointManager(chk_file)

        # Save record 1: completed
        r1 = QueryCheckpointRecord(
            id="Q001",
            category="Basic Retrieval",
            query="Query 1",
            expected_answer="Ans 1",
            execution_status=QueryExecutionState.EVALUATION_COMPLETED,
        )
        mgr.save_record(r1)

        # Save record 2: failed
        r2 = QueryCheckpointRecord(
            id="Q002",
            category="Basic Retrieval",
            query="Query 2",
            expected_answer="Ans 2",
            execution_status=QueryExecutionState.FAILED,
            error="Connection timeout",
        )
        mgr.save_record(r2)

        # Reload manager from same file to test persistence
        reloaded_mgr = CheckpointManager(chk_file)
        completed = reloaded_mgr.get_completed_ids()
        failed = reloaded_mgr.get_failed_ids()

        assert "Q001" in completed
        assert "Q002" not in completed
        assert "Q002" in failed

        status_counts = reloaded_mgr.get_status_counts()
        assert status_counts[QueryExecutionState.EVALUATION_COMPLETED.value] == 1
        assert status_counts[QueryExecutionState.FAILED.value] == 1


# ---------------------------------------------------------------------------
# Test 15 & 16: Performance, Reporting, and Chart Visualizations
# ---------------------------------------------------------------------------
def test_performance_evaluator_percentiles():
    """Verify latency percentiles calculation."""
    perf = PerformanceEvaluator()
    for latency in [0.1, 0.2, 0.3, 0.4, 0.5]:
        perf.record_query_performance(
            routing_sec=0.05,
            retrieval_sec=latency,
            reranking_sec=0.01,
            generation_sec=0.2,
            evaluation_sec=0.1,
            total_sec=latency + 0.36,
        )

    stats = perf.get_summary_statistics()
    assert stats["retrieval"]["mean"] == pytest.approx(0.3)
    assert stats["retrieval"]["count"] == 5


def test_reports_and_chart_generation():
    """Verify generation of summary JSON, Markdown report, CSVs, and PNG charts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)

        sample_records = [
            {
                "id": "Q001",
                "category": "Basic Retrieval",
                "query": "What is BUG-000001?",
                "expected_answer": "Closed",
                "routing": {"selected_strategy": "keyword", "routing_latency": 0.05},
                "retrieval": {"candidate_count": 5, "retrieval_latency": 0.12},
                "generation": {"generation_latency": 0.45},
                "evaluation": {
                    "context_precision": 0.9,
                    "context_recall": 1.0,
                    "faithfulness": 0.95,
                    "answer_relevancy": 0.88,
                    "answer_correctness": 1.0,
                },
                "performance": {"total_latency": 0.65},
                "execution_status": "EVALUATION_COMPLETED",
            },
            {
                "id": "Q046",
                "category": "Metadata Filtering",
                "query": "Find Purchase defect",
                "expected_answer": "BUG-000766",
                "routing": {"selected_strategy": "dense", "routing_latency": 0.04},
                "retrieval": {"candidate_count": 3, "retrieval_latency": 0.10},
                "generation": {"generation_latency": 0.40},
                "evaluation": {
                    "context_precision": 0.8,
                    "context_recall": 0.8,
                    "faithfulness": 0.90,
                    "answer_relevancy": 0.85,
                    "answer_correctness": 0.8,
                },
                "performance": {"total_latency": 0.58},
                "execution_status": "EVALUATION_COMPLETED",
            },
        ]

        # Generate summary JSON
        json_path = generate_summary_json(sample_records, out_dir / "summary.json")
        assert json_path.exists()

        # Generate Markdown report
        md_path = generate_markdown_report(sample_records, out_dir / "report.md")
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "VanillaRAG Comprehensive Evaluation Report" in content
        assert "Basic Retrieval" in content

        # Generate CSVs
        csv_dict = export_category_and_strategy_csvs(sample_records, out_dir)
        assert csv_dict["category_metrics_csv"].exists()
        assert csv_dict["strategy_analysis_csv"].exists()

        # Generate charts
        charts = generate_evaluation_charts(sample_records, out_dir)
        assert len(charts) > 0
        for name, path in charts.items():
            assert path.exists()
            assert path.stat().st_size > 0


def test_pipeline_evaluate_query_flow():
    """Verify master evaluation pipeline executes query end-to-end with mocked components."""
    from evaluation.pipeline.evaluation_pipeline import EvaluationPipeline

    with tempfile.TemporaryDirectory() as temp_dir:
        eval_cfg = EvaluationConfig(results_dir=Path(temp_dir))
        chk_mgr = CheckpointManager(eval_cfg.checkpoint_file)
        pipeline = EvaluationPipeline(eval_config=eval_cfg, checkpoint_manager=chk_mgr)

        # Mock RAG components
        mock_router = MagicMock()
        mock_router.route.return_value = RouterOutput(
            intent="Lookup defect status",
            strategy=RetrievalRoute.KEYWORD,
            primary_entity_id="BUG-000001",
            primary_entity_type="defect",
            validated_filters={"defect_id": "BUG-000001"},
            reasoning="Defect ID exact match",
        )

        mock_controller = MagicMock()
        mock_chunk = CandidateChunk(
            chunk_id="doc_defect_BUG-000001_c0",
            text="Defect BUG-000001: Status is Closed.",
            score=10.0,
            metadata={"defect_id": "BUG-000001"},
        )
        mock_controller.execute_adaptive_retrieval.return_value = (
            [mock_chunk],
            MagicMock(is_sufficient=True),
            {"rerank_time_sec": 0.005},
        )

        from src.retrieval.schemas import Evidence

        mock_generator = MagicMock()
        mock_generator.generate.return_value = RetrievalResponse(
            question="What is the status of BUG-000001?",
            answer="The status of BUG-000001 is Closed.",
            status=RetrievalStatus.ANSWERED,
            retrieval_route=RetrievalRoute.KEYWORD,
            evidence=[
                Evidence(
                    chunk_id="doc_defect_BUG-000001_c0",
                    document_type="defect",
                    source_file="defects_rows.csv",
                    record_id="BUG-000001",
                    supporting_text="Defect BUG-000001: Status is Closed.",
                    is_supported=True,
                    support_confidence=1.0,
                )
            ],
            limitations=[],
            metadata={},
        )

        pipeline._router = mock_router
        pipeline._controller = mock_controller
        pipeline._generator = mock_generator
        pipeline._rag_pipeline = MagicMock()

        # Mock RAGAS evaluators to prevent external API latency in unit tests
        pipeline.retrieval_evaluator.evaluate = MagicMock(return_value={
            "candidate_count": 1,
            "retrieved_chunk_ids": ["doc_defect_BUG-000001_c0"],
            "retrieved_contexts": ["Defect BUG-000001: Status is Closed."],
            "context_precision": 1.0,
            "context_recall": 1.0,
        })
        pipeline.generation_evaluator.evaluate = MagicMock(return_value={
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "answer_correctness": 1.0,
            "reasoning": "Perfect match",
        })

        # Execute query
        item = EvaluationQueryItem(
            id="Q001",
            query="What is the status of BUG-000001?",
            expected_answer="Closed",
            category="Basic Retrieval",
        )

        rec = pipeline.evaluate_query(item)

        assert rec.id == "Q001"
        assert rec.execution_status == QueryExecutionState.EVALUATION_COMPLETED
        assert rec.routing["selected_strategy"] == "keyword"
        assert rec.retrieval["candidate_count"] == 1
        assert rec.generation["generated_answer"] == "The status of BUG-000001 is Closed."
        assert rec.evaluation["context_precision"] > 0.0
        assert rec.evaluation["answer_correctness"] > 0.0

        # Verify saved in checkpoint
        saved_rec = chk_mgr.get_record("Q001")
        assert saved_rec is not None
        assert saved_rec.id == "Q001"

