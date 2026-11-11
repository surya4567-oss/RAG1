"""Master Evaluation Pipeline for VanillaRAG."""

from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from evaluation.config.evaluation_config import EvaluationConfig, default_eval_config
from evaluation.dataset.dataset_loader import EvaluationQueryItem
from evaluation.pipeline.generation_evaluator import GenerationEvaluator
from evaluation.pipeline.performance_evaluator import PerformanceEvaluator
from evaluation.pipeline.retrieval_evaluator import RetrievalEvaluator
from evaluation.reliability.checkpoint_manager import (
    CheckpointManager,
    QueryCheckpointRecord,
    QueryExecutionState,
)
from evaluation.reliability.retry_handler import RetryHandler
from evaluation.utils.logging_utils import get_eval_logger
from evaluation.utils.timing_utils import StageTimer

from src.retrieval.answer_generator import EvidenceGroundedAnswerGenerator
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.controller import QueryAdaptiveRetrievalController
from src.retrieval.pipeline import AdvancedRetrievalPipeline
from src.retrieval.router import LLMQueryRouter
from src.retrieval.schemas import CandidateChunk, RetrievalResponse, RetrievalRoute, RouterOutput


class EvaluationPipeline:
    """Orchestrates end-to-end evaluation execution across routing, retrieval, generation, and metrics."""

    def __init__(
        self,
        eval_config: Optional[EvaluationConfig] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self.eval_config = eval_config or default_eval_config
        self.retrieval_config = retrieval_config or default_retrieval_config
        self.logger = get_eval_logger("vanillarag_eval_pipeline")

        # Initialize output directories
        self.eval_config.ensure_directories()

        # Checkpoint Manager
        self.checkpoints = checkpoint_manager or CheckpointManager(
            self.eval_config.checkpoint_file
        )

        # Retry Handler
        self.retry_handler = RetryHandler(
            max_retries=self.eval_config.max_retries,
            initial_delay=self.eval_config.retry_initial_delay,
            backoff_factor=self.eval_config.retry_backoff_factor,
            max_delay=self.eval_config.retry_max_delay,
            jitter=self.eval_config.retry_jitter,
        )

        # Evaluator components
        self.retrieval_evaluator = RetrievalEvaluator(
            relevance_threshold=self.eval_config.min_retrieval_relevance_threshold
        )
        self.generation_evaluator = GenerationEvaluator(config=self.retrieval_config)
        self.performance_evaluator = PerformanceEvaluator()

        # Lazily initialized RAG components
        self._rag_pipeline: Optional[AdvancedRetrievalPipeline] = None
        self._router: Optional[LLMQueryRouter] = None
        self._controller: Optional[QueryAdaptiveRetrievalController] = None
        self._generator: Optional[EvidenceGroundedAnswerGenerator] = None

    def initialize_rag_components(self) -> None:
        """Initialize production VanillaRAG modules reusing existing configurations."""
        if self._rag_pipeline is not None:
            return

        self.logger.info("Initializing VanillaRAG production components for evaluation...")
        self._rag_pipeline = AdvancedRetrievalPipeline(config=self.retrieval_config)
        self._router = self._rag_pipeline.router
        self._controller = self._rag_pipeline.controller
        self._generator = self._rag_pipeline.generator
        self.logger.info("VanillaRAG components successfully initialized.")

    def evaluate_query(self, item: EvaluationQueryItem) -> QueryCheckpointRecord:
        """Execute and evaluate a single query end-to-end with detailed telemetry and checkpointing."""
        self.initialize_rag_components()
        start_total = time.perf_counter()

        record = QueryCheckpointRecord(
            id=item.id,
            category=item.category,
            query=item.query,
            expected_answer=item.expected_answer,
            execution_status=QueryExecutionState.PENDING,
        )

        try:
            # 1. Routing Stage
            with StageTimer() as t_route:
                router_out: RouterOutput = self.retry_handler.execute_with_retry(
                    self._router.route,
                    item.query,
                    operation_name=f"Routing {item.id}",
                )
            routing_latency = t_route.elapsed

            selected_strat = router_out.strategy.value if hasattr(router_out.strategy, "value") else str(router_out.strategy)
            record.routing = {
                "selected_strategy": selected_strat,
                "routing_latency": routing_latency,
                "routing_source": getattr(router_out, "routing_source", "unknown"),
                "primary_entity_id": router_out.primary_entity_id,
                "primary_entity_type": router_out.primary_entity_type,
                "validated_filters": router_out.validated_filters,
            }
            record.execution_status = QueryExecutionState.ROUTING_COMPLETED

            # 2. Adaptive Retrieval & Reranking Stage
            with StageTimer() as t_retrieval:
                evidence, assessment, telemetry = self._controller.execute_adaptive_retrieval(
                    item.query, router_out
                )
            retrieval_latency = t_retrieval.elapsed
            rerank_latency = float(telemetry.get("rerank_time_sec", 0.0))

            # Evaluate Retrieval via RAGAS
            r_metrics = self.retrieval_evaluator.evaluate(
                retrieved_chunks=evidence,
                expected_answer=item.expected_answer,
                query=item.query,
            )

            record.retrieval = {
                "candidate_count": r_metrics["candidate_count"],
                "retrieved_chunk_ids": r_metrics["retrieved_chunk_ids"],
                "retrieved_contexts": r_metrics["retrieved_contexts"],
                "retrieval_latency": retrieval_latency,
            }
            record.reranking = {
                "reranked_chunk_ids": r_metrics["retrieved_chunk_ids"][:5],
                "selected_contexts": r_metrics["retrieved_contexts"][:5],
                "reranking_latency": rerank_latency,
            }
            record.execution_status = QueryExecutionState.RETRIEVAL_COMPLETED

            # 3. Answer Generation Stage
            with StageTimer() as t_gen:
                gen_response: RetrievalResponse = self.retry_handler.execute_with_retry(
                    self._generator.generate,
                    item.query,
                    evidence,
                    operation_name=f"Generation {item.id}",
                )
            gen_latency = t_gen.elapsed

            evidence_dicts = []
            for ev in getattr(gen_response, "evidence", []):
                if hasattr(ev, "model_dump"):
                    evidence_dicts.append(ev.model_dump())
                elif hasattr(ev, "__dataclass_fields__"):
                    evidence_dicts.append(asdict(ev))
                elif isinstance(ev, dict):
                    evidence_dicts.append(ev)

            record.generation = {
                "generated_answer": gen_response.answer,
                "model": self.retrieval_config.get_model_name(role="generator"),
                "provider": self.retrieval_config.get_provider(role="generator"),
                "generation_latency": gen_latency,
                "status": gen_response.status.value if hasattr(gen_response.status, "value") else str(gen_response.status),
                "evidence": evidence_dicts,
            }
            record.execution_status = QueryExecutionState.GENERATION_COMPLETED

            # 4. Evaluation Metrics Stage (LLM Judge)
            with StageTimer() as t_eval:
                g_metrics = self.generation_evaluator.evaluate(
                    query=item.query,
                    generated_answer=gen_response.answer,
                    expected_answer=item.expected_answer,
                    retrieved_contexts=r_metrics["retrieved_contexts"],
                )
            eval_latency = t_eval.elapsed

            # Merge metric status and metric errors from retrieval and generation
            combined_status = {
                **r_metrics.get("metric_status", {}),
                **g_metrics.get("metric_status", {}),
            }
            combined_errors = {
                **r_metrics.get("metric_errors", {}),
                **g_metrics.get("metric_errors", {}),
            }

            record.evaluation = {
                "context_precision": r_metrics.get("context_precision"),
                "context_recall": r_metrics.get("context_recall"),
                "faithfulness": g_metrics.get("faithfulness"),
                "answer_relevancy": g_metrics.get("answer_relevancy"),
                "answer_correctness": g_metrics.get("answer_correctness"),
                "metric_status": combined_status,
                "metric_errors": combined_errors,
                "reasoning": g_metrics.get("reasoning", ""),
            }

            total_latency = time.perf_counter() - start_total
            record.performance = {
                "routing_latency": routing_latency,
                "retrieval_latency": retrieval_latency,
                "reranking_latency": rerank_latency,
                "generation_latency": gen_latency,
                "evaluation_latency": eval_latency,
                "total_latency": total_latency,
            }
            record.execution_status = QueryExecutionState.EVALUATION_COMPLETED

            # Record telemetry
            self.performance_evaluator.record_query_performance(
                routing_sec=routing_latency,
                retrieval_sec=retrieval_latency,
                reranking_sec=rerank_latency,
                generation_sec=gen_latency,
                evaluation_sec=eval_latency,
                total_sec=total_latency,
            )

        except Exception as exc:
            total_latency = time.perf_counter() - start_total
            self.logger.error("Evaluation failed for query %s: %s", item.id, exc)
            record.error = str(exc)
            record.performance["total_latency"] = total_latency
            if "rate limit" in str(exc).lower() or "429" in str(exc):
                record.execution_status = QueryExecutionState.RATE_LIMITED
            else:
                record.execution_status = QueryExecutionState.FAILED

        # Save checkpoint
        self.checkpoints.save_record(record)
        return record
