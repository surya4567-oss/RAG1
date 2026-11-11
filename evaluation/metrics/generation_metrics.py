"""Generation metrics evaluation using official RAGAS metrics alone."""

from typing import Any, Dict, List, Optional
from ragas import SingleTurnSample
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, Faithfulness
from ragas.metrics._answer_similarity import AnswerSimilarity

from evaluation.metrics.ragas_factory import get_ragas_embeddings, get_ragas_llm
from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config


class GenerationMetricsEvaluator:
    """Evaluates answer generation strictly using RAGAS Faithfulness, AnswerRelevancy, and AnswerCorrectness."""

    def __init__(self, config: Optional[RetrievalConfig] = None) -> None:
        self.config = config or default_retrieval_config
        self.faithfulness_metric = Faithfulness()
        self.answer_relevancy_metric = AnswerRelevancy()
        self.answer_correctness_metric = AnswerCorrectness()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            try:
                ragas_llm = get_ragas_llm(self.config)
                ragas_emb = get_ragas_embeddings(self.config)

                self.faithfulness_metric.llm = ragas_llm

                self.answer_relevancy_metric.llm = ragas_llm
                self.answer_relevancy_metric.embeddings = ragas_emb

                self.answer_correctness_metric.llm = ragas_llm
                self.answer_correctness_metric.embeddings = ragas_emb
                # Explicitly initialize AnswerSimilarity to support single_turn_score execution
                self.answer_correctness_metric.answer_similarity = AnswerSimilarity(embeddings=ragas_emb)

                self._initialized = True
            except Exception as e:
                logger.error("Failed to initialize Ragas LLM/Embeddings for generation metrics: %s", e)

    def evaluate_generation(
        self,
        query: str,
        generated_answer: Optional[str],
        expected_answer: str,
        retrieved_contexts: List[str],
    ) -> Dict[str, Any]:
        """Compute RAGAS Faithfulness, Answer Relevancy, and Answer Correctness.

        Returns:
            Dict containing metric scores (Optional[float], None if evaluation failed),
            'metric_status', 'metric_errors', and 'reasoning'.
        """
        metric_status: Dict[str, str] = {}
        metric_errors: Dict[str, str] = {}

        ans_text = (generated_answer or "").strip()
        exp_text = expected_answer.strip()

        if not ans_text:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "answer_correctness": 0.0,
                "metric_status": {
                    "faithfulness": "SUCCESS",
                    "answer_relevancy": "SUCCESS",
                    "answer_correctness": "SUCCESS",
                },
                "metric_errors": {},
                "reasoning": "Generated answer is empty.",
            }

        self._ensure_initialized()
        sample = SingleTurnSample(
            user_input=query,
            response=ans_text,
            reference=exp_text,
            retrieved_contexts=retrieved_contexts or ["No context retrieved."],
        )

        faith_score: Optional[float] = None
        try:
            raw_faith = self.faithfulness_metric.single_turn_score(sample)
            faith_score = max(0.0, min(1.0, float(raw_faith)))
            metric_status["faithfulness"] = "SUCCESS"
        except Exception as e:
            logger.warning("Ragas Faithfulness scoring failed: %s", e)
            faith_score = None
            metric_status["faithfulness"] = "ERROR"
            metric_errors["faithfulness"] = str(e)

        rel_score: Optional[float] = None
        try:
            raw_rel = self.answer_relevancy_metric.single_turn_score(sample)
            rel_score = max(0.0, min(1.0, float(raw_rel)))
            metric_status["answer_relevancy"] = "SUCCESS"
        except Exception as e:
            logger.warning("Ragas AnswerRelevancy scoring failed: %s", e)
            rel_score = None
            metric_status["answer_relevancy"] = "ERROR"
            metric_errors["answer_relevancy"] = str(e)

        corr_score: Optional[float] = None
        try:
            raw_corr = self.answer_correctness_metric.single_turn_score(sample)
            corr_score = max(0.0, min(1.0, float(raw_corr)))
            metric_status["answer_correctness"] = "SUCCESS"
        except Exception as e:
            logger.warning("Ragas AnswerCorrectness scoring failed: %s", e)
            corr_score = None
            metric_status["answer_correctness"] = "ERROR"
            metric_errors["answer_correctness"] = str(e)

        return {
            "faithfulness": faith_score,
            "answer_relevancy": rel_score,
            "answer_correctness": corr_score,
            "metric_status": metric_status,
            "metric_errors": metric_errors,
            "reasoning": "Computed via official RAGAS metric single_turn_score.",
        }
