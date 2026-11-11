"""Independent verification tests for the three previously failing Ragas metrics.

Verifies:
1. Context Precision (robust JSON parsing, handles unescaped inner quotes)
2. Answer Relevancy (BGERagasEmbeddingAdapter with embed_query/embed_documents)
3. Answer Correctness (AnswerSimilarity configured with embeddings and factuality)
"""

import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from unittest.mock import MagicMock
import pytest
from ragas import SingleTurnSample
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, ContextPrecision
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._context_precision import Verification
from ragas.metrics._faithfulness import StatementGeneratorOutput
from ragas.metrics._answer_correctness import ClassificationWithReason, StatementsWithReason
from ragas.metrics._answer_relevance import ResponseRelevanceOutput

from evaluation.metrics.ragas_factory import (
    BGERagasEmbeddingAdapter,
    RobustQwenRagasLLM,
    get_ragas_embeddings,
)
from src.retrieval.config import default_retrieval_config


# ---------------------------------------------------------------------------
# Test 1: Context Precision with Unescaped Quotes
# ---------------------------------------------------------------------------
def test_context_precision_handles_unescaped_quotes():
    """Verify Context Precision resolves the malformed JSON issue without crashing."""
    mock_client = MagicMock()
    # Malformed JSON with literal unescaped double quotes inside "reason"
    malformed_json_response = (
        '{\n'
        '  "reason": "The provided context included the version number (1.0) and the date (11 October 2026), '
        'which are relevant to the answer. However, the exact phrase \'Baseline draft, with status "For stakeholder '
        'review and approval"\' was not explicitly mentioned in the context, making the answer partially derived '
        'from the context and partly inferred.",\n'
        '  "verdict": 1\n'
        '}'
    )
    mock_choice = MagicMock()
    mock_choice.message.content = malformed_json_response
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    robust_llm = RobustQwenRagasLLM(client=mock_client, model="Qwen/Qwen2.5-7B-Instruct")
    cp = ContextPrecision()
    cp.llm = robust_llm

    sample = SingleTurnSample(
        user_input="What is the status of BUG-000001?",
        retrieved_contexts=[
            "Defect BUG-000001 status: Baseline draft, with status 'For stakeholder review and approval'."
        ],
        reference="Status is Closed",
    )

    score = cp.single_turn_score(sample)
    assert score is not None
    assert score == pytest.approx(1.0)
    print(f"\n[PASSED] Context Precision successfully handled unescaped inner quotes! Score: {score}")


# ---------------------------------------------------------------------------
# Test 2: Answer Relevancy with BGERagasEmbeddingAdapter
# ---------------------------------------------------------------------------
def test_answer_relevancy_embedding_adapter():
    """Verify Answer Relevancy computes similarity without ''HuggingFaceEmbeddings' object has no attribute 'embed_query''."""
    emb_adapter = get_ragas_embeddings(default_retrieval_config)

    # Verify adapter has all required methods
    assert hasattr(emb_adapter, "embed_query")
    assert hasattr(emb_adapter, "embed_documents")
    assert hasattr(emb_adapter, "embed_text")
    assert hasattr(emb_adapter, "embed_texts")

    # Test single query and document embedding
    q_vec = emb_adapter.embed_query("What is the status of BUG-000001?")
    assert len(q_vec) == 768
    docs_vec = emb_adapter.embed_documents(["Defect BUG-000001 is Closed."])
    assert len(docs_vec) == 1
    assert len(docs_vec[0]) == 768

    ar = AnswerRelevancy()
    ar.embeddings = emb_adapter

    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"question": "What is the status of BUG-000001?", "noncommittal": 0}'
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    ar.llm = RobustQwenRagasLLM(client=mock_client, model="Qwen/Qwen2.5-7B-Instruct")

    sample = SingleTurnSample(
        user_input="What is the status of BUG-000001?",
        response="The status of BUG-000001 is Closed in the defect tracking database.",
        retrieved_contexts=["Defect BUG-000001 status is Closed."],
    )

    score = ar.single_turn_score(sample)
    assert score is not None
    assert 0.0 <= score <= 1.0
    print(f"\n[PASSED] Answer Relevancy successfully scored via BGE embedding adapter! Score: {score:.4f}")


# ---------------------------------------------------------------------------
# Test 3: Answer Correctness with AnswerSimilarity
# ---------------------------------------------------------------------------
def test_answer_correctness_with_answer_similarity():
    """Verify Answer Correctness evaluates with AnswerSimilarity and factuality configured."""
    emb_adapter = get_ragas_embeddings(default_retrieval_config)

    ac = AnswerCorrectness()
    ac.embeddings = emb_adapter
    ac.answer_similarity = AnswerSimilarity(embeddings=emb_adapter)

    # Ensure answer_similarity is not None
    assert ac.answer_similarity is not None

    mock_client = MagicMock()
    # Mock LLM response to provide statements for StatementGenerator and classification for CorrectnessClassifier
    def mock_create(*args, **kwargs):
        msgs = kwargs.get("messages", [])
        sys_text = msgs[0]["content"] if msgs else ""

        if "ClassificationWithReason" in sys_text:
            content = (
                '{\n'
                '  "TP": [{"statement": "Defect status is Closed", "reason": "Direct match"}],\n'
                '  "FP": [],\n'
                '  "FN": []\n'
                '}'
            )
        else:
            content = '{"statements": ["Defect BUG-000001 status is Closed."]}'

        choice = MagicMock()
        choice.message.content = content
        return MagicMock(choices=[choice])

    mock_client.chat.completions.create.side_effect = mock_create
    ac.llm = RobustQwenRagasLLM(client=mock_client, model="Qwen/Qwen2.5-7B-Instruct")

    sample = SingleTurnSample(
        user_input="What is the status of BUG-000001?",
        response="Defect BUG-000001 status is Closed.",
        reference="Defect BUG-000001 status is Closed.",
        retrieved_contexts=["Defect BUG-000001 status is Closed."],
    )

    score = ac.single_turn_score(sample)
    assert score is not None
    assert 0.0 <= score <= 1.0
    print(f"\n[PASSED] Answer Correctness successfully scored with AnswerSimilarity set! Score: {score:.4f}")


if __name__ == "__main__":
    print("Running independent verification of Ragas metrics...")
    test_context_precision_handles_unescaped_quotes()
    test_answer_relevancy_embedding_adapter()
    test_answer_correctness_with_answer_similarity()
    print("\nALL 3 RAGAS METRICS PASSED INDEPENDENT VERIFICATION!")
