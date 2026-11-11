"""Unit tests for multi-provider LLM Factory and Colab integration."""

from unittest.mock import MagicMock, patch
import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from src.retrieval.config import RetrievalConfig
from src.retrieval.llm_factory import get_chat_llm
from src.retrieval.router import LLMQueryRouter
from src.retrieval.answer_generator import EvidenceGroundedAnswerGenerator


def test_factory_returns_none_when_unconfigured():
    """Verify factory safely returns None when required API keys or URLs are missing."""
    empty_cfg = RetrievalConfig(
        groq_api_key="",
        colab_llm_base_url="",
        llm_provider="groq",
        router_llm_provider="groq",
    )
    assert get_chat_llm(empty_cfg, role="router") is None

    colab_cfg = RetrievalConfig(
        llm_provider="colab",
        colab_llm_base_url="",
        generator_llm_provider="colab",
    )
    assert get_chat_llm(colab_cfg, role="generator") is None


def test_factory_initializes_groq_by_default():
    """Verify factory instantiates ChatGroq when provider is groq."""
    cfg = RetrievalConfig(
        groq_api_key="gsk_dummy_test_key",
        groq_model="llama-3.3-70b-versatile",
        llm_provider="groq",
        router_llm_provider="groq",
    )
    with patch("langchain_groq.ChatGroq") as mock_groq:
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_groq.return_value = mock_instance

        llm = get_chat_llm(cfg, role="router", max_tokens=350, json_mode=True)
        assert llm is not None
        mock_groq.assert_called_once()
        _, kwargs = mock_groq.call_args
        assert kwargs.get("model_name") == "llama-3.3-70b-versatile"
        assert kwargs.get("max_tokens") == 350
        assert kwargs.get("response_format") == {"type": "json_object"}


def test_factory_initializes_colab_openai_compatible():
    """Verify factory instantiates ChatOpenAI when provider is colab or kaggle."""
    cfg = RetrievalConfig(
        llm_provider="colab",
        generator_llm_provider="colab",
        colab_llm_base_url="https://test-tunnel.trycloudflare.com/v1",
        colab_llm_api_key="EMPTY",
        colab_llm_model="Qwen/Qwen2.5-7B-Instruct",
    )
    with patch("langchain_openai.ChatOpenAI") as mock_openai:
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_openai.return_value = mock_instance

        llm = get_chat_llm(cfg, role="generator", max_tokens=1024, json_mode=True)
        assert llm is not None
        mock_openai.assert_called_once()
        _, kwargs = mock_openai.call_args
        assert kwargs.get("base_url") == "https://test-tunnel.trycloudflare.com/v1"
        assert kwargs.get("model") == "Qwen/Qwen2.5-7B-Instruct"
        assert kwargs.get("max_tokens") == 1024
        assert kwargs.get("model_kwargs") == {"response_format": {"type": "json_object"}}


def test_factory_initializes_kaggle_provider():
    """Verify factory instantiates ChatOpenAI when provider is kaggle."""
    cfg = RetrievalConfig(
        llm_provider="kaggle",
        router_llm_provider="kaggle",
        colab_llm_base_url="https://kaggle-ngrok-tunnel.ngrok-free.app/v1",
        colab_llm_api_key="EMPTY",
        colab_llm_model="Qwen/Qwen2.5-7B-Instruct",
    )
    with patch("langchain_openai.ChatOpenAI") as mock_openai:
        mock_instance = MagicMock(spec=BaseChatModel)
        mock_openai.return_value = mock_instance

        llm = get_chat_llm(cfg, role="router", max_tokens=256, json_mode=True)
        assert llm is not None
        mock_openai.assert_called_once()
        _, kwargs = mock_openai.call_args
        assert kwargs.get("base_url") == "https://kaggle-ngrok-tunnel.ngrok-free.app/v1"
        assert kwargs.get("model") == "Qwen/Qwen2.5-7B-Instruct"


def test_factory_supports_role_specific_overrides():
    """Verify router can use Groq while generator uses Colab simultaneously."""
    cfg = RetrievalConfig(
        llm_provider="groq",
        groq_api_key="gsk_dummy_key",
        router_llm_provider="groq",
        generator_llm_provider="colab",
        colab_llm_base_url="https://colab-tunnel.trycloudflare.com/v1",
        colab_llm_model="Qwen/Qwen2.5-7B-Instruct",
    )

    with patch("langchain_groq.ChatGroq") as mock_groq, \
         patch("langchain_openai.ChatOpenAI") as mock_openai:

        mock_groq.return_value = MagicMock(spec=BaseChatModel)
        mock_openai.return_value = MagicMock(spec=BaseChatModel)

        router_llm = get_chat_llm(cfg, role="router")
        generator_llm = get_chat_llm(cfg, role="generator")

        assert router_llm is not None
        assert generator_llm is not None
        mock_groq.assert_called_once()
        mock_openai.assert_called_once()


def test_router_and_generator_integrate_with_factory():
    """Verify router and generator call get_chat_llm with respective roles."""
    cfg = RetrievalConfig(
        groq_api_key="gsk_test_key",
        llm_provider="groq",
    )

    with patch("src.retrieval.router.get_chat_llm") as mock_router_factory, \
         patch("src.retrieval.answer_generator.get_chat_llm") as mock_gen_factory:

        mock_llm = MagicMock(spec=BaseChatModel)
        mock_router_factory.return_value = mock_llm
        mock_gen_factory.return_value = mock_llm

        router = LLMQueryRouter(config=cfg)
        r_llm = router._get_llm()
        assert r_llm is mock_llm
        mock_router_factory.assert_called_with(config=cfg, role="router", max_tokens=350, json_mode=True)

        generator = EvidenceGroundedAnswerGenerator(config=cfg)
        g_llm = generator._get_llm()
        assert g_llm is mock_llm
        mock_gen_factory.assert_called_with(config=cfg, role="generator", max_tokens=1024, json_mode=True)
