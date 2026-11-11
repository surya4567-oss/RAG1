"""Centralized LLM Provider Factory for VanillaRAG.

Supports dynamic selection between:
- 'groq': Groq Cloud API via ChatGroq
- 'colab' / 'kaggle' / 'remote': Kaggle Notebooks, Google Colab (vLLM / Ollama / OpenAI-compatible endpoint) via ChatOpenAI
"""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel

from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config


def get_chat_llm(
    config: Optional[RetrievalConfig] = None,
    role: str = "general",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    json_mode: bool = True,
) -> Optional[BaseChatModel]:
    """Instantiate a chat model dynamically based on configuration and component role.

    Args:
        config: RetrievalConfig instance (defaults to default_retrieval_config).
        role: Component role ('router', 'generator', 'judge', or 'general').
        max_tokens: Maximum completion tokens (optional).
        temperature: Sampling temperature (optional; falls back to config.groq_temperature).
        json_mode: Whether to enforce JSON output format.

    Returns:
        Configured BaseChatModel instance (ChatGroq or ChatOpenAI), or None if unconfigured.
    """
    cfg = config or default_retrieval_config

    # Determine provider for the given role
    role_provider = getattr(cfg, f"{role}_llm_provider", None)
    provider = (role_provider or cfg.llm_provider or "groq").lower().strip()
    temp = cfg.groq_temperature if temperature is None else temperature

    if provider in ("colab", "kaggle", "remote", "openai"):
        if not cfg.colab_llm_base_url:
            logger.warning(
                f"Endpoint base URL is not set for provider '{provider}' and role '{role}'. "
                f"Ensure your Kaggle/Colab tunnel is active and KAGGLE_LLM_BASE_URL / COLAB_LLM_BASE_URL is configured in .env."
            )
            return None

        try:
            from langchain_openai import ChatOpenAI

            model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
            llm = ChatOpenAI(
                base_url=cfg.colab_llm_base_url,
                api_key=cfg.colab_llm_api_key or "EMPTY",
                model=cfg.colab_llm_model,
                temperature=temp,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
                default_headers={"ngrok-skip-browser-warning": "true"},
            )
            logger.info(
                f"Initialized {provider.capitalize()} ChatOpenAI for role '{role}': model='{cfg.colab_llm_model}' "
                f"endpoint='{cfg.colab_llm_base_url}'"
            )
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize {provider.capitalize()} ChatOpenAI for role '{role}': {e}")
            return None

    elif provider == "groq":
        if not cfg.groq_api_key:
            logger.warning(f"GROQ_API_KEY is not set for role '{role}'. Fallback mode will be used.")
            return None

        try:
            from langchain_groq import ChatGroq

            kwargs = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            llm = ChatGroq(
                groq_api_key=cfg.groq_api_key,
                model_name=cfg.groq_model,
                temperature=temp,
                **kwargs,
            )
            logger.info(
                f"Initialized Groq ChatGroq for role '{role}': model='{cfg.groq_model}'"
            )
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq for role '{role}': {e}")
            return None

    else:
        logger.error(f"Unknown LLM provider '{provider}' requested for role '{role}'.")
        return None

