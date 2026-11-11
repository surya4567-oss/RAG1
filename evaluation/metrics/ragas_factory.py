"""Factory for initializing RAGAS LLMs and Embeddings with VanillaRAG endpoints."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar
import numpy as np
import openai
from pydantic import BaseModel
import json_repair

from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import InstructorBaseRagasLLM
from src.ingestion.embeddings import LocalBGEEmbeddings
from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config

T = TypeVar("T", bound=BaseModel)

_shared_bge_embeddings: Optional[LocalBGEEmbeddings] = None
_cached_ragas_emb: Optional["BGERagasEmbeddingAdapter"] = None
_cached_ragas_llm: Optional[InstructorBaseRagasLLM] = None


def get_shared_bge_embeddings() -> LocalBGEEmbeddings:
    """Retrieve or create the shared singleton instance of LocalBGEEmbeddings."""
    global _shared_bge_embeddings
    if _shared_bge_embeddings is None:
        logger.info("Initializing shared LocalBGEEmbeddings singleton (BAAI/bge-base-en-v1.5)...")
        _shared_bge_embeddings = LocalBGEEmbeddings()
    return _shared_bge_embeddings


class BGERagasEmbeddingAdapter(BaseRagasEmbeddings):
    """Ragas-compatible embedding adapter reusing local BAAI/bge-base-en-v1.5 model.

    Implements both LangChain-style (embed_query, embed_documents) and modern
    Ragas-style (embed_text, embed_texts, aembed_text, aembed_texts) interfaces.
    """

    def __init__(self, embedder: Optional[LocalBGEEmbeddings] = None) -> None:
        super().__init__()
        self._embedder = embedder or get_shared_bge_embeddings()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        return self._embedder.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings."""
        if not texts:
            return []
        return self._embedder.embed_texts(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single query string."""
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a list of document strings."""
        return await asyncio.to_thread(self.embed_documents, texts)

    def embed_text(self, text: str) -> List[float]:
        """Modern Ragas interface for embedding a single text string."""
        return self.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Modern Ragas interface for embedding multiple text strings."""
        return self.embed_documents(texts)

    async def aembed_text(self, text: str) -> List[float]:
        """Modern Ragas async interface for embedding a single text string."""
        return await self.aembed_query(text)

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """Modern Ragas async interface for embedding multiple text strings."""
        return await self.aembed_documents(texts)


class RobustQwenRagasLLM(InstructorBaseRagasLLM):
    """Robust OpenAI-compatible Ragas LLM adapter with schema-aware JSON repair and retries.

    Prevents malformed JSON errors from unescaped quotes while preserving the output schema.
    """

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        temperature: float = 0.01,
        max_tokens: int = 1024,
        max_retries: int = 1,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.is_async = True

    def _invoke_and_parse(
        self,
        prompt: str,
        response_model: Type[T],
    ) -> T:
        schema = response_model.model_json_schema()
        system_prompt = (
            f"You are a strict evaluation judge. Respond strictly in valid RFC 8259 JSON matching this schema: "
            f"{json.dumps(schema)}.\n"
            f"CRITICAL REQUIREMENT: Inside any explanation, reason, or text fields, any quotation marks MUST "
            f"be escaped with a backslash (\\\") or replaced with single quotes (''). Never output unescaped double quotes inside a JSON string value."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_text = resp.choices[0].message.content or ""
                raw_text = raw_text.strip()

                # Step 1: Direct Pydantic JSON parsing
                try:
                    return response_model.model_validate_json(raw_text)
                except Exception as parse_err:
                    logger.debug(
                        "Direct model_validate_json failed (%s). Attempting json_repair on raw output...",
                        parse_err,
                    )

                # Step 2: Schema-aware JSON repair (fixes unescaped quotes, trailing commas, etc.)
                repaired_obj = json_repair.loads(raw_text)
                if isinstance(repaired_obj, dict):
                    return response_model.model_validate(repaired_obj)
                elif isinstance(repaired_obj, list) and repaired_obj and isinstance(repaired_obj[0], dict):
                    return response_model.model_validate(repaired_obj[0])

                raise ValueError(f"Repaired output is not a JSON object: {repaired_obj}")

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "RobustQwenRagasLLM attempt %d/%d failed: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    # Append feedback for a single bounded retry
                    messages.append(
                        {
                            "role": "assistant",
                            "content": raw_text if "raw_text" in locals() else "",
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Your previous response produced a JSON syntax error: {exc}. "
                                f"Please regenerate your answer in strictly valid JSON, escaping all double quotes inside string fields with backslash."
                            ),
                        }
                    )

        raise ValueError(
            f"Failed to generate and parse valid {response_model.__name__} after {self.max_retries + 1} attempts: {last_error}"
        )

    def generate(self, prompt: str, response_model: Type[T]) -> T:
        """Synchronously generate structured output."""
        return self._invoke_and_parse(prompt, response_model)

    async def agenerate(self, prompt: str, response_model: Type[T]) -> T:
        """Asynchronously generate structured output in worker thread."""
        return await asyncio.to_thread(self._invoke_and_parse, prompt, response_model)


def get_ragas_llm(config: Optional[RetrievalConfig] = None) -> InstructorBaseRagasLLM:
    """Instantiate Ragas LLM adapter using the configured Kaggle/Colab or Groq provider."""
    global _cached_ragas_llm
    if _cached_ragas_llm is not None:
        return _cached_ragas_llm

    cfg = config or default_retrieval_config
    provider = cfg.get_provider(role="judge")
    model_name = cfg.get_model_name(role="judge")

    if provider in ("kaggle", "colab", "remote", "openai") and cfg.colab_llm_base_url:
        client = openai.OpenAI(
            base_url=cfg.colab_llm_base_url,
            api_key=cfg.colab_llm_api_key or "EMPTY",
            default_headers={"ngrok-skip-browser-warning": "true"},
        )
        logger.info(
            "Initializing RobustQwenRagasLLM via endpoint %s (model: %s)",
            cfg.colab_llm_base_url,
            model_name,
        )
        _cached_ragas_llm = RobustQwenRagasLLM(
            client=client,
            model=model_name,
            temperature=0.01,
            max_tokens=1024,
            max_retries=1,
        )
    else:
        # Default Groq provider via OpenAI-compatible endpoint
        client = openai.OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=cfg.groq_api_key,
        )
        logger.info("Initializing RobustQwenRagasLLM via Groq API (model: %s)", model_name)
        _cached_ragas_llm = RobustQwenRagasLLM(
            client=client,
            model=model_name or "llama-3.3-70b-versatile",
            temperature=0.01,
            max_tokens=1024,
            max_retries=1,
        )

    return _cached_ragas_llm


def get_ragas_embeddings(config: Optional[RetrievalConfig] = None) -> BGERagasEmbeddingAdapter:
    """Instantiate Ragas embedding adapter reusing local BGE embedding model."""
    global _cached_ragas_emb
    if _cached_ragas_emb is not None:
        return _cached_ragas_emb

    logger.info("Initializing BGERagasEmbeddingAdapter with shared BGE model...")
    _cached_ragas_emb = BGERagasEmbeddingAdapter()
    return _cached_ragas_emb
