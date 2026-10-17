"""LLM Query Router powered by Groq with heuristic identifier extraction and schema validation."""

import json
import re
from typing import Any, Dict, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT
from src.retrieval.schemas import RetrievalRoute, RouterOutput


class LLMQueryRouter:
    """Classifies user query intent, extracts entities, and selects retrieval strategy."""

    # Regex patterns for deterministic QA entity recognition
    ID_PATTERNS = {
        "test_case_id": re.compile(r"\b(TC[-_]?\d+)\b", re.IGNORECASE),
        "defect_id": re.compile(r"\b((?:BUG|DEFECT|ISSUE)[-_]?\d+)\b", re.IGNORECASE),
        "execution_id": re.compile(r"\b((?:EXEC|RUN)[-_]?\d+)\b", re.IGNORECASE),
        "requirement_id": re.compile(r"\b(REQ[-_]?\d+)\b", re.IGNORECASE),
        "error_code": re.compile(
            r"\b(AccessError|PermissionError|ValidationException|ConnectionError|TimeoutException|NullPointer|403|404|500)\b",
            re.IGNORECASE,
        ),
        "build_number": re.compile(r"\b(?:build|bld)[-_ #]?(\d+)\b", re.IGNORECASE),
    }

    def __init__(self, config: Optional[RetrievalConfig] = None, llm: Optional[ChatGroq] = None):
        self.config = config or default_retrieval_config
        self._llm = llm

    def _get_llm(self) -> Optional[ChatGroq]:
        if self._llm is not None:
            return self._llm

        if not self.config.groq_api_key:
            logger.warning("GROQ_API_KEY is not set. Router will operate in deterministic heuristic fallback mode.")
            return None

        try:
            self._llm = ChatGroq(
                groq_api_key=self.config.groq_api_key,
                model_name=self.config.groq_model,
                temperature=self.config.groq_temperature,
                response_format={"type": "json_object"},
            )
            return self._llm
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq for router: {e}")
            return None

    def extract_regex_identifiers(self, question: str) -> Dict[str, str]:
        """Extract explicit domain identifiers via compiled regex rules."""
        extracted = {}
        for entity_type, pattern in self.ID_PATTERNS.items():
            match = pattern.search(question)
            if match:
                val = match.group(1).upper()
                extracted[entity_type] = val
        return extracted

    def _validate_and_sanitize_filters(
        self, proposed_filters: Dict[str, Any], regex_ids: Dict[str, str]
    ) -> Dict[str, Any]:
        """Ensure filters strictly match allowed schema keys and merge with regex IDs."""
        validated: Dict[str, Any] = {}

        # 1. Add regex-verified IDs as filters
        for key in ["test_case_id", "defect_id", "execution_id", "requirement_id"]:
            if key in regex_ids and regex_ids[key]:
                validated[key] = regex_ids[key]

        if "build_number" in regex_ids and regex_ids["build_number"]:
            validated["build_number"] = regex_ids["build_number"]

        # 2. Add LLM proposed filters only if in allowed_filter_fields
        for field_name, value in proposed_filters.items():
            if field_name in self.config.allowed_filter_fields and value not in (None, "", "null"):
                # Normalize values
                if isinstance(value, str):
                    val_str = value.strip()
                    if field_name.endswith("_id"):
                        val_str = val_str.upper()
                    validated[field_name] = val_str
                else:
                    validated[field_name] = value

        return validated

    def _fallback_route(self, question: str) -> RouterOutput:
        """Deterministic heuristic route when LLM is unavailable or fails."""
        regex_ids = self.extract_regex_identifiers(question)
        filters = self._validate_and_sanitize_filters({}, regex_ids)

        q_lower = question.lower()
        has_id = bool(
            regex_ids.get("test_case_id")
            or regex_ids.get("defect_id")
            or regex_ids.get("execution_id")
            or regex_ids.get("requirement_id")
            or regex_ids.get("error_code")
        )
        is_conceptual = any(w in q_lower for w in ["explain", "overview", "what is", "how does", "summarize", "describe", "why does"])

        if has_id and not is_conceptual:
            strategy = RetrievalRoute.KEYWORD
            reasoning = "Deterministic fallback: exact identifier or error code detected."
        elif is_conceptual and not has_id:
            strategy = RetrievalRoute.DENSE
            reasoning = "Deterministic fallback: conceptual/explanatory question without specific ID."
        else:
            strategy = RetrievalRoute.HYBRID
            reasoning = "Deterministic fallback: question involves mixed entities or cross-source reasoning."

        return RouterOutput(
            intent=f"Retrieve QA records for: {question[:80]}",
            strategy=strategy,
            extracted_identifiers=regex_ids,
            validated_filters=filters,
            reasoning=reasoning,
        )

    def route(self, question: str) -> RouterOutput:
        """Classify query intent, extract entities, validate schema filters, and select strategy."""
        clean_q = question.strip()
        if not clean_q:
            return RouterOutput(
                intent="Empty query",
                strategy=RetrievalRoute.DENSE,
                extracted_identifiers={},
                validated_filters={},
                reasoning="Empty input defaulted to dense.",
            )

        regex_ids = self.extract_regex_identifiers(clean_q)
        llm = self._get_llm()

        if llm is None:
            return self._fallback_route(clean_q)

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=ROUTER_USER_PROMPT.format(question=clean_q)),
        ]

        try:
            response = llm.invoke(messages)
            raw_content = response.content if hasattr(response, "content") else str(response)

            # Parse JSON object
            data = json.loads(raw_content)

            # Map strategy
            strategy_str = str(data.get("strategy", "hybrid")).lower().strip()
            if strategy_str not in ("keyword", "dense", "hybrid"):
                strategy_str = "hybrid"
            strategy = RetrievalRoute(strategy_str)

            # Merge and sanitize identifiers
            llm_ids = data.get("extracted_identifiers") or {}
            all_ids = {**regex_ids, **{k: v for k, v in llm_ids.items() if v}}

            # Sanitize filters against existing schema
            proposed_filters = data.get("validated_filters") or {}
            validated_filters = self._validate_and_sanitize_filters(proposed_filters, all_ids)

            router_output = RouterOutput(
                intent=str(data.get("intent") or clean_q),
                strategy=strategy,
                extracted_identifiers=all_ids,
                validated_filters=validated_filters,
                reasoning=str(data.get("reasoning") or "Classified by Groq router."),
            )

            logger.info(
                f"Router selected strategy='{router_output.strategy.value}' | "
                f"Filters={router_output.validated_filters} | "
                f"IDs={router_output.extracted_identifiers}"
            )
            return router_output

        except Exception as e:
            logger.warning(f"Groq router failed ({e}). Reverting to controlled heuristic fallback.")
            return self._fallback_route(clean_q)
