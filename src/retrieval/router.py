"""LLM Query Router powered by Groq with heuristic identifier extraction and schema validation."""

import collections
import json
import random
import re
import threading
import time
from typing import Any, Dict, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from src.ingestion.logging_config import logger
from src.retrieval.config import RetrievalConfig, default_retrieval_config
from src.retrieval.llm_factory import get_chat_llm
from src.retrieval.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT
from src.retrieval.schemas import RetrievalMode, RetrievalRoute, RouterOutput


class LLMQueryRouter:
    """Classifies user query intent, extracts entities, and selects retrieval strategy."""

    # Canonical mapping of document_type aliases, plurals, and terminology to Qdrant payload enums
    DOCUMENT_TYPE_MAP = {
        # screenshot_evidence (14,854 points)
        "screenshot": "screenshot_evidence",
        "screenshots": "screenshot_evidence",
        "screenshot_evidence": "screenshot_evidence",
        "ocr": "screenshot_evidence",
        "ocr_evidence": "screenshot_evidence",
        "screen": "screenshot_evidence",
        "screens": "screenshot_evidence",
        "screen_evidence": "screenshot_evidence",
        "ui_screenshot": "screenshot_evidence",
        "ui_screenshots": "screenshot_evidence",
        "ui_capture": "screenshot_evidence",
        # failure_log (30,000 points)
        "failure_log": "failure_log",
        "failure_logs": "failure_log",
        "failure": "failure_log",
        "failures": "failure_log",
        "log": "failure_log",
        "logs": "failure_log",
        "error_log": "failure_log",
        "error_logs": "failure_log",
        "traceback": "failure_log",
        "tracebacks": "failure_log",
        "stack_trace": "failure_log",
        "stack_traces": "failure_log",
        "stacktrace": "failure_log",
        "server_log": "failure_log",
        "server_logs": "failure_log",
        # test_case (39,839 points)
        "test_case": "test_case",
        "test_cases": "test_case",
        "testcase": "test_case",
        "testcases": "test_case",
        "tc": "test_case",
        # test_execution (20,000 points)
        "test_execution": "test_execution",
        "test_executions": "test_execution",
        "execution": "test_execution",
        "executions": "test_execution",
        "test_run": "test_execution",
        "test_runs": "test_execution",
        "run": "test_execution",
        "runs": "test_execution",
        # defect (18,339 points)
        "defect": "defect",
        "defects": "defect",
        "bug": "defect",
        "bugs": "defect",
        "issue": "defect",
        "issues": "defect",
        # srs_section (744 points)
        "srs": "srs_section",
        "srs_section": "srs_section",
        "srs_sections": "srs_section",
        "requirement": "srs_section",
        "requirements": "srs_section",
        "spec": "srs_section",
        "specs": "srs_section",
        "specification": "srs_section",
        "specifications": "srs_section",
    }

    # Regex patterns for deterministic QA entity recognition
    ID_PATTERNS = {
        "test_case_id": re.compile(r"\b(TC[-_]?\d+)\b", re.IGNORECASE),
        "defect_id": re.compile(r"\b((?:BUG|DEFECT|ISSUE)[-_]?\d+)\b", re.IGNORECASE),
        "execution_id": re.compile(r"\b((?:EXEC|RUN)[-_]?\d+)\b", re.IGNORECASE),
        "requirement_id": re.compile(r"\b((?:REQ|FR[-_][A-Z]+|NFR[-_][A-Z]+)[-_]?\d+)\b", re.IGNORECASE),
        "epic": re.compile(r"\b(EPIC[-_]?\d+)\b", re.IGNORECASE),
        "story": re.compile(r"\b(STORY[-_]?\d+)\b", re.IGNORECASE),
        "error_code": re.compile(
            r"\b(AccessError|PermissionError|ValidationException|ConnectionError|TimeoutException|NullPointer|403|404|500)\b",
            re.IGNORECASE,
        ),
        "build_number": re.compile(r"\b(?:build|bld)[-_ #]?([a-zA-Z0-9_-]+)\b|\b(B-\d+)\b", re.IGNORECASE),
    }

    # Regex rules for detecting explicit document modality intent
    DOC_TYPE_RULES = [
        (re.compile(r"\b(?:screenshots?|ui captures?|screen images?|ocr evidence)\b", re.IGNORECASE), "screenshot_evidence"),
        (re.compile(r"\b(?:failure logs?|error logs?|tracebacks?|stack traces?|server logs?)\b", re.IGNORECASE), "failure_log"),
        (re.compile(r"\b(?:test executions?|execution runs?)\b", re.IGNORECASE), "test_execution"),
        (re.compile(r"\b(?:test cases?|testcases?)\b", re.IGNORECASE), "test_case"),
        (re.compile(r"\b(?:defects?|bugs?)\b", re.IGNORECASE), "defect"),
        (re.compile(r"\b(?:srs sections?|software requirements?|srs spec|functional requirements?)\b", re.IGNORECASE), "srs_section"),
    ]

    def __init__(self, config: Optional[RetrievalConfig] = None, llm: Optional[BaseChatModel] = None):
        self.config = config or default_retrieval_config
        self._llm = llm
        self._routing_cache: collections.OrderedDict[str, RouterOutput] = collections.OrderedDict()
        self._routing_cache_max_size: int = 500
        self._routing_lock: threading.Lock = threading.Lock()

    def _get_llm(self) -> Optional[BaseChatModel]:
        if self._llm is not None:
            return self._llm

        self._llm = get_chat_llm(
            config=self.config,
            role="router",
            max_tokens=350,
            json_mode=True,
        )
        if self._llm is None:
            logger.warning("Router LLM could not be initialized. Operating in deterministic heuristic fallback mode.")
        return self._llm

    def extract_regex_identifiers(self, question: str) -> Dict[str, str]:
        """Extract explicit domain identifiers and modality intent via compiled regex rules."""
        extracted = {}
        for entity_type, pattern in self.ID_PATTERNS.items():
            match = pattern.search(question)
            if match:
                val = next((g for g in match.groups() if g is not None), None)
                if val:
                    extracted[entity_type] = (
                        val.upper()
                        if entity_type.endswith("_id") or entity_type in ("epic", "story", "error_code")
                        else val
                    )

        # Detect document modality
        for pattern, doc_type in self.DOC_TYPE_RULES:
            if pattern.search(question):
                extracted["document_type"] = doc_type
                break

        return extracted

    def _validate_and_sanitize_filters(
        self,
        proposed_filters: Dict[str, Any],
        regex_ids: Dict[str, str],
        question: str = "",
    ) -> Dict[str, Any]:
        """Ensure filters strictly match allowed schema keys, normalize enums, and merge with regex IDs.

        Filters proposed by LLM must be justified by the actual user question to prevent hallucinated
        attributes (e.g. unrequested priority/status or incorrect role mapping).
        """
        validated: Dict[str, Any] = {}
        q_lower = question.strip().lower() if question else ""

        # 1. Add regex-verified IDs and modality as filters
        for key in ["test_case_id", "defect_id", "execution_id", "requirement_id", "epic", "story", "error_code", "document_type"]:
            if key in regex_ids and regex_ids[key]:
                validated[key] = regex_ids[key]

        if "build_number" in regex_ids and regex_ids["build_number"]:
            validated["build_number"] = regex_ids["build_number"]

        # 2. Add LLM proposed filters only if in allowed_filter_fields and justified by query
        for field_name, value in proposed_filters.items():
            if field_name not in self.config.allowed_filter_fields or value in (None, "", "null"):
                continue

            # Prevent hallucinated filters: if question is available, verify filter is grounded in query
            if q_lower:
                str_val = str(value).strip().lower()
                # Check categorical / attribute fields
                if field_name in ("priority", "status", "severity", "module", "component", "automation_framework", "environment", "bug_type", "test_type"):
                    if str_val not in q_lower:
                        # Value not mentioned in user query; LLM hallucinated an answer attribute as a filter constraint
                        continue

                # Check assignee / person fields
                if field_name in ("tester", "assigned_to", "developer", "reported_by"):
                    if str_val not in q_lower:
                        continue
                    # Role remapping: defects use assigned_to or developer, never tester
                    is_defect_context = "defect_id" in regex_ids or "defect" in q_lower or "bug" in q_lower
                    if field_name == "tester" and is_defect_context:
                        field_name = "assigned_to"

                # Check boolean fields
                if field_name == "regression" and "regress" not in q_lower:
                    continue
                if field_name == "has_error_message" and not any(k in q_lower for k in ("error", "exception", "traceback", "fail")):
                    continue
                if field_name == "has_warning_message" and "warning" not in q_lower:
                    continue

            # Handle document_type normalization
            if field_name == "document_type" and isinstance(value, str):
                clean_dtype = value.strip().lower()
                normalized_dtype = self.DOCUMENT_TYPE_MAP.get(clean_dtype, clean_dtype)
                validated["document_type"] = normalized_dtype
                continue

            # Handle boolean fields
            if field_name in ("has_error_message", "has_warning_message", "regression"):
                if isinstance(value, bool):
                    validated[field_name] = value
                elif isinstance(value, str):
                    val_lower = value.strip().lower()
                    if val_lower in ("true", "yes", "1"):
                        validated[field_name] = True
                    elif val_lower in ("false", "no", "0"):
                        validated[field_name] = False
                continue

            # Handle integer fields (http_status, page_number)
            if field_name in ("http_status", "page_number"):
                try:
                    validated[field_name] = int(value)
                except (ValueError, TypeError):
                    pass
                continue

            # Normalize strings
            if isinstance(value, str):
                val_str = value.strip()
                if field_name.endswith("_id") or field_name in ("epic", "story", "error_code"):
                    val_str = val_str.upper()
                validated[field_name] = val_str
            else:
                validated[field_name] = value

        # Ensure document_type in validated is normalized if it came from regex_ids
        if "document_type" in validated and isinstance(validated["document_type"], str):
            clean_dtype = validated["document_type"].strip().lower()
            validated["document_type"] = self.DOCUMENT_TYPE_MAP.get(clean_dtype, clean_dtype)

        return validated

    def determine_fallback_strategy(self, question: str) -> RetrievalRoute:
        """Determine search strategy based on query semantics when metadata filtering fails.

        When metadata filters return 0 results, this evaluates the user question to select
        the optimal unfiltered retrieval strategy (DENSE, KEYWORD, or HYBRID).
        """
        q_lower = question.lower()
        regex_ids = self.extract_regex_identifiers(question)
        has_id = bool(regex_ids)
        is_conceptual = any(
            w in q_lower
            for w in [
                "explain",
                "overview",
                "what is",
                "how does",
                "summarize",
                "describe",
                "why does",
                "interpret",
            ]
        )

        if is_conceptual and not has_id:
            return RetrievalRoute.DENSE
        elif has_id and not is_conceptual and len(question.split()) <= 4:
            return RetrievalRoute.KEYWORD
        else:
            return RetrievalRoute.HYBRID

    def route_on_filter_failure(
        self, question: str, failed_filters: Optional[Dict[str, Any]] = None
    ) -> RouterOutput:
        """Switch directly to a query-based search strategy when metadata filtering yields no results.

        Drops the restrictive metadata filters, selects the appropriate strategy directly from
        the user question, and allows full-text / semantic candidate discovery.
        """
        fallback_strategy = self.determine_fallback_strategy(question)
        regex_ids = self.extract_regex_identifiers(question)

        logger.warning(
            f"Metadata filtering failed with filters={failed_filters}. "
            f"Directly switching to query-based strategy '{fallback_strategy.value}' with filters relaxed."
        )

        return RouterOutput(
            intent=f"Query-based fallback retrieval for: {question[:80]}",
            strategy=fallback_strategy,
            fallback_strategy=fallback_strategy,
            extracted_identifiers=regex_ids,
            validated_filters={},  # Filter constraints dropped
            routing_source="heuristic",
            reasoning=(
                f"Metadata filtering returned no candidates for {failed_filters}. "
                f"Directly switched to query-based {fallback_strategy.value} strategy without strict filters."
            ),
        )

    def classify_intent_and_mode(
        self, question: str, regex_ids: Dict[str, str]
    ) -> Dict[str, Any]:
        """Classify entity type, requested attribute, target document types, and retrieval mode."""
        q_lower = question.lower()

        # 1. Determine primary entity
        primary_entity_type = None
        primary_entity_id = None

        if "defect_id" in regex_ids and regex_ids["defect_id"]:
            primary_entity_type = "defect"
            primary_entity_id = regex_ids["defect_id"]
        elif "test_case_id" in regex_ids and regex_ids["test_case_id"]:
            primary_entity_type = "test_case"
            primary_entity_id = regex_ids["test_case_id"]
        elif "execution_id" in regex_ids and regex_ids["execution_id"]:
            primary_entity_type = "test_execution"
            primary_entity_id = regex_ids["execution_id"]
        elif "requirement_id" in regex_ids and regex_ids["requirement_id"]:
            primary_entity_type = "requirement"
            primary_entity_id = regex_ids["requirement_id"]

        # 2. Determine requested attribute and target document types
        requested_attribute = None
        target_document_types = []

        if any(w in q_lower for w in ["screen name", "screen", "screenshot", "ui capture", "ocr"]):
            requested_attribute = "screen_name"
            target_document_types = ["screenshot_evidence"]
        elif any(w in q_lower for w in ["failure log", "error log", "server log", "log", "logs", "traceback", "stack trace", "exception"]):
            requested_attribute = "failure_log"
            target_document_types = ["failure_log"]
        elif "status" in q_lower:
            requested_attribute = "status"
            if primary_entity_type:
                target_document_types = [primary_entity_type]
        elif "severity" in q_lower:
            requested_attribute = "severity"
            target_document_types = ["defect"]
        elif "priority" in q_lower:
            requested_attribute = "priority"
            if primary_entity_type:
                target_document_types = [primary_entity_type]
        elif any(w in q_lower for w in ["associated test case", "linked test case", "which test cases", "what test case"]):
            requested_attribute = "test_cases"
            target_document_types = ["test_case", "test_execution"]
        elif any(w in q_lower for w in ["associated defect", "linked bug", "which defects", "what bugs"]):
            requested_attribute = "defects"
            target_document_types = ["defect"]
        elif any(w in q_lower for w in ["precondition", "test step", "expected result"]):
            requested_attribute = "test_specification"
            target_document_types = ["test_case"]

        # 3. Determine Retrieval Mode
        is_conceptual = any(
            w in q_lower
            for w in [
                "explain",
                "overview",
                "common causes",
                "why do",
                "how do",
                "best practice",
                "what are the causes",
                "general",
            ]
        )

        if not primary_entity_id:
            retrieval_mode = RetrievalMode.CONCEPTUAL
        else:
            # If asking for a property that lives in a different document type -> expansion
            modality_mismatch = bool(
                target_document_types
                and primary_entity_type
                and any(t != primary_entity_type for t in target_document_types)
            )
            cross_entity_phrases = any(
                w in q_lower
                for w in [
                    "screen name",
                    "screenshot",
                    "associated test case",
                    "which test cases",
                    "linked defect",
                    "failure log",
                    "server log",
                    "traceback",
                ]
            )

            if modality_mismatch or cross_entity_phrases:
                retrieval_mode = RetrievalMode.ENTITY_EXPANSION
            elif is_conceptual:
                retrieval_mode = RetrievalMode.CONCEPTUAL
            else:
                retrieval_mode = RetrievalMode.DIRECT_ENTITY

        return {
            "primary_entity_type": primary_entity_type,
            "primary_entity_id": primary_entity_id,
            "requested_attribute": requested_attribute,
            "target_document_types": target_document_types,
            "retrieval_mode": retrieval_mode,
        }

    def _fallback_route(self, question: str) -> RouterOutput:
        """Deterministic heuristic route when LLM is unavailable or fails."""
        regex_ids = self.extract_regex_identifiers(question)
        classification = self.classify_intent_and_mode(question, regex_ids)

        retrieval_mode = classification["retrieval_mode"]
        primary_entity_type = classification["primary_entity_type"]
        primary_entity_id = classification["primary_entity_id"]
        requested_attribute = classification["requested_attribute"]
        target_document_types = classification["target_document_types"]

        is_conceptual = (retrieval_mode == RetrievalMode.CONCEPTUAL)
        filter_ids = {} if is_conceptual else regex_ids
        filters = self._validate_and_sanitize_filters({}, filter_ids, question=question)
        fallback_strategy = self.determine_fallback_strategy(question)

        q_lower = question.lower()
        has_id = bool(primary_entity_id)

        if has_id and not is_conceptual and retrieval_mode == RetrievalMode.DIRECT_ENTITY:
            strategy = RetrievalRoute.KEYWORD
            reasoning = "Deterministic fallback: exact identifier direct entity lookup."
        elif is_conceptual and not has_id:
            strategy = RetrievalRoute.DENSE
            reasoning = "Deterministic fallback: conceptual/explanatory question without specific ID."
        else:
            strategy = RetrievalRoute.HYBRID
            reasoning = (
                "Deterministic fallback: entity expansion across related documents."
                if retrieval_mode == RetrievalMode.ENTITY_EXPANSION
                else "Deterministic fallback: hybrid search."
            )

        return RouterOutput(
            intent=f"Retrieve QA records for: {question[:80]}",
            strategy=strategy,
            fallback_strategy=fallback_strategy,
            retrieval_mode=retrieval_mode,
            primary_entity_type=primary_entity_type,
            primary_entity_id=primary_entity_id,
            requested_attribute=requested_attribute,
            target_document_types=target_document_types,
            extracted_identifiers=regex_ids,
            validated_filters=filters,
            routing_source="heuristic",
            reasoning=reasoning,
        )

    def route(self, question: str) -> RouterOutput:
        """Classify query intent, extract entities, validate schema filters, and select strategy."""
        clean_q = question.strip()
        if not clean_q:
            return RouterOutput(
                intent="Empty query",
                strategy=RetrievalRoute.DENSE,
                fallback_strategy=RetrievalRoute.DENSE,
                retrieval_mode=RetrievalMode.CONCEPTUAL,
                extracted_identifiers={},
                validated_filters={},
                routing_source="heuristic",
                reasoning="Empty input defaulted to dense.",
            )

        cache_key = clean_q.lower()
        with self._routing_lock:
            if cache_key in self._routing_cache:
                cached = self._routing_cache[cache_key]
                if hasattr(cached, "model_copy"):
                    return cached.model_copy(update={"routing_source": "cache"})
                return cached.copy(update={"routing_source": "cache"})

        regex_ids = self.extract_regex_identifiers(clean_q)
        heuristic_class = self.classify_intent_and_mode(clean_q, regex_ids)
        llm = self._get_llm()

        if llm is None:
            return self._fallback_route(clean_q)

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=ROUTER_USER_PROMPT.format(question=clean_q)),
        ]

        raw_content = None
        max_retries = 3
        backoff_base = 1.5

        for attempt in range(max_retries):
            try:
                response = llm.invoke(messages)
                raw_content = response.content if hasattr(response, "content") else str(response)
                break
            except Exception as exc:
                err_msg = str(exc)
                is_rate_limit = "429" in err_msg or "rate limit" in err_msg.lower() or "too many requests" in err_msg.lower()
                if is_rate_limit and attempt < max_retries - 1:
                    retry_match = re.search(r"try again in ([\d\.]+)s", err_msg, re.IGNORECASE)
                    if retry_match:
                        delay = float(retry_match.group(1)) + 0.2
                    else:
                        delay = (backoff_base ** attempt) + random.uniform(0.1, 0.5)
                    provider_name = self.config.get_provider(role="router").capitalize()
                    logger.warning(
                        f"{provider_name} 429 rate limit encountered (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying after {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    continue
                else:
                    provider_name = self.config.get_provider(role="router").capitalize()
                    logger.warning(f"{provider_name} router invocation failed ({exc}). Reverting to controlled heuristic fallback.")
                    return self._fallback_route(clean_q)

        if not raw_content:
            return self._fallback_route(clean_q)

        try:
            # Parse JSON object
            data = json.loads(raw_content)

            # Map strategy
            strategy_str = str(data.get("strategy", "hybrid")).lower().strip()
            if strategy_str not in ("keyword", "dense", "hybrid"):
                strategy_str = "hybrid"
            strategy = RetrievalRoute(strategy_str)

            # Map retrieval mode safely
            mode_str = str(data.get("retrieval_mode", "")).lower().strip()
            if mode_str in ("direct_entity", "entity_expansion", "conceptual"):
                retrieval_mode = RetrievalMode(mode_str)
            else:
                retrieval_mode = heuristic_class["retrieval_mode"]

            primary_type = data.get("primary_entity_type") or heuristic_class["primary_entity_type"]
            primary_id = data.get("primary_entity_id") or heuristic_class["primary_entity_id"]
            req_attr = data.get("requested_attribute") or heuristic_class["requested_attribute"]

            raw_targets = data.get("target_document_types")
            if isinstance(raw_targets, list) and raw_targets:
                target_docs = [self.DOCUMENT_TYPE_MAP.get(str(t).lower().strip(), str(t)) for t in raw_targets]
            else:
                target_docs = heuristic_class["target_document_types"]

            # Merge and sanitize identifiers
            llm_ids = data.get("extracted_identifiers") or {}
            all_ids = {**regex_ids, **{k: v for k, v in llm_ids.items() if v}}

            # If in entity expansion mode, do NOT restrict document_type to target_doc in Phase 1 filters
            filter_ids = dict(all_ids)
            if retrieval_mode == RetrievalMode.ENTITY_EXPANSION and primary_type:
                filter_ids.pop("document_type", None)

            # Sanitize filters against existing schema and query context
            proposed_filters = data.get("validated_filters") or {}
            validated_filters = self._validate_and_sanitize_filters(proposed_filters, filter_ids, question=clean_q)
            if retrieval_mode == RetrievalMode.ENTITY_EXPANSION and primary_type:
                validated_filters.pop("document_type", None)

            fallback_strategy = self.determine_fallback_strategy(clean_q)

            router_output = RouterOutput(
                intent=str(data.get("intent") or clean_q),
                strategy=strategy,
                fallback_strategy=fallback_strategy,
                retrieval_mode=retrieval_mode,
                primary_entity_type=primary_type,
                primary_entity_id=primary_id,
                requested_attribute=req_attr,
                target_document_types=target_docs,
                extracted_identifiers=all_ids,
                validated_filters=validated_filters,
                routing_source="llm",
                reasoning=str(data.get("reasoning") or "Classified by Groq router."),
            )

            # Cache the router output
            with self._routing_lock:
                self._routing_cache[cache_key] = router_output
                if len(self._routing_cache) > self._routing_cache_max_size:
                    self._routing_cache.popitem(last=False)

            logger.info(
                f"Router selected strategy='{router_output.strategy.value}' | "
                f"Mode='{router_output.retrieval_mode.value}' | "
                f"Entity={router_output.primary_entity_id} ({router_output.primary_entity_type}) | "
                f"Attr={router_output.requested_attribute} | "
                f"Filters={router_output.validated_filters} | "
                f"Source={router_output.routing_source}"
            )
            return router_output

        except Exception as e:
            logger.warning(f"Groq router failed to parse JSON ({e}). Reverting to controlled heuristic fallback.")
            return self._fallback_route(clean_q)
