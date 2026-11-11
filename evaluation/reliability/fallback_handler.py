"""Fallback handler for graceful model failover and error classification."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

from src.ingestion.logging_config import logger

T = TypeVar("T")


@dataclass
class FailureRecord:
    query_id: str
    stage: str
    error_type: str
    error_message: str
    used_fallback: bool = False


class FallbackHandler:
    """Manages primary-to-fallback execution and structured error recovery."""

    def __init__(self, fallback_enabled: bool = True) -> None:
        self.fallback_enabled = fallback_enabled

    def execute_with_fallback(
        self,
        primary_func: Callable[[], T],
        fallback_func: Optional[Callable[[], T]] = None,
        query_id: str = "unknown",
        operation_name: str = "operation",
    ) -> Tuple[Optional[T], Optional[FailureRecord], bool]:
        """Attempt primary function; if it fails, invoke fallback function.

        Returns:
            Tuple[result, failure_record, used_fallback]
        """
        try:
            result = primary_func()
            return result, None, False
        except Exception as primary_exc:
            logger.warning(
                "[%s] Primary execution failed for query %s: %s",
                operation_name,
                query_id,
                primary_exc,
            )

            if self.fallback_enabled and fallback_func is not None:
                try:
                    logger.info("[%s] Attempting fallback for query %s", operation_name, query_id)
                    result = fallback_func()
                    return (
                        result,
                        FailureRecord(
                            query_id=query_id,
                            stage=operation_name,
                            error_type="PrimaryFailed_FallbackRecovered",
                            error_message=str(primary_exc),
                            used_fallback=True,
                        ),
                        True,
                    )
                except Exception as fallback_exc:
                    logger.error(
                        "[%s] Fallback execution also failed for query %s: %s",
                        operation_name,
                        query_id,
                        fallback_exc,
                    )
                    return (
                        None,
                        FailureRecord(
                            query_id=query_id,
                            stage=operation_name,
                            error_type="AllModelsFailed",
                            error_message=f"Primary: {primary_exc} | Fallback: {fallback_exc}",
                            used_fallback=True,
                        ),
                        True,
                    )

            return (
                None,
                FailureRecord(
                    query_id=query_id,
                    stage=operation_name,
                    error_type=type(primary_exc).__name__,
                    error_message=str(primary_exc),
                    used_fallback=False,
                ),
                False,
            )

