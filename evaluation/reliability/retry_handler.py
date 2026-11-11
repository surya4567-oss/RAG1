"""Robust retry handler with exponential backoff, jitter, and 429/50x detection."""

import random
import re
import time
from typing import Any, Callable, Optional, Tuple, TypeVar

from src.ingestion.logging_config import logger

T = TypeVar("T")


class RetryHandler:
    """Manages bounded retries with exponential backoff and jitter for LLM API calls."""

    RATE_LIMIT_STATUS_CODES = {429}
    SERVER_ERROR_STATUS_CODES = {500, 502, 503, 504}

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        max_delay: float = 30.0,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max(0, max_retries)
        self.initial_delay = max(0.1, initial_delay)
        self.backoff_factor = max(1.0, backoff_factor)
        self.max_delay = max(self.initial_delay, max_delay)
        self.jitter = jitter

    def is_retryable_exception(self, exc: Exception) -> Tuple[bool, bool, Optional[float]]:
        """Determine if an exception is retryable.

        Returns:
            Tuple[is_retryable, is_rate_limit, retry_after_seconds]
        """
        err_msg = str(exc).lower()

        # Check for explicit HTTP status codes
        for code in self.RATE_LIMIT_STATUS_CODES:
            if str(code) in err_msg or "rate limit" in err_msg or "resourceexhausted" in err_msg:
                # Look for Retry-After or wait time in error message
                match = re.search(r"retry[-_ ]after[:\s]+(\d+(?:\.\d+)?)", err_msg)
                retry_after = float(match.group(1)) if match else None
                return True, True, retry_after

        for code in self.SERVER_ERROR_STATUS_CODES:
            if str(code) in err_msg or "bad gateway" in err_msg or "service unavailable" in err_msg:
                return True, False, None

        # Network timeouts or connection resets
        if "timeout" in err_msg or "connection" in err_msg or "econnreset" in err_msg:
            return True, False, None

        return False, False, None

    def execute_with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        operation_name: str = "LLM Call",
        **kwargs: Any,
    ) -> T:
        """Execute a callable with retry policy.

        Raises:
            Exception: If all retries are exhausted or a non-retryable exception occurs.
        """
        attempt = 0
        current_delay = self.initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                attempt += 1
                is_retryable, is_rate_limit, retry_after = self.is_retryable_exception(exc)

                if not is_retryable or attempt > self.max_retries:
                    logger.error(
                        "[%s] Permanent failure after %d/%d attempts: %s",
                        operation_name,
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    raise

                # Calculate sleep duration
                if retry_after is not None and retry_after > 0:
                    sleep_time = min(retry_after, self.max_delay)
                    logger.warning(
                        "[%s] Rate limited (HTTP 429). Provider specified Retry-After: %.2fs. Waiting...",
                        operation_name,
                        sleep_time,
                    )
                else:
                    sleep_time = min(current_delay, self.max_delay)
                    if self.jitter:
                        sleep_time = sleep_time * (0.8 + 0.4 * random.random())

                    reason = "Rate limit" if is_rate_limit else "Server/Network error"
                    logger.warning(
                        "[%s] Attempt %d/%d failed (%s: %s). Retrying in %.2fs...",
                        operation_name,
                        attempt,
                        self.max_retries,
                        reason,
                        exc,
                        sleep_time,
                    )
                    current_delay *= self.backoff_factor

                time.sleep(sleep_time)

