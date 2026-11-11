"""Timing utilities and percentile calculation for latency profiling."""

import time
from typing import Dict, List, Optional
import numpy as np


class StageTimer:
    """Context manager to measure elapsed execution time in seconds."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "StageTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self.start_time


def compute_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Calculate mean, median, P90, P95, min, and max for a list of latencies in seconds."""
    if not latencies:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    arr = np.array(latencies, dtype=float)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

