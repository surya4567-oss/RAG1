"""Unit tests for dataset discovery and source inventory scanner."""

import pytest
from pathlib import Path

from src.ingestion.config import PipelineConfig
from src.ingestion.discovery import DatasetDiscoverer


@pytest.fixture
def discoverer():
    config = PipelineConfig(data_dir=Path("data"))
    return DatasetDiscoverer(config)


def test_discovery_all_categories(discoverer):
    report = discoverer.discover()
    assert report.total_files_discovered > 0
    
    categories = {s.category for s in report.discovered_sources}
    assert "testcases" in categories
    assert "test_execution" in categories
    assert "defects" in categories
    assert "srs" in categories
    assert "failure_logs" in categories
    assert "screenshots" in categories


def test_failure_logs_discovery_10_files(discoverer):
    report = discoverer.discover()
    log_sources = [s for s in report.discovered_sources if s.category == "failure_logs"]
    assert len(log_sources) == 10, f"Expected 10 failure log files, got {len(log_sources)}"
    for s in log_sources:
        assert s.size_bytes > 0
        assert s.record_count > 0


def test_screenshots_discovery_10_files(discoverer):
    report = discoverer.discover()
    shot_sources = [s for s in report.discovered_sources if s.category == "screenshots"]
    assert len(shot_sources) == 10, f"Expected 10 screenshot JSON files, got {len(shot_sources)}"
    for s in shot_sources:
        assert s.size_bytes > 0
        assert s.record_count > 0
