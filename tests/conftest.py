import os
import sys
from pathlib import Path
import pytest

# Add project root to sys.path
root_dir = Path(__file__).parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


@pytest.fixture(autouse=True)
def hermetic_env_fixture(monkeypatch):
    """Ensure test suites run hermetically without making live network calls to tunnels in .env."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("ROUTER_LLM_PROVIDER", "groq")
    monkeypatch.setenv("GENERATOR_LLM_PROVIDER", "groq")
    monkeypatch.setenv("JUDGE_LLM_PROVIDER", "groq")
    monkeypatch.setenv("COLAB_LLM_BASE_URL", "")
    monkeypatch.setenv("KAGGLE_LLM_BASE_URL", "")
    monkeypatch.setenv("REMOTE_LLM_BASE_URL", "")
