"""Shared test fixtures for study-buddy tests."""
import os
import sys
import pytest
from pathlib import Path

# Add project root and src/ to sys.path so 'from src.models import ...' works
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Test data directory
TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_config(tmp_path):
    """Provide a mock Config object for tests that don't need real Neo4j/LLM."""
    # Create a minimal config file
    config_file = tmp_path / "test_study_buddy.yaml"
    config_file.write_text("""
llm:
  model: "test-model"
  api_base: "https://api.test.com/v1"
embedding:
  model: "test-embedding-model"
neo4j:
  url: "bolt://localhost:7687"
  username: "neo4j"
  password: "test-password"
server:
  port: 8000
  host: "127.0.0.1"
  log_level: "WARNING"
graphrag:
  max_paths_per_chunk: 2
  extraction_prompt: "kg_extract_template.txt"
  community_summary_prompt: "community_summary.txt"
  answer_from_summary_prompt: "answer_from_summary.txt"
  aggregate_answers_prompt: "aggregate_answers.txt"
docker:
  container_name: "neo4j-test"
""")
    return str(config_file)