"""Tests for Config class and config loading."""

import os
import tempfile
from pathlib import Path

import pytest

# Ensure src is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import Config, ConfigError


class TestConfigDefaults:
    """Config with a nonexistent config file should use defaults."""

    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch):
        """Provide a fake API key so Config.__init__ doesn't raise."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-unit-tests")

    def test_llm_defaults(self):
        cfg = Config(config_path="/tmp/nonexistent_study_buddy.yaml")
        assert cfg.llm.model == "meta-llama/llama-4-scout-17b-16e-instruct"
        assert cfg.llm.api_base == "https://api.groq.com/openai/v1"

    def test_neo4j_defaults(self):
        cfg = Config(config_path="/tmp/nonexistent_study_buddy.yaml")
        assert cfg.neo4j.url == "bolt://localhost:7687"
        assert cfg.neo4j.username == "neo4j"

    def test_graphrag_defaults(self):
        cfg = Config(config_path="/tmp/nonexistent_study_buddy.yaml")
        assert cfg.graphrag.max_paths_per_chunk == 2
        assert cfg.graphrag.extraction_prompt == "kg_extract_template.txt"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            Config(config_path="/tmp/nonexistent.yaml")


class TestConfigFromFile:
    """Config loading from YAML files."""

    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-unit-tests")

    def test_load_from_yaml(self, tmp_path):
        config_file = tmp_path / "study_buddy.yaml"
        config_file.write_text(
            "llm:\n"
            "  model: my-custom-model\n"
            "  api_base: https://custom.api/v1\n"
            "graphrag:\n"
            "  max_paths_per_chunk: 10\n"
            "  extraction_prompt: my_prompt.txt\n"
        )
        cfg = Config(config_path=str(config_file))
        assert cfg.llm.model == "my-custom-model"
        assert cfg.llm.api_base == "https://custom.api/v1"
        assert cfg.graphrag.max_paths_per_chunk == 10
        assert cfg.graphrag.extraction_prompt == "my_prompt.txt"

    def test_config_reload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config_file = tmp_path / "study_buddy.yaml"
        config_file.write_text(
            "llm:\n"
            "  model: initial-model\n"
        )
        cfg = Config(config_path=str(config_file))
        assert cfg.llm.model == "initial-model"

        # Rewrite file and reload
        config_file.write_text(
            "llm:\n"
            "  model: reloaded-model\n"
        )
        cfg.reload()
        assert cfg.llm.model == "reloaded-model"

    def test_config_singleton(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        Config.reset()
        c1 = Config.get()
        c2 = Config.get()
        assert c1 is c2
        Config.reset()