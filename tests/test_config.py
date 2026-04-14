"""Tests for WorkspaceConfig and Config.workspace_defaults()."""

import os
import tempfile
from pathlib import Path

import pytest

# Ensure src is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import WorkspaceConfig, Config, ConfigError


# ---------------------------------------------------------------------------
# WorkspaceConfig default construction
# ---------------------------------------------------------------------------

class TestWorkspaceConfigDefaults:
    """WorkspaceConfig with no overrides should have all None fields."""

    def test_all_fields_default_to_none(self):
        ws = WorkspaceConfig()
        assert ws.llm_model is None
        assert ws.llm_api_base is None
        assert ws.embedding_model is None
        assert ws.max_paths_per_chunk is None
        assert ws.extraction_prompt is None
        assert ws.neo4j_database is None

    def test_explicit_overrides(self):
        ws = WorkspaceConfig(
            llm_model="gpt-4",
            llm_api_base="https://api.openai.com/v1",
            embedding_model="text-embedding-3-small",
            max_paths_per_chunk=5,
            extraction_prompt="custom_template.txt",
            neo4j_database="research_db",
        )
        assert ws.llm_model == "gpt-4"
        assert ws.llm_api_base == "https://api.openai.com/v1"
        assert ws.embedding_model == "text-embedding-3-small"
        assert ws.max_paths_per_chunk == 5
        assert ws.extraction_prompt == "custom_template.txt"
        assert ws.neo4j_database == "research_db"


# ---------------------------------------------------------------------------
# WorkspaceConfig.resolve()
# ---------------------------------------------------------------------------

class TestWorkspaceConfigResolve:
    """WorkspaceConfig.resolve() should merge overrides with global defaults."""

    GLOBAL_DEFAULTS = {
        "llm_model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "llm_api_base": "https://api.groq.com/openai/v1",
        "embedding_model": "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
        "max_paths_per_chunk": 2,
        "extraction_prompt": "kg_extract_template.txt",
        "neo4j_database": "neo4j",
    }

    def test_resolve_no_overrides_returns_defaults(self):
        ws = WorkspaceConfig()
        result = ws.resolve(self.GLOBAL_DEFAULTS)
        assert result == self.GLOBAL_DEFAULTS

    def test_resolve_override_takes_precedence(self):
        ws = WorkspaceConfig(llm_model="gpt-4", max_paths_per_chunk=10)
        result = ws.resolve(self.GLOBAL_DEFAULTS)
        assert result["llm_model"] == "gpt-4"
        assert result["max_paths_per_chunk"] == 10
        # Non-overridden keys fall back
        assert result["llm_api_base"] == self.GLOBAL_DEFAULTS["llm_api_base"]
        assert result["embedding_model"] == self.GLOBAL_DEFAULTS["embedding_model"]
        assert result["extraction_prompt"] == self.GLOBAL_DEFAULTS["extraction_prompt"]

    def test_resolve_all_overridden(self):
        ws = WorkspaceConfig(
            llm_model="custom-model",
            embedding_model="custom-embed",
            max_paths_per_chunk=7,
            extraction_prompt="custom_prompt.txt",
            neo4j_database="test_db",
        )
        result = ws.resolve(self.GLOBAL_DEFAULTS)
        assert result["llm_model"] == "custom-model"
        assert result["embedding_model"] == "custom-embed"
        assert result["max_paths_per_chunk"] == 7
        assert result["extraction_prompt"] == "custom_prompt.txt"
        assert result["neo4j_database"] == "test_db"

    def test_resolve_missing_global_keys_get_none(self):
        ws = WorkspaceConfig()
        # If global_defaults doesn't have a key, result should be None
        result = ws.resolve({"llm_model": "m1"})
        assert result["llm_model"] == "m1"
        assert result["llm_api_base"] is None
        assert result["embedding_model"] is None


# ---------------------------------------------------------------------------
# WorkspaceConfig.from_yaml()
# ---------------------------------------------------------------------------

class TestWorkspaceConfigFromYaml:
    """WorkspaceConfig.from_yaml() loading from YAML files."""

    def test_from_yaml_nonexistent_path_returns_defaults(self, tmp_path):
        ws = WorkspaceConfig.from_yaml(tmp_path / "does_not_exist.yaml")
        assert ws == WorkspaceConfig()

    def test_from_yaml_with_valid_file(self, tmp_path):
        yaml_file = tmp_path / "workspace.yaml"
        yaml_file.write_text(
            "llm_model: gpt-4o\n"
            "llm_api_base: https://api.openai.com/v1\n"
            "embedding_model: text-embedding-3-small\n"
            "max_paths_per_chunk: 5\n"
            "extraction_prompt: custom_template.txt\n"
            "neo4j_database: research_db\n"
        )
        ws = WorkspaceConfig.from_yaml(yaml_file)
        assert ws.llm_model == "gpt-4o"
        assert ws.llm_api_base == "https://api.openai.com/v1"
        assert ws.embedding_model == "text-embedding-3-small"
        assert ws.max_paths_per_chunk == 5
        assert ws.extraction_prompt == "custom_template.txt"
        assert ws.neo4j_database == "research_db"

    def test_from_yaml_partial_file(self, tmp_path):
        yaml_file = tmp_path / "workspace_partial.yaml"
        yaml_file.write_text("llm_model: claude-3\n")
        ws = WorkspaceConfig.from_yaml(yaml_file)
        assert ws.llm_model == "claude-3"
        assert ws.llm_api_base is None
        assert ws.embedding_model is None

    def test_from_yaml_empty_file(self, tmp_path):
        yaml_file = tmp_path / "workspace_empty.yaml"
        yaml_file.write_text("")
        ws = WorkspaceConfig.from_yaml(yaml_file)
        assert ws == WorkspaceConfig()

    def test_from_yaml_ignores_comments(self, tmp_path):
        yaml_file = tmp_path / "workspace_comments.yaml"
        yaml_file.write_text(
            "# This is a comment\n"
            "llm_model: test-model\n"
            "\n"
            "# Another comment\n"
        )
        ws = WorkspaceConfig.from_yaml(yaml_file)
        assert ws.llm_model == "test-model"


# ---------------------------------------------------------------------------
# Config.workspace_defaults()
# ---------------------------------------------------------------------------

class TestConfigWorkspaceDefaults:
    """Config.workspace_defaults() should return the expected dict."""

    @pytest.fixture(autouse=True)
    def _set_env(self, monkeypatch):
        """Provide a fake API key so Config.__init__ doesn't raise."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-unit-tests")

    def test_workspace_defaults_returns_expected_keys(self):
        # Use a nonexistent config path so defaults are used
        cfg = Config(config_path="/tmp/nonexistent_study_buddy.yaml")
        defaults = cfg.workspace_defaults()
        assert isinstance(defaults, dict)
        assert "llm_model" in defaults
        assert "llm_api_base" in defaults
        assert "embedding_model" in defaults
        assert "max_paths_per_chunk" in defaults
        assert "extraction_prompt" in defaults

    def test_workspace_defaults_values_match_config(self):
        cfg = Config(config_path="/tmp/nonexistent_study_buddy.yaml")
        defaults = cfg.workspace_defaults()
        assert defaults["llm_model"] == cfg.llm.model
        assert defaults["llm_api_base"] == cfg.llm.api_base
        assert defaults["embedding_model"] == cfg.embedding.model
        assert defaults["max_paths_per_chunk"] == cfg.graphrag.max_paths_per_chunk
        assert defaults["extraction_prompt"] == cfg.graphrag.extraction_prompt

    def test_workspace_defaults_with_overridden_config(self, tmp_path, monkeypatch):
        """When config file provides custom values, workspace_defaults reflects them."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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
        defaults = cfg.workspace_defaults()
        assert defaults["llm_model"] == "my-custom-model"
        assert defaults["llm_api_base"] == "https://custom.api/v1"
        assert defaults["max_paths_per_chunk"] == 10
        assert defaults["extraction_prompt"] == "my_prompt.txt"

    def test_workspace_config_resolve_with_config_defaults(self, monkeypatch):
        """Integration: WorkspaceConfig.resolve() with real Config defaults."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        cfg = Config(config_path="/tmp/nonexistent_study_buddy.yaml")
        ws = WorkspaceConfig(llm_model="override-model")
        result = ws.resolve(cfg.workspace_defaults())
        assert result["llm_model"] == "override-model"
        assert result["llm_api_base"] == cfg.llm.api_base
        assert result["embedding_model"] == cfg.embedding.model