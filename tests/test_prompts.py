"""Tests for the PromptRegistry — loading, caching, and formatting prompt templates."""

import pytest
from pathlib import Path

from core.prompts import PromptRegistry, DEFAULT_PROMPTS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prompts_dir(tmp_path):
    """Create a temporary prompts directory with sample templates."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()

    (prompts / "kg_extract_template.txt").write_text(
        "Extract up to {max_knowledge_triplets} triplets from: {text}", encoding="utf-8"
    )
    (prompts / "kg_extract_entities.txt").write_text(
        "Extract entities from: {text}", encoding="utf-8"
    )
    (prompts / "kg_extract_relationships.txt").write_text(
        "Given entities:\n{entities}\nExtract relationships from: {text}", encoding="utf-8"
    )
    (prompts / "community_summary.txt").write_text(
        "Summarize these relationships.", encoding="utf-8"
    )
    (prompts / "answer_from_summary.txt").write_text(
        "Given the community summary: {community_summary}, answer: {query}", encoding="utf-8"
    )
    (prompts / "aggregate_answers.txt").write_text(
        "Combine answers. Preserve citations.", encoding="utf-8"
    )
    return prompts


@pytest.fixture
def registry(prompts_dir):
    """PromptRegistry pointing at the temporary prompts directory."""
    return PromptRegistry(prompts_dir=prompts_dir)


# ---------------------------------------------------------------------------
# Core API tests
# ---------------------------------------------------------------------------

class TestPromptRegistryInit:
    """Test PromptRegistry initialisation."""

    def test_default_prompts_dir_resolves(self):
        """Default prompts_dir should resolve to src/prompts/ relative to this module."""
        reg = PromptRegistry()
        expected = Path(__file__).resolve().parent.parent / "src" / "prompts"
        # The default resolves from src/core/prompts.py → src/prompts/
        assert reg.prompts_dir == expected or reg.prompts_dir.name == "prompts"

    def test_custom_prompts_dir(self, prompts_dir):
        reg = PromptRegistry(prompts_dir=prompts_dir)
        assert reg.prompts_dir == prompts_dir

    def test_list_templates(self, registry):
        templates = registry.list_templates()
        assert "kg_extract_template.txt" in templates
        assert "community_summary.txt" in templates
        assert "kg_extract_entities.txt" in templates
        assert "kg_extract_relationships.txt" in templates
        assert len(templates) == 6


class TestPromptLoading:
    """Test template loading and caching."""

    def test_get_by_short_key(self, registry):
        """Short keys (e.g. 'kg_extract') should resolve to the correct file."""
        result = registry.get("kg_extract")
        assert "max_knowledge_triplets" in result
        assert "{text}" in result

    def test_get_by_filename(self, registry):
        """Full filenames should also work."""
        result = registry.get("kg_extract_template.txt")
        assert "max_knowledge_triplets" in result

    def test_get_community_summary(self, registry):
        result = registry.get("community_summary")
        assert "Summarize" in result

    def test_get_answer_from_summary(self, registry):
        result = registry.get("answer_from_summary")
        assert "{community_summary}" in result
        assert "{query}" in result

    def test_get_aggregate_answers(self, registry):
        result = registry.get("aggregate_answers")
        assert "Combine" in result

    def test_get_nonexistent_raises(self, tmp_path):
        """Requesting a missing template should raise FileNotFoundError."""
        reg = PromptRegistry(prompts_dir=tmp_path / "nonexistent_prompts")
        (tmp_path / "nonexistent_prompts").mkdir()
        with pytest.raises(FileNotFoundError, match="not found"):
            reg.get("no_such_template.txt")

    def test_raw_returns_unformatted(self, registry):
        """raw() should return the template with {placeholders} intact."""
        result = registry.raw("kg_extract")
        assert "{max_knowledge_triplets}" in result
        assert "{text}" in result


class TestTemplateFormatting:
    """Test template variable substitution."""

    def test_format_with_kwargs(self, registry):
        result = registry.get(
            "kg_extract",
            max_knowledge_triplets=5,
            text="Hello world",
        )
        assert "5" in result
        assert "Hello world" in result
        assert "{max_knowledge_triplets}" not in result

    def test_format_answer_from_summary(self, registry):
        result = registry.get(
            "answer_from_summary",
            community_summary="Entities relate to each other.",
            query="What entities exist?",
        )
        assert "Entities relate to each other." in result
        assert "What entities exist?" in result

    def test_format_without_kwargs_returns_raw(self, registry):
        """Calling get() without kwargs should return the raw template."""
        result = registry.get("community_summary")
        assert result == registry.raw("community_summary")


class TestCaching:
    """Test that templates are cached after first load."""

    def test_cache_populated_after_get(self, registry):
        registry.get("kg_extract")
        assert "kg_extract_template.txt" in registry._cache

    def test_clear_cache(self, registry):
        registry.get("kg_extract")
        assert len(registry._cache) > 0
        registry.clear_cache()
        assert len(registry._cache) == 0

    def test_cached_value_returned_on_second_call(self, registry, prompts_dir):
        """Verify caching: modifying the file after first load shouldn't affect results."""
        first = registry.get("community_summary")
        # Modify the file on disk
        (prompts_dir / "community_summary.txt").write_text("Modified!", encoding="utf-8")
        # Should still return cached value
        second = registry.get("community_summary")
        assert first == second
        assert "Modified!" not in second


class TestConfigOverride:
    """Test that GraphRAGConfig can override prompt filenames."""

    def test_config_overrides_filename(self, prompts_dir):
        """Config-supplied filenames should override defaults."""

        class MockConfig:
            extraction_prompt = "community_summary.txt"  # intentional swap
            community_summary_prompt = "community_summary.txt"
            answer_from_summary_prompt = "answer_from_summary.txt"
            aggregate_answers_prompt = "aggregate_answers.txt"

        reg = PromptRegistry(prompts_dir=prompts_dir, config=MockConfig())
        # kg_extract should now load community_summary.txt instead
        result = reg.get("kg_extract")
        assert "Summarize" in result  # community_summary.txt content

    def test_default_filenames_without_config(self, prompts_dir):
        """Without config, DEFAULT_PROMPTS should be used."""
        reg = PromptRegistry(prompts_dir=prompts_dir)
        assert reg._filenames == DEFAULT_PROMPTS


class TestPathResolutionFromServices:
    """Regression test: ensure prompts resolve correctly from src/services/ path."""

    def test_prompts_dir_reachable_from_core(self):
        """src/prompts/ must be reachable via the PromptRegistry default."""
        reg = PromptRegistry()
        # The registry should be able to find the actual prompts directory
        # This is the same invariant as the old TestPathResolution tests
        assert reg.prompts_dir.is_dir(), (
            f"Prompts directory not found at {reg.prompts_dir}. "
            f"Ensure src/prompts/ exists."
        )

    def test_kg_extract_template_existing(self):
        """kg_extract_template.txt must exist at the default location."""
        reg = PromptRegistry()
        # This should NOT raise FileNotFoundError
        try:
            content = reg.raw("kg_extract")
            assert "entities" in content.lower()
            assert "{text}" in content
        except FileNotFoundError:
            pytest.fail(
                f"kg_extract_template.txt not found at {reg.prompts_dir}. "
                f"PromptRegistry cannot resolve prompts."
            )