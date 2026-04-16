"""Tests for prompt versioning and frontmatter handling.

Verifies:
1. YAML frontmatter is correctly parsed from prompt templates
2. Frontmatter is stripped when templates are loaded for LLM use
3. PromptVersion data class holds metadata correctly
4. KGExtractionSignature and KGExtractionModule work standalone
5. DSPy conversion fails gracefully when dspy is not installed
"""

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.prompt_versioning import (
    KGExtractionModule,
    KGExtractionSignature,
    PromptVersion,
    dspy_available,
    load_versioned_prompt,
    parse_prompt_frontmatter,
)


# ------------------------------------------------------------------
# Frontmatter parsing tests
# ------------------------------------------------------------------


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        text = "This is just a plain prompt template with {variable}."
        metadata, content = parse_prompt_frontmatter(text)
        assert metadata == {}
        assert content == text

    def test_simple_frontmatter(self):
        text = "---\nversion: 3\ndescription: \"Test prompt\"\n---\n-Goal-\nExtract stuff from {text}."
        metadata, content = parse_prompt_frontmatter(text)
        assert metadata["version"] == 3
        assert metadata["description"] == "Test prompt"
        assert "-Goal-" in content
        assert "{text}" in content
        assert "---" not in content

    def test_frontmatter_with_all_fields(self):
        text = (
            "---\n"
            "version: 2\n"
            "description: \"Coreference-aware extraction\"\n"
            "model_target: gpt-4\n"
            "last_evaluated: 2026-04-15\n"
            "eval_score: 0.85\n"
            "---\n"
            "The actual prompt content here."
        )
        metadata, content = parse_prompt_frontmatter(text)
        assert metadata["version"] == 2
        assert metadata["description"] == "Coreference-aware extraction"
        assert metadata["model_target"] == "gpt-4"
        assert metadata["last_evaluated"] == "2026-04-15"
        assert metadata["eval_score"] == 0.85
        assert content.strip() == "The actual prompt content here."

    def test_frontmatter_with_quoted_strings(self):
        text = '---\nversion: 1\ndescription: "A prompt with \\"quotes\\""\n---\nContent'
        metadata, content = parse_prompt_frontmatter(text)
        assert metadata["version"] == 1
        assert "quotes" in metadata["description"]

    def test_version_type_coercion(self):
        text = "---\nversion: 5\neval_score: 0.92\n---\nContent"
        metadata, _ = parse_prompt_frontmatter(text)
        assert isinstance(metadata["version"], int)
        assert isinstance(metadata["eval_score"], float)
        assert metadata["version"] == 5
        assert metadata["eval_score"] == 0.92

    def test_empty_frontmatter(self):
        text = "---\n---\nActual content"
        metadata, content = parse_prompt_frontmatter(text)
        assert metadata == {}
        assert content.strip() == "Actual content"

    def test_multiline_content_after_frontmatter(self):
        text = "---\nversion: 1\n---\nLine 1\nLine 2\nLine 3"
        _, content = parse_prompt_frontmatter(text)
        assert "Line 1" in content
        assert "Line 3" in content


class TestLoadVersionedPrompt:
    def test_load_versioned_prompt(self, tmp_path):
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text(
            "---\nversion: 2\ndescription: \"Test\"\nmodel_target: gpt-4\n---\nActual prompt {text}.",
            encoding="utf-8",
        )
        pv = load_versioned_prompt(prompt_file)
        assert pv.filename == "test_prompt.txt"
        assert pv.version == 2
        assert pv.description == "Test"
        assert pv.model_target == "gpt-4"
        assert "{text}" in pv.content
        assert "---" not in pv.content

    def test_load_unversioned_prompt(self, tmp_path):
        prompt_file = tmp_path / "plain.txt"
        prompt_file.write_text("Just a prompt {text}", encoding="utf-8")
        pv = load_versioned_prompt(prompt_file)
        assert pv.version == 1  # default
        assert pv.content == "Just a prompt {text}"


# ------------------------------------------------------------------
# PromptVersion data class tests
# ------------------------------------------------------------------


class TestPromptVersion:
    def test_defaults(self):
        pv = PromptVersion(filename="test.txt")
        assert pv.version == 1
        assert pv.description == ""
        assert pv.model_target == ""
        assert pv.eval_score == 0.0

    def test_custom_values(self):
        pv = PromptVersion(
            filename="kg_extract.txt",
            version=3,
            description="Coreference resolution",
            model_target="gpt-4",
            last_evaluated="2026-04-16",
            eval_score=0.85,
            content="Prompt text",
        )
        assert pv.version == 3
        assert pv.eval_score == 0.85
        assert pv.content == "Prompt text"


# ------------------------------------------------------------------
# KGExtractionSignature tests
# ------------------------------------------------------------------


class TestKGExtractionSignature:
    def test_default_instruction(self):
        sig = KGExtractionSignature()
        assert "entities" in sig.instruction.lower() or "extract" in sig.instruction.lower()

    def test_default_entity_types(self):
        sig = KGExtractionSignature()
        assert "Person" in sig.entity_types
        assert "Organization" in sig.entity_types
        assert len(sig.entity_types) == 10  # including "Other"

    def test_custom_instruction(self):
        sig = KGExtractionSignature(instruction="Custom extraction task")
        assert sig.instruction == "Custom extraction task"

    def test_to_dspy_without_dspy(self):
        sig = KGExtractionSignature()
        if not dspy_available():
            with pytest.raises(ImportError, match="DSPy is not installed"):
                sig.to_dspy()


# ------------------------------------------------------------------
# KGExtractionModule tests
# ------------------------------------------------------------------


class TestKGExtractionModule:
    def test_default_parse(self):
        module = KGExtractionModule()
        llm_output = json.dumps({
            "entities": [
                {"entity_name": "Test", "entity_type": "Concept", "entity_description": "A test entity"},
            ],
            "relationships": [
                {"source_entity": "A", "target_entity": "B", "relation": "created", "relationship_description": "A created B"},
            ],
        })
        entities, rels = module._default_parse(llm_output)
        assert len(entities) == 1
        assert entities[0] == ("Test", "Concept", "A test entity")
        assert len(rels) == 1
        assert rels[0] == ("A", "B", "created", "A created B")

    def test_default_parse_no_json(self):
        module = KGExtractionModule()
        entities, rels = module._default_parse("No JSON here")
        assert entities == []
        assert rels == []

    def test_forward_requires_llm(self):
        module = KGExtractionModule()
        with pytest.raises(ValueError, match="llm callable is required"):
            module.forward(text="test")

    def test_forward_with_mock_llm(self):
        mock_llm = MagicMock(return_value=json.dumps({
            "entities": [{"entity_name": "X", "entity_type": "Concept", "entity_description": "Desc"}],
            "relationships": [],
        }))
        module = KGExtractionModule(
            prompt_template="Extract from: {text}",
        )
        entities, rels = module.forward(text="test document", llm=mock_llm)
        assert len(entities) == 1
        assert entities[0][0] == "X"


# ------------------------------------------------------------------
# PromptRegistry frontmatter integration tests
# ------------------------------------------------------------------


class TestPromptRegistryFrontmatter:
    """Test that PromptRegistry correctly strips frontmatter from templates."""

    def test_frontmatter_stripped_from_template(self, tmp_path):
        """Templates loaded via PromptRegistry should not contain frontmatter."""
        from core.prompts import PromptRegistry

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test_prompt.txt").write_text(
            "---\nversion: 42\ndescription: \"Should be stripped\"\n---\nActual template with {text}",
            encoding="utf-8",
        )

        reg = PromptRegistry(prompts_dir=prompts_dir)
        # This should fail because we need to register the template
        # Let's just test _load strips frontmatter
        content = reg._load("test_prompt.txt")
        assert "---" not in content
        assert "version" not in content
        assert "Actual template with {text}" in content

    def test_template_formatting_after_frontmatter_strip(self, tmp_path):
        """format_map should work after frontmatter is stripped."""
        from core.prompts import PromptRegistry

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "fmt_test.txt").write_text(
            "---\nversion: 1\n---\nHello {person}, you have {count} items.",
            encoding="utf-8",
        )

        reg = PromptRegistry(prompts_dir=prompts_dir)
        result = reg.get("fmt_test.txt", person="Alice", count=5)
        assert result == "Hello Alice, you have 5 items."

    def test_unversioned_template_still_works(self, tmp_path):
        """Templates without frontmatter should still load correctly."""
        from core.prompts import PromptRegistry

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "plain.txt").write_text(
            "Plain prompt {text}",
            encoding="utf-8",
        )

        reg = PromptRegistry(prompts_dir=prompts_dir)
        content = reg._load("plain.txt")
        assert content == "Plain prompt {text}"

    def test_version_info(self, tmp_path):
        """version_info() should return PromptVersion with metadata."""
        from core.prompts import PromptRegistry

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "test.txt").write_text(
            "---\nversion: 5\ndescription: \"Versioned prompt\"\nmodel_target: claude-3\n---\nContent {var}",
            encoding="utf-8",
        )

        reg = PromptRegistry(prompts_dir=prompts_dir)
        pv = reg.version_info("test.txt")
        assert pv.version == 5
        assert pv.description == "Versioned prompt"
        assert pv.model_target == "claude-3"