"""DSPy-compatible extraction interface for prompt optimization.

This module provides DSPy Signatures and Modules that wrap the KG
extraction pipeline, enabling programmatic prompt optimization via
DSPy's MIPROv2 or other optimizers.

Usage::

    # Define the extraction signature
    sig = KGExtractionSignature()
    
    # Create a module
    module = KGExtractionModule(signature=sig)
    
    # Run extraction on a text
    result = module(text="Margaret Hamilton led the team...")
    # result.entities, result.relationships

For optimization, see ``scripts/optimize_prompts.py``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt versioning
# ---------------------------------------------------------------------------

# Each prompt file can include YAML frontmatter with version metadata.
# Example:
#   ---
#   version: 2
#   description: "Added entity type taxonomy and coreference resolution"
#   model_target: gpt-4
#   last_evaluated: 2026-04-16
#   eval_score: 0.85
#   ---
#   (prompt content follows)

_VERSION_PATTERN = re.compile(
    r"^---[ \t]*\n(.*?)\n?---[ \t]*\n", re.DOTALL
)


@dataclass
class PromptVersion:
    """Metadata for a versioned prompt template."""
    filename: str
    version: int = 1
    description: str = ""
    model_target: str = ""
    last_evaluated: str = ""
    eval_score: float = 0.0
    content: str = ""  # The actual prompt content (after frontmatter)


def parse_prompt_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a prompt template file.

    Returns (metadata_dict, content_after_frontmatter).
    If no frontmatter is found, returns ({}, original_text).
    """
    match = _VERSION_PATTERN.match(text)
    if not match:
        return {}, text

    frontmatter = match.group(1)
    content = text[match.end():]

    # Simple YAML parsing for flat key-value pairs
    metadata: dict[str, Any] = {}
    for line in frontmatter.strip().split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Type coercion
        if key == "version":
            metadata[key] = int(value)
        elif key == "eval_score":
            metadata[key] = float(value)
        else:
            metadata[key] = value

    return metadata, content


def load_versioned_prompt(filepath: str | Path) -> PromptVersion:
    """Load a prompt template with version metadata.

    Reads the file, parses frontmatter if present, and returns
    a PromptVersion with both metadata and content.
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8")
    metadata, content = parse_prompt_frontmatter(text)

    return PromptVersion(
        filename=filepath.name,
        version=metadata.get("version", 1),
        description=metadata.get("description", ""),
        model_target=metadata.get("model_target", ""),
        last_evaluated=metadata.get("last_evaluated", ""),
        eval_score=metadata.get("eval_score", 0.0),
        content=content,
    )


# ---------------------------------------------------------------------------
# DSPy Signature (optional — only used when dspy is installed)
# ---------------------------------------------------------------------------

_DSPY_AVAILABLE = False
try:
    import dspy
    _DSPY_AVAILABLE = True
except ImportError:
    pass


def dspy_available() -> bool:
    """Check whether DSPy is installed and available."""
    return _DSPY_AVAILABLE


class KGExtractionSignature:
    """DSPy-compatible signature for KG extraction.

    This is a lightweight wrapper that works with or without DSPy
    installed. When DSPy is available, it can be converted to a
    ``dspy.Signature`` for use with DSPy optimizers.

    The signature captures the input/output contract:
    - Input: ``text`` (the document to extract from)
    - Output: ``entities`` (list of entity tuples),
              ``relationships`` (list of relationship tuples)
    """

    def __init__(
        self,
        instruction: str = "",
        entity_types: list[str] | None = None,
    ):
        self.instruction = instruction or (
            "Given a text document, identify all entities and the "
            "relationships among them, then return the result as structured JSON."
        )
        self.entity_types = entity_types or [
            "Person", "Organization", "Technology", "Location",
            "Event", "Concept", "Document", "Product", "Award", "Other",
        ]

    def to_dspy(self) -> "dspy.Signature":
        """Convert to a DSPy Signature (requires dspy installed)."""
        if not _DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is not installed. Install with: pip install dspy-ai"
            )

        entity_types_str = ", ".join(self.entity_types)
        instruction = (
            f"{self.instruction}\n\n"
            f"Entity types: {entity_types_str}"
        )

        return dspy.Signature(
            "text -> entities, relationships",
            instruction=instruction,
        )


class KGExtractionModule:
    """Wraps KG extraction as a callable module.

    Can be used standalone (without DSPy) or as part of a DSPy
    optimization pipeline. The extraction logic delegates to the
    project's GraphRAGExtractor or a direct LLM call.
    """

    def __init__(
        self,
        signature: KGExtractionSignature | None = None,
        prompt_template: str = "",
        parse_fn: Any | None = None,
    ):
        self.signature = signature or KGExtractionSignature()
        self.prompt_template = prompt_template
        self.parse_fn = parse_fn or self._default_parse

    @staticmethod
    def _default_parse(llm_output: str) -> tuple[list, list]:
        """Default parser: extract JSON from LLM output."""
        match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if not match:
            return [], []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return [], []

        entities = [
            (e["entity_name"], e["entity_type"], e["entity_description"])
            for e in data.get("entities", [])
        ]
        relationships = [
            (
                r["source_entity"],
                r["target_entity"],
                r["relation"],
                r["relationship_description"],
            )
            for r in data.get("relationships", [])
        ]
        return entities, relationships

    def forward(self, text: str, llm: Any = None) -> tuple[list, list]:
        """Run extraction on a single text.

        Parameters
        ----------
        text : str
            The document text to extract from.
        llm : callable, optional
            A callable that takes a prompt string and returns an LLM
            response string. If not provided, extraction cannot run.

        Returns
        -------
        (entities, relationships) : tuple of lists
        """
        if llm is None:
            raise ValueError("llm callable is required for extraction")

        prompt = self.prompt_template.format(text=text, max_knowledge_triplets=10)
        response = llm(prompt)
        return self.parse_fn(response)

    # Alias for DSPy compatibility
    __call__ = forward