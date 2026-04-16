"""Prompt registry — load, cache, and format prompt templates from src/prompts/.

This module provides a centralized, file-based approach to managing LLM
prompt templates.  Instead of hardcoding prompts inline in Python source,
templates live as plain-text files under ``src/prompts/`` and are loaded
on demand through :class:`PromptRegistry`.

Design principles (inspired by Hermes Agent's prompt architecture):

1. **External templates** — prompts live in ``src/prompts/*.txt``, making
   them easy to edit, version-control, and A/B-test without touching Python.
2. **Frozen snapshots** — templates are cached after first load; they never
   change within a session.  This mirrors Hermes's "frozen at session start"
   pattern.
3. **Centralised path resolution** — one place to compute the prompts directory,
   avoiding the ``__file__``-based path bugs that have bitten us before.
4. **Template variable substitution** — uses Python's ``str.format_map()``
   which is compatible with LlamaIndex's ``PromptTemplate`` syntax.
5. **Configurable filenames** — prompt filenames come from
   :class:`~config.GraphRAGConfig`, so they can be overridden via YAML config
   without code changes.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default prompt filenames (matched to GraphRAGConfig defaults)
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS: Dict[str, str] = {
    "kg_extract": "kg_extract_template.txt",
    "community_summary": "community_summary.txt",
    "answer_from_summary": "answer_from_summary.txt",
    "aggregate_answers": "aggregate_answers.txt",
}


class PromptRegistry:
    """Load, cache, and format prompt templates from the prompts directory.

    Parameters
    ----------
    prompts_dir:
        Path to the directory containing ``*.txt`` prompt templates.
        Defaults to ``src/prompts/`` resolved relative to this module.
    config:
        Optional :class:`~config.GraphRAGConfig` instance.  When provided,
        filenames are read from the config object so that YAML overrides
        are respected.  When omitted, :data:`DEFAULT_PROMPTS` is used.

    Usage::

        from core.prompts import PromptRegistry

        registry = PromptRegistry()
        prompt = registry.get("kg_extract", text="Hello", max_knowledge_triplets=2)
        community_prompt = registry.get("community_summary")

        # With config override:
        from config import get_config
        registry = PromptRegistry(config=get_config().graphrag)
    """

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        config: Optional[object] = None,
    ) -> None:
        if prompts_dir is None:
            # Resolve relative to this file: src/core/prompts.py → src/prompts/
            prompts_dir = Path(__file__).resolve().parent.parent / "prompts"
        self._prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}

        # Build filename mapping from config or defaults
        self._filenames: Dict[str, str] = dict(DEFAULT_PROMPTS)
        if config is not None:
            self._filenames["kg_extract"] = getattr(config, "extraction_prompt", DEFAULT_PROMPTS["kg_extract"])
            self._filenames["community_summary"] = getattr(config, "community_summary_prompt", DEFAULT_PROMPTS["community_summary"])
            self._filenames["answer_from_summary"] = getattr(config, "answer_from_summary_prompt", DEFAULT_PROMPTS["answer_from_summary"])
            self._filenames["aggregate_answers"] = getattr(config, "aggregate_answers_prompt", DEFAULT_PROMPTS["aggregate_answers"])

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, name: str, **kwargs: object) -> str:
        """Load a prompt template and optionally format it with variables.

        Parameters
        ----------
        name:
            Template filename (e.g. ``"kg_extract_template.txt"``) **or**
            a short key from the configured filenames (e.g. ``"kg_extract"``).
        **kwargs:
            Template variables to substitute.  If omitted, the raw template
            string is returned without formatting.

        Returns
        -------
        str
            The formatted prompt string.

        Raises
        ------
        FileNotFoundError
            If the template file does not exist.
        KeyError
            If ``kwargs`` contains variables not present in the template.
        """
        filename = self._filenames.get(name, name)
        template = self._load(filename)
        if kwargs:
            return template.format_map(kwargs)
        return template

    def raw(self, name: str) -> str:
        """Return the raw template string without any variable substitution.

        This is useful when you need to pass the template to LlamaIndex's
        ``PromptTemplate()`` which does its own ``{variable}`` substitution.

        Parameters
        ----------
        name:
            Template filename or short key.

        Returns
        -------
        str
            The raw template text (with ``{placeholders}`` intact).
        """
        filename = self._filenames.get(name, name)
        return self._load(filename)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, filename: str) -> str:
        """Load a template file, caching the result.

        Templates are loaded once and cached for the lifetime of the
        registry instance (frozen-snapshot pattern).
        """
        if filename not in self._cache:
            path = self._prompts_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"Prompt template not found: {path}.  "
                    f"Ensure the file exists in {self._prompts_dir}/."
                )
            self._cache[filename] = path.read_text(encoding="utf-8")
            logger.debug("Loaded prompt template: %s", filename)
        return self._cache[filename]

    # ------------------------------------------------------------------
    # Convenience: expose prompt directory for validation
    # ------------------------------------------------------------------

    @property
    def prompts_dir(self) -> Path:
        """The resolved prompts directory path."""
        return self._prompts_dir

    def list_templates(self) -> list[str]:
        """List all ``*.txt`` template files in the prompts directory."""
        if not self._prompts_dir.is_dir():
            return []
        return sorted(
            f.name for f in self._prompts_dir.iterdir()
            if f.suffix == ".txt"
        )

    def clear_cache(self) -> None:
        """Clear the template cache (useful in tests or when hot-reloading)."""
        self._cache.clear()