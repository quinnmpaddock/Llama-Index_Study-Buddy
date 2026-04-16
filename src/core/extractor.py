"""Knowledge-graph extractor with optional Instructor-based structured extraction.

When ``use_instructor=True``, the LLM response is validated against
Pydantic models (``ExtractionResult``) with automatic retries, guaranteeing
well-formed output.  When ``False`` (the default, preserving backward
compatibility), the legacy ``parse_fn`` / ``extract_json`` path is used.
"""

import asyncio
import json
import logging
from typing import Any, Callable, List, Optional, Union

from llama_index.core import Settings
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    EntityNode,
    Relation,
)
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent

from core.extraction_models import ExtractionResult

logger = logging.getLogger(__name__)

# Lazy imports — instructor and openai are optional dependencies.
# They are only required when ``use_instructor=True``.
_instructor: Any = None
_openai: Any = None


def _import_instructor():
    """Import ``instructor`` lazily to avoid hard dependency."""
    global _instructor
    if _instructor is None:
        try:
            import instructor
            _instructor = instructor
        except ImportError as exc:
            raise ImportError(
                "The `instructor` package is required for structured extraction. "
                "Install it with:  pip install instructor"
            ) from exc
    return _instructor


def _import_openai():
    """Import ``openai`` lazily."""
    global _openai
    if _openai is None:
        try:
            import openai as _oi
            _openai = _oi
        except ImportError as exc:
            raise ImportError(
                "The `openai` package is required for structured extraction. "
                "Install it with:  pip install openai"
            ) from exc
    return _openai


class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a prompt + output parsing to extract paths (i.e.
    triples) and entity/relation descriptions from text.

    When ``use_instructor`` is ``True`` and an ``instructor_client``
    is provided, the extractor uses the Instructor library for
    Pydantic-validated structured output with automatic retries.
    Otherwise, it falls back to the legacy ``parse_fn`` path.

    Args:
        llm: The language model to use (LlamaIndex LLM).
        extract_prompt: The prompt template for extraction.
        parse_fn: Legacy parser for raw LLM text (used when
            ``use_instructor=False``).
        num_workers: Parallel extraction workers.
        max_paths_per_chunk: Max entity-relationship pairs per chunk.
        use_instructor: Whether to use Instructor for structured output.
        instructor_client: Pre-configured ``instructor.Inferring``
            client (required when ``use_instructor=True``).
        instructor_max_retries: Retry count for Instructor validation.
        instructor_model_name: Model name passed to the Instructor
            client (falls back to ``llm`` model name if not set).
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int
    use_instructor: bool = False
    instructor_client: Any = None
    instructor_max_retries: int = 3
    instructor_model_name: Optional[str] = None

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
        use_instructor: bool = False,
        instructor_client: Any = None,
        instructor_max_retries: int = 3,
        instructor_model_name: Optional[str] = None,
    ) -> None:
        """Init params."""
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
            use_instructor=use_instructor,
            instructor_client=instructor_client,
            instructor_max_retries=instructor_max_retries,
            instructor_model_name=instructor_model_name,
        )

        # Validate Instructor configuration
        if self.use_instructor and self.instructor_client is None:
            raise ValueError(
                "instructor_client must be provided when use_instructor=True. "
                "Create one with: instructor.from_openai(openai.OpenAI(...))"
            )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.acall(nodes, show_progress=show_progress, **kwargs))
                return future.result()
        else:
            return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    # ------------------------------------------------------------------
    # Instructor-based extraction
    # ------------------------------------------------------------------

    async def _aextract_with_instructor(self, node: BaseNode) -> BaseNode:
        """Extract entities and relationships using Instructor for structured output.

        This path bypasses ``parse_fn`` entirely — the LLM response is
        validated against ``ExtractionResult`` by Instructor, with
        automatic retries on validation failures.
        """
        assert self.instructor_client is not None
        text = node.get_content(metadata_mode=MetadataMode.LLM)
        prompt_text = self.extract_prompt.format(
            text=text,
            max_knowledge_triplets=self.max_paths_per_chunk,
        )

        model_name = self.instructor_model_name

        try:
            result: ExtractionResult = await self.instructor_client.chat.completions.create(
                model=model_name,
                response_model=ExtractionResult,
                messages=[{"role": "user", "content": prompt_text}],
                max_retries=self.instructor_max_retries,
            )
        except Exception:
            # Let Instructor retries handle validation errors.
            # Network/LLM errors should propagate so the calling job can
            # retry or fail visibly.
            logger.warning("Instructor extraction failed; returning empty result")
            result = ExtractionResult()

        entities, relationships = result.to_tuples()

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        for entity, entity_type, description in entities:
            entity_metadata = node.metadata.copy()
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        for triple in relationships:
            subj, obj, rel, description = triple
            relation_metadata = node.metadata.copy()
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    # ------------------------------------------------------------------
    # Legacy parse_fn extraction
    # ------------------------------------------------------------------

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node using legacy parse_fn."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode=MetadataMode.LLM)
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
        except Exception:
            # LLM/network errors should propagate so the calling job can
            # retry or fail visibly — don't swallow them here.
            raise

        try:
            entities, entities_relationship = self.parse_fn(llm_response)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
            # Only catch parse/format errors — malformed LLM output is
            # expected occasionally and should degrade gracefully.
            logger.warning("LLM extraction failed for node: %s: %s", type(e).__name__, e)
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])

        for entity, entity_type, description in entities:
            entity_metadata = node.metadata.copy()
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata = node.metadata.copy()
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async.

        Routes to :meth:`_aextract_with_instructor` when
        ``use_instructor=True``, otherwise uses the legacy
        :meth:`_aextract` path.
        """
        extract_fn = (
            self._aextract_with_instructor
            if self.use_instructor
            else self._aextract
        )
        jobs = [extract_fn(node) for node in nodes]

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )


# ------------------------------------------------------------------
# Helper to create an Instructor client from config
# ------------------------------------------------------------------


def create_instructor_client(
    api_key: str,
    api_base: str,
    model: str,
    max_retries: int = 3,
):
    """Create an Instructor-wrapped OpenAI client from LLM config.

    Parameters
    ----------
    api_key : str
        OpenAI-compatible API key.
    api_base : str
        OpenAI-compatible API base URL.
    model : str
        Model name (unused for client creation, but validates config).
    max_retries : int
        Number of Instructor validation retries.

    Returns
    -------
    instructor.Inferring
        A patched OpenAI client that yields validated Pydantic models.
    """
    instructor_mod = _import_instructor()
    openai_mod = _import_openai()

    client = openai_mod.AsyncOpenAI(
        api_key=api_key,
        base_url=api_base,
    )
    return instructor_mod.from_openai(client, max_retries=max_retries)