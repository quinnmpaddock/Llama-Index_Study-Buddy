"""Ingestion service — background document processing pipeline.

Manages background ingestion tasks, LLM-based entity/relationship
extraction, and knowledge-graph construction.  Extracted from
``app.py`` to decouple the business logic from the FastAPI routes.
"""

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Supported file extensions for ingestion
SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".html",
    ".xlsx",
    ".md",
    ".csv",
    ".txt",
    ".json",
}


def extract_json(text: str):
    """Extract and parse JSON from LLM output text.

    First tries a fast regex match, then falls back to progressively
    shrinking the substring from the end until valid JSON is found.

    Returns parsed dict on success, ``None`` on failure.
    """
    # Fast path: try regex match first
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass  # Fall through to slow path

    # Slow path: find first { and shrink from end
    start = text.find("{")
    if start == -1:
        return None

    for end in range(len(text), start, -1):
        substring = text[start:end]
        try:
            return json.loads(substring)
        except json.JSONDecodeError:
            continue

    return None


def parse_fn(response_str: str):
    """Parse LLM response for entity/relationship extraction."""
    entities = []
    relationships = []
    data = extract_json(response_str)
    if not data or not isinstance(data, dict):
        return entities, relationships
    try:
        entities = [
            (
                entity["entity_name"],
                entity["entity_type"],
                entity["entity_description"],
            )
            for entity in data.get("entities", [])
        ]
        relationships = [
            (
                relation["source_entity"],
                relation["target_entity"],
                relation["relation"],
                relation["relationship_description"],
            )
            for relation in data.get("relationships", [])
        ]
        return entities, relationships
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Error parsing JSON: %s", e)
        return entities, relationships


class IngestionService:
    """Manage background document ingestion tasks."""

    def __init__(self, config, workspace_registry=None):
        """Initialise the service.

        Parameters
        ----------
        config:
            A ``Config`` object with ``llm``, ``embedding``, ``neo4j``, and
            ``graphrag`` attributes.
        workspace_registry:
            Optional :class:`WorkspaceRegistry` for multi-workspace
            support (currently unused but accepted for forward
            compatibility).
        """
        self.config = config
        self.workspace_registry = workspace_registry
        self._ingestion_status: Dict[str, dict] = {}
        self._state_lock = threading.Lock()
        # ``app.state`` reference — set by :meth:`attach_state` when the
        # FastAPI app is initialised.
        self._app_state = None

    def attach_state(self, state) -> None:
        """(Re-)attach a reference to ``app.state``.

        This is required so that the background ingestion task can
        reload the engine after completion.
        """
        self._app_state = state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_ingestion(
        self,
        directory: str,
        files: Optional[List[str]] = None,
        workspace_id: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """Validate inputs and enqueue a background ingestion task.

        Returns ``(task_id, response_dict)`` where the dict contains the
        immediate HTTP response data (status, file list, etc.).
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Resolve files to process
        dir_root = dir_path.resolve()
        if files:
            files_to_process = self._resolve_files(dir_root, files)
        else:
            files_to_process = [
                str(dir_path / f)
                for f in os.listdir(dir_path)
                if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
            ]

        if not files_to_process:
            response = {
                "status": "warning",
                "directory": str(dir_path),
                "files_processed": [],
                "total_nodes": 0,
                "message": "No supported files found to process",
            }
            return "", response

        # Check for API key
        if not self.config.llm.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it in your shell or .env file."
            )

        task_id = str(uuid.uuid4())
        self._ingestion_status[task_id] = {"status": "queued", "progress": 0}
        return task_id, {
            "status": "processing",
            "directory": str(dir_path.absolute()),
            "files_processed": [Path(f).name for f in files_to_process],
            "total_nodes": 0,
            "task_id": task_id,
            "message": (
                f"Ingestion started in background. "
                f"{len(files_to_process)} file(s) being processed. "
                f"Task ID: {task_id}"
            ),
        }

    def get_status(self, task_id: str) -> Optional[dict]:
        """Return the status dict for a background task, or ``None``."""
        return self._ingestion_status.get(task_id)

    # ------------------------------------------------------------------
    # Background ingestion runner
    # ------------------------------------------------------------------

    def run_ingestion(
        self,
        directory: str,
        files_to_process: List[str],
        task_id: str,
        workspace_id: Optional[str] = None,
    ) -> None:
        """Run the complete ingestion pipeline (called in a background thread).

        This is the core logic extracted from ``app.py``'s
        ``run_full_ingestion()``.
        """
        from llama_index.core import PropertyGraphIndex, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.openai_like import OpenAILike

        from core_classes import GraphRAGExtractor, GraphRAGStore
        from ingestion import DocumentIngestion

        from services.community import CommunityService

        try:
            self._ingestion_status[task_id] = {
                "status": "extracting_nodes",
                "progress": 0,
            }

            # 1. Initialise document ingestion
            ingester = DocumentIngestion()

            if not self.config.llm.api_key:
                self._ingestion_status[task_id] = {
                    "status": "error",
                    "error": "OPENAI_API_KEY not set",
                }
                return

            llm = OpenAILike(
                model=self.config.llm.model,
                api_base=self.config.llm.api_base,
                api_key=self.config.llm.api_key,
                is_chat_model=True,
            )

            # 2. Load extraction prompt
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # prompts/ lives next to src/
            template_path = os.path.join(
                os.path.dirname(base_dir), "src", "prompts", "kg_extract_template.txt"
            )
            # Also try the old location for compat
            if not os.path.exists(template_path):
                template_path = os.path.join(
                    base_dir, "prompts", "kg_extract_template.txt"
                )
            with open(template_path, "r", encoding="utf-8") as f:
                kg_triplet_extract_tmpl = f.read()

            # 3. Extract nodes from documents
            logger.info(
                "Extracting nodes from %d files...", len(files_to_process)
            )

            # Group files by directory
            dir_to_files: dict = {}
            for file_path in files_to_process:
                dir_path = os.path.dirname(file_path)
                if dir_path not in dir_to_files:
                    dir_to_files[dir_path] = []
                dir_to_files[dir_path].append(file_path)

            all_nodes = []
            processed_files = []

            for dir_path_str, files_in_dir in dir_to_files.items():
                try:
                    logger.info("Ingesting directory: %s", dir_path_str)
                    nodes = ingester.ingestion(dir_path_str)

                    for file_path in files_in_dir:
                        file_name = Path(file_path).name
                        file_nodes = [
                            n
                            for n in nodes
                            if n.metadata.get("file_path", "").endswith(file_name)
                        ]
                        all_nodes.extend(file_nodes)
                        processed_files.append(file_name)
                        logger.info("Extracted %d nodes from %s", len(file_nodes), file_name)
                except Exception as exc:
                    logger.warning("Failed to process directory %s: %s", dir_path_str, exc)
                    for file_path in files_in_dir:
                        processed_files.append(f"{Path(file_path).name} (FAILED)")

            self._ingestion_status[task_id] = {
                "status": "building_knowledge_graph",
                "progress": 30,
                "total_nodes": len(all_nodes),
            }

            if not all_nodes:
                self._ingestion_status[task_id] = {
                    "status": "error",
                    "error": "No nodes extracted",
                }
                return

            # 4. Create knowledge graph extractor
            logger.info("Creating GraphRAG extractor...")
            kg_extractor = GraphRAGExtractor(
                llm=llm,
                extract_prompt=kg_triplet_extract_tmpl,
                max_paths_per_chunk=2,
                parse_fn=parse_fn,
            )

            # 5. Connect to Neo4j
            logger.info("Connecting to Neo4j...")
            graph_store = GraphRAGStore(
                username=self.config.neo4j.username,
                password=self.config.neo4j.password,
                url=self.config.neo4j.url,
                refresh_schema=False,
                create_indexes=True,
                timeout=self.config.neo4j.timeout,
            )

            # 6. Build property graph index
            logger.info(
                "Building PropertyGraphIndex with %d nodes...", len(all_nodes)
            )
            self._ingestion_status[task_id] = {
                "status": "extracting_entities",
                "progress": 50,
            }

            index = PropertyGraphIndex(
                nodes=all_nodes,
                kg_extractors=[kg_extractor],
                property_graph_store=graph_store,
                show_progress=True,
            )

            self._ingestion_status[task_id] = {
                "status": "building_communities",
                "progress": 80,
            }

            # 7. Build communities
            logger.info("Building communities...")
            index.property_graph_store.get_community_summaries()

            # 8. Save summaries (via CommunityService)
            community_svc = CommunityService(
                data_dir=self._get_data_dir(workspace_id)
            )
            raw_summaries = index.property_graph_store.community_summary
            entity_info = index.property_graph_store.entity_info

            community_svc.save_summaries(
                workspace_id=workspace_id,
                community_summaries=raw_summaries,
                entity_info=entity_info,
            )

            # 9. Reload engine in app.state
            logger.info("Reloading GraphRAG engine with new summaries...")
            try:
                new_graph_store = GraphRAGStore(
                    username=self.config.neo4j.username,
                    password=self.config.neo4j.password,
                    url=self.config.neo4j.url,
                    community_summary=index.property_graph_store.community_summary,
                    entity_info=index.property_graph_store.entity_info,
                    refresh_schema=False,
                    create_indexes=False,
                    timeout=self.config.neo4j.timeout,
                )

                new_index = PropertyGraphIndex.from_existing(
                    property_graph_store=new_graph_store,
                    embed_model=Settings.embed_model,
                )

                # Import here to avoid circular imports at module level
                from core_classes import GraphRAGQueryEngine

                new_engine = GraphRAGQueryEngine(
                    graph_store=new_graph_store,
                    index=new_index,
                    llm=Settings.llm,
                )

                new_summaries = {
                    str(k): v
                    for k, v in index.property_graph_store.community_summary.items()
                }
                new_entity_info = index.property_graph_store.entity_info
                new_summaries_loaded = len(new_summaries) > 0

                with self._state_lock:
                    if self._app_state is not None:
                        self._app_state.engine = new_engine
                        self._app_state.community_summaries = new_summaries
                        self._app_state.entity_info = new_entity_info
                        self._app_state.summaries_loaded = new_summaries_loaded

                logger.info("GraphRAG engine reloaded successfully.")
            except Exception as reload_error:
                logger.warning("Failed to reload engine: %s", reload_error)

            # 10. Final status
            total_entities = len(index.property_graph_store.entity_info)
            total_communities = len(index.property_graph_store.community_summary)

            self._ingestion_status[task_id] = {
                "status": "completed",
                "progress": 100,
                "total_nodes": len(all_nodes),
                "total_entities": total_entities,
                "total_communities": total_communities,
                "files_processed": processed_files,
            }

        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            import traceback
            traceback.print_exc()
            self._ingestion_status[task_id] = {
                "status": "error",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_data_dir(self, workspace_id: Optional[str]) -> Optional[str]:
        """Return the data directory for a workspace, or ``None`` for legacy."""
        if self.workspace_registry is not None:
            return str(self.workspace_registry.data_dir)
        return None

    def _resolve_files(
        self, dir_root: Path, filenames: List[str]
    ) -> List[str]:
        """Resolve a list of relative filenames against *dir_root*.

        Raises ``ValueError`` on path-traversal or unsupported extensions.
        """
        files_to_process: List[str] = []
        for filename in filenames:
            # Reject absolute paths
            if Path(filename).is_absolute():
                raise ValueError(f"Absolute paths are not allowed: {filename}")

            candidate = (dir_root / filename).resolve()

            # Prevent path traversal
            try:
                candidate.relative_to(dir_root)
            except ValueError:
                raise ValueError(
                    f"Path traversal detected: {filename} is outside the directory"
                )

            # Validate extension
            if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported file extension: {candidate.suffix}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )

            # Check file exists
            if not candidate.exists():
                raise FileNotFoundError(f"File not found: {candidate}")

            files_to_process.append(str(candidate))
        return files_to_process

    # ------------------------------------------------------------------
    # Preview helper (no background task needed)
    # ------------------------------------------------------------------

    @staticmethod
    def preview_directory(directory: str) -> dict:
        """Return a list of ingestible files in *directory*."""
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        files = []
        for f in sorted(os.listdir(dir_path)):
            file_path = dir_path / f
            if file_path.is_file() and Path(f).suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(
                    {
                        "name": f,
                        "extension": Path(f).suffix.lower(),
                        "size_bytes": file_path.stat().st_size,
                    }
                )

        return {
            "directory": str(dir_path.absolute()),
            "supported_extensions": list(SUPPORTED_EXTENSIONS),
            "files": files,
            "total_files": len(files),
        }