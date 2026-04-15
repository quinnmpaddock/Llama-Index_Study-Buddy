"""Study Buddy GraphRAG API — thin FastAPI router.

All business logic has been extracted into service modules under
``services/``.  This file wires routes to the services and manages
application lifecycle via the ``lifespan`` context manager.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Load environment variables before importing config
load_dotenv()

# --- Configuration ---
from config import get_config, ConfigError

config = get_config()
logging.basicConfig(level=config.server.log_level)
logger = logging.getLogger(__name__)

# --- Service imports ---
from services.community import CommunityService
from services.graph import GraphService
from services.query import QueryService
from services.ingestion import IngestionService

# --- Workspace imports ---
from workspace import WorkspaceRegistry

# --- Models ---
from models import (
    QueryRequest,
    GraphQueryResponse,
    WorkspaceCreate,
    WorkspaceInfo,
    WorkspaceListResponse,
    WorkspaceStatsResponse,
)

# --- Default workspace ID ---
DEFAULT_WORKSPACE_ID = "default"


def _migrate_data_dir(old_dir: Path, new_dir: Path) -> None:
    """One-time migration: move app files from old data/ to new app_data/ directory.

    This handles the case where Neo4j's Docker volume was previously sharing
    the data/ directory with the Python app. After this migration, Neo4j uses
    neo4j_data/ and the app uses app_data/.
    """
    import shutil

    app_items = ["workspaces.json", "default"]

    migrated = False
    for item in app_items:
        src = old_dir / item
        if src.exists():
            dst = new_dir / item
            if not dst.exists():
                new_dir.mkdir(parents=True, exist_ok=True)
                if src.is_file():
                    shutil.copy2(src, dst)
                    logger.info("Migrated %s to app_data/", item)
                elif src.is_dir():
                    shutil.copytree(src, dst)
                    logger.info("Migrated %s/ to app_data/", item)
                migrated = True
            else:
                logger.debug("Skipping migration of %s — already exists in app_data/", item)

    if migrated:
        logger.info("App data migration complete (old data/ now belongs to Neo4j only)")


def _migrate_legacy_summaries(data_dir: Path, _legacy_dir: Optional[Path] = None) -> None:
    """One-time migration: copy files from legacy summaries/ to data/default/summaries/.

    If data/default/summaries/ doesn't exist or is empty but summaries/ does,
    copy everything over.  This ensures backward compatibility when upgrading
    from the single-workspace layout.
    """
    import shutil

    if _legacy_dir is None:
        legacy_dir = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "summaries"
    else:
        legacy_dir = _legacy_dir
    target_dir = data_dir / "default" / "summaries"

    if target_dir.exists() and any(target_dir.iterdir()):
        # Already have workspace-scoped data — skip migration
        return

    if not legacy_dir.exists() or not legacy_dir.is_dir():
        # No legacy data to migrate
        return

    logger.info("Migrating legacy summaries/ to %s...", target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in legacy_dir.iterdir():
        src = legacy_dir / item.name
        dst = target_dir / item.name
        if src.is_file():
            if not dst.exists():
                shutil.copy2(src, dst)
                logger.debug("Migrated %s", item.name)
            else:
                logger.debug("Skipping existing %s", item.name)

    logger.info("Migration complete: %d files migrated", len(list(target_dir.iterdir())))


class EntitySearchResponse(BaseModel):
    """Response for entity search."""

    entities: List[Dict[str, object]]
    total: int


class EntityDetail(BaseModel):
    """Details for a single entity."""

    name: str
    communities: List[int]


class CommunityListResponse(BaseModel):
    """List of all communities."""

    communities: List[Dict[str, object]]
    total: int


class CommunityDetail(BaseModel):
    """Details for a single community."""

    id: int
    summary: str
    entity_count: int


class CommunityEntitiesResponse(BaseModel):
    """Entities belonging to a community."""

    community_id: int
    entities: List[str]
    total: int


class IngestRequest(BaseModel):
    """Request for document ingestion."""

    directory: str = Field(..., description="Directory path containing documents")
    files: Optional[List[str]] = Field(
        None, description="Specific files to ingest (if None, ingest all in directory)"
    )


class IngestResponse(BaseModel):
    """Response from ingestion."""

    status: str
    directory: str
    files_processed: List[str]
    total_nodes: int
    total_entities: int = 0
    total_relationships: int = 0
    communities_built: int = 0
    message: str
    task_id: Optional[str] = None


class SummaryVersion(BaseModel):
    """A version of community summaries."""

    version: str
    created_at: str
    files: Dict[str, str]
    stats: Dict[str, int]


class SummaryListResponse(BaseModel):
    """Response for listing summary versions."""

    current: Optional[SummaryVersion]
    versions: List[Dict[str, str]]


class SummaryCleanupResponse(BaseModel):
    """Response for cleanup operation."""

    deleted: List[str]
    kept: List[str]
    message: str


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown of the GraphRAG components."""
    import time

    from llama_index.core import PropertyGraphIndex, Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai_like import OpenAILike

    from core_classes import GraphRAGQueryEngine, GraphRAGStore

    logger.info("Initializing Study Buddy GraphRAG Engine...")
    start_time = time.time()

    try:
        # 1. Initialize data directory, migrate legacy summaries, set up workspace registry
        data_dir = Path(os.environ.get(
            "STUDY_BUDDY_DATA_DIR",
            os.path.join(os.path.dirname(__file__), "..", "app_data"),
        ))

        old_data_dir = Path(os.path.join(os.path.dirname(__file__), "..", "data"))
        if old_data_dir.exists() and data_dir != old_data_dir:
            _migrate_data_dir(old_data_dir, data_dir)

        _migrate_legacy_summaries(data_dir)

        logger.info("[STARTUP] Step 1: Loading embedding model...")
        logger.info("[STARTUP] Using embedding model: %s", config.embedding.model)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=config.embedding.model
        )
        logger.info("[STARTUP] Step 1 complete (%.2fs)", time.time() - start_time)

        if not config.llm.api_key:
            raise ConfigError("OPENAI_API_KEY environment variable is required")

        logger.info("[STARTUP] Using LLM: %s @ %s", config.llm.model, config.llm.api_base)
        Settings.llm = OpenAILike(
            model=config.llm.model,
            api_base=config.llm.api_base,
            api_key=config.llm.api_key,
            is_chat_model=True,
        )

        # 2. Load Persisted Summaries (via CommunityService)
        logger.info("[STARTUP] Step 2: Loading summaries...")
        community_svc = CommunityService(data_dir=str(data_dir))
        community_summaries, entity_info = community_svc.load_summaries_and_entity_info(
            workspace_id=DEFAULT_WORKSPACE_ID
        )
        logger.info("[STARTUP] Step 2 complete (%.2fs)", time.time() - start_time)

        # 3. Initialize Store and Index
        logger.info("[STARTUP] Step 3: Initializing GraphRAGStore...")
        logger.info("[STARTUP] Connecting to Neo4j at %s", config.neo4j.url)
        graph_store = GraphRAGStore(
            username=config.neo4j.username,
            password=config.neo4j.password,
            url=config.neo4j.url,
            community_summary=community_summaries,
            entity_info=entity_info,
            refresh_schema=False,
            create_indexes=False,
            timeout=config.neo4j.timeout,
        )
        logger.info("[STARTUP] Step 3a complete (%.2fs)", time.time() - start_time)

        logger.info("[STARTUP] Step 3b: Creating PropertyGraphIndex.from_existing...")
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store, embed_model=Settings.embed_model
        )
        logger.info("[STARTUP] Step 3 complete (%.2fs)", time.time() - start_time)

        # 4. Initialize Query Engine
        logger.info("[STARTUP] Step 4: Initializing QueryEngine...")
        new_engine = GraphRAGQueryEngine(
            graph_store=graph_store, index=index, llm=Settings.llm
        )
        logger.info("[STARTUP] Step 4 complete (%.2fs)", time.time() - start_time)

        # 5. Store data on app.state
        new_summaries = {str(k): v for k, v in community_summaries.items()}
        new_summaries_loaded = len(community_summaries) > 0

        app.state.engine = new_engine
        app.state.community_summaries = new_summaries
        app.state.entity_info = entity_info
        app.state.summaries_loaded = new_summaries_loaded

        # 6. Initialise services and attach to app.state
        graph_svc = GraphService(state=app.state)
        query_svc = QueryService(state=app.state)
        ingestion_svc = IngestionService(config=config)
        ingestion_svc.attach_state(app.state)

        app.state.community_svc = community_svc
        app.state.graph_svc = graph_svc
        app.state.query_svc = query_svc
        app.state.ingestion_svc = ingestion_svc

        # 7. Initialise WorkspaceRegistry and auto-create default workspace
        workspace_registry = WorkspaceRegistry(data_dir=data_dir)
        if workspace_registry.get(DEFAULT_WORKSPACE_ID) is None:
            workspace_registry.create(name="Default", description="Default workspace (auto-created)")
            logger.info("Auto-created default workspace")
        app.state.workspace_registry = workspace_registry

        if not app.state.summaries_loaded:
            logger.warning(
                "Started with empty knowledge graph. "
                "Use 'sb ingest <directory>' to add data before querying."
            )

        logger.info(
            "GraphRAG Engine successfully initialized in %.2fs",
            time.time() - start_time,
        )
    except Exception as e:
        logger.error("Failed to initialize engine: %s", str(e))
        raise

    yield
    logger.info("Shutting down...")


# --- FastAPI Application ---
app = FastAPI(
    title="Study Buddy GraphRAG API",
    description="A dynamically queryable API for Knowledge Graph RAG",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Root ---
@app.get("/")
async def root():
    return {
        "message": "Study Buddy GraphRAG API is online. Go to /docs for Swagger UI."
    }


# ============================================================
# Workspace (Knowledge Base) Management Endpoints
# ============================================================

@app.post("/kb", response_model=WorkspaceInfo, status_code=201)
async def create_workspace(request: WorkspaceCreate):
    """Create a new knowledge base workspace."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    try:
        ws = registry.create(name=request.name, description=request.description)
        return WorkspaceInfo(
            id=ws.id,
            name=ws.name,
            description=ws.description,
            neo4j_database=ws.neo4j_database,
            created_at=ws.created_at,
            updated_at=ws.updated_at,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/kb", response_model=WorkspaceListResponse)
async def list_workspaces():
    """List all workspaces."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    workspaces = registry.list()
    return WorkspaceListResponse(
        workspaces=[
            WorkspaceInfo(
                id=ws.id, name=ws.name, description=ws.description,
                neo4j_database=ws.neo4j_database, created_at=ws.created_at,
                updated_at=ws.updated_at,
            )
            for ws in workspaces
        ],
        total=len(workspaces),
    )


@app.get("/kb/{workspace_id}", response_model=WorkspaceInfo)
async def get_workspace(workspace_id: str):
    """Get workspace details."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")
    return WorkspaceInfo(
        id=ws.id, name=ws.name, description=ws.description,
        neo4j_database=ws.neo4j_database, created_at=ws.created_at,
        updated_at=ws.updated_at,
    )


@app.delete("/kb/{workspace_id}")
async def delete_workspace(workspace_id: str):
    """Delete a workspace and its data directory.

    Does NOT drop the Neo4j database — that must be done separately.
    """
    registry: WorkspaceRegistry = app.state.workspace_registry
    if workspace_id == DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=403, detail="Cannot delete the default workspace")
    if not registry.delete(workspace_id):
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")
    return {"status": "deleted", "workspace_id": workspace_id}


# ============================================================
# Workspace-scoped endpoints (/kb/{workspace_id}/...)
# ============================================================

@app.post("/kb/{workspace_id}/query", response_model=GraphQueryResponse)
async def query_workspace(workspace_id: str, request: QueryRequest):
    """Query a specific workspace's knowledge graph."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    # Currently only the default workspace has a loaded engine
    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(
            status_code=501,
            detail="Multi-workspace querying not yet implemented. Use the default workspace.",
        )

    if not hasattr(app.state, "engine"):
        raise HTTPException(status_code=503, detail="Engine not initialized")
    if not getattr(app.state, "summaries_loaded", False):
        raise HTTPException(status_code=503, detail="No data ingested")

    try:
        response = await app.state.engine.acustom_query(request.query, similarity_top_k=request.similarity_top_k)
        return QueryService.format_response(response)
    except Exception as e:
        logger.exception("Query error for workspace %s", workspace_id)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/kb/{workspace_id}/ingest", response_model=IngestResponse)
async def ingest_workspace(
    workspace_id: str,
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """Ingest documents into a specific workspace."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(
            status_code=501,
            detail="Multi-workspace ingestion not yet implemented. Use the default workspace.",
        )

    svc: IngestionService = app.state.ingestion_svc
    try:
        task_id, response_data, files_to_process = svc.start_ingestion(
            directory=request.directory,
            files=request.files,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise HTTPException(status_code=400, detail=str(exc))
        if isinstance(exc, RuntimeError):
            raise HTTPException(status_code=500, detail=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))

    if not task_id:
        return IngestResponse(**response_data)

    background_tasks.add_task(
        svc.run_ingestion,
        str(request.directory),
        files_to_process,
        task_id,
    )
    return IngestResponse(**response_data)


@app.get("/kb/{workspace_id}/entities", response_model=EntitySearchResponse)
async def search_workspace_entities(
    workspace_id: str,
    q: Optional[str] = Query(None, description="Search term for entity names"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results to return"),
):
    """Search for entities in a specific workspace's knowledge graph."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=501, detail="Multi-workspace not yet implemented.")

    if not hasattr(app.state, "entity_info"):
        raise HTTPException(status_code=503, detail="Entity info not loaded")

    return app.state.graph_svc.search_entities(query=q, limit=limit)


@app.get("/kb/{workspace_id}/entities/{name}", response_model=EntityDetail)
async def get_workspace_entity(workspace_id: str, name: str):
    """Get details for a specific entity by name in a workspace."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=501, detail="Multi-workspace not yet implemented.")

    if not hasattr(app.state, "entity_info"):
        raise HTTPException(status_code=503, detail="Entity info not loaded")

    result = app.state.graph_svc.get_entity(name=name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
    return result


@app.get("/kb/{workspace_id}/communities", response_model=CommunityListResponse)
async def list_workspace_communities(workspace_id: str):
    """List all communities in a specific workspace."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=501, detail="Multi-workspace not yet implemented.")

    if not hasattr(app.state, "community_summaries"):
        raise HTTPException(status_code=503, detail="Community summaries not loaded")

    return app.state.graph_svc.list_communities()


@app.get("/kb/{workspace_id}/communities/{id}", response_model=CommunityDetail)
async def get_workspace_community(workspace_id: str, id: int):
    """Get details for a specific community in a workspace."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=501, detail="Multi-workspace not yet implemented.")

    if not hasattr(app.state, "community_summaries"):
        raise HTTPException(status_code=503, detail="Community summaries not loaded")

    result = app.state.graph_svc.get_community(id=id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Community {id} not found")
    return result


@app.get("/kb/{workspace_id}/communities/{id}/entities", response_model=CommunityEntitiesResponse)
async def get_workspace_community_entities(workspace_id: str, id: int):
    """Get all entities belonging to a specific community in a workspace."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=501, detail="Multi-workspace not yet implemented.")

    if not hasattr(app.state, "community_summaries"):
        raise HTTPException(status_code=503, detail="Community summaries not loaded")

    result = app.state.graph_svc.get_community_entities(id=id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Community {id} not found")
    return result


@app.get("/kb/{workspace_id}/ingest/status/{task_id}")
async def get_workspace_ingestion_status(workspace_id: str, task_id: str):
    """Get the status of a background ingestion task in a workspace."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=501, detail="Multi-workspace not yet implemented.")

    svc: IngestionService = app.state.ingestion_svc
    status = svc.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return status


@app.get("/kb/{workspace_id}/ingest/preview")
async def preview_workspace_ingest(
    workspace_id: str,
    directory: str = Query(..., description="Directory path to preview"),
):
    """Preview what files would be ingested from a directory for a workspace."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=501, detail="Multi-workspace not yet implemented.")

    try:
        return IngestionService.preview_directory(directory)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/kb/{workspace_id}/summaries", response_model=SummaryListResponse)
async def list_workspace_summaries(workspace_id: str):
    """List all available summary versions for a workspace."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=501, detail="Multi-workspace not yet implemented.")

    svc: CommunityService = app.state.community_svc
    current, versions = svc.list_versions(workspace_id=workspace_id)
    current_version = SummaryVersion(**current) if current else None
    return SummaryListResponse(current=current_version, versions=versions)


@app.get("/kb/{workspace_id}/summaries/current")
async def get_workspace_current_summary(workspace_id: str):
    """Get the current (active) summary version info for a workspace."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=501, detail="Multi-workspace not yet implemented.")

    svc: CommunityService = app.state.community_svc
    data = svc.get_current_version(workspace_id=workspace_id)
    if data is None:
        raise HTTPException(status_code=404, detail="No current summary version found")
    return data


@app.get("/kb/{workspace_id}/summaries/{version}")
async def get_workspace_summary_version(workspace_id: str, version: str):
    """Get a specific summary version's content for a workspace."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=501, detail="Multi-workspace not yet implemented.")

    svc: CommunityService = app.state.community_svc
    data = svc.get_version(version, workspace_id=workspace_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Summary version '{version}' not found")
    return data


@app.delete("/kb/{workspace_id}/summaries", response_model=SummaryCleanupResponse)
async def cleanup_workspace_summaries(
    workspace_id: str,
    keep: int = Query(5, description="Number of versions to keep"),
):
    """Delete old summary versions for a workspace, keeping the N most recent."""
    registry: WorkspaceRegistry = app.state.workspace_registry
    ws = registry.get(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found")

    if workspace_id != DEFAULT_WORKSPACE_ID:
        raise HTTPException(status_code=501, detail="Multi-workspace not yet implemented.")

    if keep < 1:
        raise HTTPException(status_code=400, detail="Must keep at least 1 version")

    svc: CommunityService = app.state.community_svc
    try:
        deleted, kept = svc.cleanup_versions(workspace_id=workspace_id, keep=keep)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return SummaryCleanupResponse(
        deleted=deleted,
        kept=kept,
        message=f"Deleted {len(deleted) // 2} version(s), keeping {len(kept)} version(s)",
    )


# ============================================================
# Legacy Endpoints (default workspace — unchanged)
# ============================================================


# --- Query Endpoint ---
@app.post("/query", response_model=GraphQueryResponse)
async def query_graph(request: QueryRequest):
    """Submit a query to the GraphRAG engine."""
    if not hasattr(app.state, "engine"):
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if not getattr(app.state, "summaries_loaded", False):
        raise HTTPException(
            status_code=503,
            detail="No data ingested. Run 'sb ingest <directory>' first.",
        )

    try:
        response = await app.state.engine.acustom_query(request.query, similarity_top_k=request.similarity_top_k)
        return QueryService.format_response(response)
    except Exception as e:
        logger.exception("Query error")
        raise HTTPException(status_code=500, detail="Internal server error")


# --- Entity Endpoints ---
@app.get("/entities", response_model=EntitySearchResponse)
async def search_entities(
    q: Optional[str] = Query(None, description="Search term for entity names"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results to return"),
):
    """Search for entities in the knowledge graph."""
    if not hasattr(app.state, "entity_info"):
        raise HTTPException(status_code=503, detail="Entity info not loaded")

    result = app.state.graph_svc.search_entities(query=q, limit=limit)
    return result


@app.get("/entities/{name}", response_model=EntityDetail)
async def get_entity(name: str):
    """Get details for a specific entity by name."""
    if not hasattr(app.state, "entity_info"):
        raise HTTPException(status_code=503, detail="Entity info not loaded")

    result = app.state.graph_svc.get_entity(name=name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
    return result


# --- Community Endpoints ---
@app.get("/communities", response_model=CommunityListResponse)
async def list_communities():
    """List all communities with entity counts."""
    if not hasattr(app.state, "community_summaries"):
        raise HTTPException(status_code=503, detail="Community summaries not loaded")

    return app.state.graph_svc.list_communities()


@app.get("/communities/{id}", response_model=CommunityDetail)
async def get_community(id: int):
    """Get details for a specific community including its summary."""
    if not hasattr(app.state, "community_summaries"):
        raise HTTPException(status_code=503, detail="Community summaries not loaded")

    result = app.state.graph_svc.get_community(id=id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Community {id} not found")
    return result


@app.get("/communities/{id}/entities", response_model=CommunityEntitiesResponse)
async def get_community_entities(id: int):
    """Get all entities belonging to a specific community."""
    if not hasattr(app.state, "community_summaries"):
        raise HTTPException(status_code=503, detail="Community summaries not loaded")

    result = app.state.graph_svc.get_community_entities(id=id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Community {id} not found")
    return result


# --- Ingestion Endpoints ---
@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """Ingest documents from a directory into the knowledge graph.

    This runs the full pipeline in the background:
    1. Extract nodes from documents
    2. Extract entities and relationships using LLM
    3. Build knowledge graph in Neo4j
    4. Generate community summaries
    """
    svc: IngestionService = app.state.ingestion_svc

    try:
        task_id, response_data, files_to_process = svc.start_ingestion(
            directory=request.directory,
            files=request.files,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        # Map known errors to HTTP errors
        if isinstance(exc, FileNotFoundError):
            raise HTTPException(status_code=400, detail=str(exc))
        if isinstance(exc, RuntimeError):
            raise HTTPException(status_code=500, detail=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))

    if not task_id:
        # No files to process — return warning response
        return IngestResponse(**response_data)

    # Enqueue background task
    background_tasks.add_task(
        svc.run_ingestion,
        str(request.directory),
        files_to_process,
        task_id,
    )

    return IngestResponse(**response_data)


@app.get("/ingest/status/{task_id}")
async def get_ingestion_status(task_id: str):
    """Get the status of a background ingestion task."""
    svc: IngestionService = app.state.ingestion_svc
    status = svc.get_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return status


@app.get("/ingest/preview")
async def preview_ingest(
    directory: str = Query(..., description="Directory path to preview"),
):
    """Preview what files would be ingested from a directory."""
    try:
        return IngestionService.preview_directory(directory)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# --- Summary Management Endpoints ---
@app.get("/summaries", response_model=SummaryListResponse)
async def list_summaries():
    """List all available summary versions."""
    svc: CommunityService = app.state.community_svc
    current, versions = svc.list_versions(workspace_id=DEFAULT_WORKSPACE_ID)
    current_version = SummaryVersion(**current) if current else None
    return SummaryListResponse(current=current_version, versions=versions)


@app.get("/summaries/current")
async def get_current_summary():
    """Get the current (active) summary version info."""
    svc: CommunityService = app.state.community_svc
    data = svc.get_current_version(workspace_id=DEFAULT_WORKSPACE_ID)
    if data is None:
        raise HTTPException(status_code=404, detail="No current summary version found")
    return data


@app.get("/summaries/{version}")
async def get_summary_version(version: str):
    """Get a specific summary version's content."""
    svc: CommunityService = app.state.community_svc
    data = svc.get_version(version, workspace_id=DEFAULT_WORKSPACE_ID)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Summary version '{version}' not found")
    return data


@app.delete("/summaries", response_model=SummaryCleanupResponse)
async def cleanup_summaries(
    keep: int = Query(5, description="Number of versions to keep"),
):
    """Delete old summary versions, keeping the N most recent."""
    if keep < 1:
        raise HTTPException(status_code=400, detail="Must keep at least 1 version")

    svc: CommunityService = app.state.community_svc
    try:
        deleted, kept = svc.cleanup_versions(workspace_id=DEFAULT_WORKSPACE_ID, keep=keep)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return SummaryCleanupResponse(
        deleted=deleted,
        kept=kept,
        message=f"Deleted {len(deleted) // 2} version(s), keeping {len(kept)} version(s)",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)