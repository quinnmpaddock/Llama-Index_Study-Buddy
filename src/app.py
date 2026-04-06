import json
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.base.response.schema import Response
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from pydantic import BaseModel, Field

# Import custom classes from local directory
from core_classes import (GraphQueryResponse, GraphRAGExtractor,
                          GraphRAGQueryEngine, GraphRAGStore)
from ingestion import DocumentIngestion

# Load environment variables
load_dotenv()

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants (mirroring main.py)
NEO4JPASSWORD = "neo4j2026"
NEO4J_URL = "bolt://localhost:7687"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARIES_PATH = os.path.join(BASE_DIR, "..", "summaries", "community_summaries.json")
ENTITY_INFO_PATH = os.path.join(BASE_DIR, "..", "summaries", "entity_info.json")


# --- API Models ---
class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="The query to ask the knowledge graph",
        examples=["What are the main news topics discussed?"],
    )
    similarity_top_k: int = Field(default=20, ge=1, le=50)


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown of the GraphRAG components."""
    logger.info("Initializing Study Buddy GraphRAG Engine...")

    try:
        # 1. Setup Models
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
        )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        Settings.llm = OpenAILike(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_base="https://api.groq.com/openai/v1",
            api_key=api_key,
            is_chat_model=True,
        )
        # 2. Load Persisted Summaries
        if not os.path.exists(SUMMARIES_PATH):
            logger.error(
                f"Summaries file not found at {SUMMARIES_PATH}. Please run main.py first."
            )
            raise FileNotFoundError(f"Missing {SUMMARIES_PATH}")

        with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
            raw_summaries = json.load(f)

        community_summaries = {int(k): v for k, v in raw_summaries.items()}
        logger.info(f"Loaded {len(community_summaries)} community summaries.")

        if not os.path.exists(ENTITY_INFO_PATH):
            logger.error(
                f"Entity info file not found at {ENTITY_INFO_PATH}. Please run main.py first."
            )
            raise FileNotFoundError(f"Missing {ENTITY_INFO_PATH}")

        with open(ENTITY_INFO_PATH, "r", encoding="utf-8") as f:
            entity_info = json.load(f)
        logger.info(f"Loaded {len(entity_info)} entity mappings.")

        # 3. Initialize Store and Index
        # We pass the loaded summaries directly to the store
        graph_store = GraphRAGStore(
            username="neo4j",
            password=NEO4JPASSWORD,
            url=NEO4J_URL,
            community_summary=community_summaries,
            entity_info=entity_info,
        )

        # Initialize PropertyGraphIndex from the existing store
        # Note: We don't need to pass nodes here as we are querying an existing graph
        index = PropertyGraphIndex.from_existing(
            property_graph_store=graph_store, embed_model=Settings.embed_model
        )

        # 4. Initialize Query Engine
        app.state.engine = GraphRAGQueryEngine(
            graph_store=graph_store, index=index, llm=Settings.llm
        )

        # 5. Store data for API endpoints
        app.state.community_summaries = {
            str(k): v for k, v in community_summaries.items()
        }
        app.state.entity_info = entity_info

        logger.info("GraphRAG Engine successfully initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize engine: {str(e)}")
        raise e

    yield
    logger.info("Shutting down...")


# --- FastAPI Application ---
app = FastAPI(
    title="Study Buddy GraphRAG API",
    description="A dynamically queryable API for Knowledge Graph RAG",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "message": "Study Buddy GraphRAG API is online. Go to /docs for Swagger UI."
    }


@app.post("/query", response_model=GraphQueryResponse)
async def query_graph(request: QueryRequest):
    """
    Submit a query to the GraphRAG engine.
    Returns the answer along with communities consulted and entities found.
    """
    if not hasattr(app.state, "engine"):
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        # Update similarity_top_k if provided in request
        app.state.engine.similarity_top_k = request.similarity_top_k

        # Execute the async query
        response = await app.state.engine.acustom_query(request.query)
        return {
            "answer": response.response,
            "communities_consulted": response.metadata.get("communities_consulted", []),
            "entities_found": response.metadata.get("entities_found", []),
        }
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Entity Models ---
class EntitySearchResponse(BaseModel):
    """Response for entity search."""

    entities: List[Dict[str, object]]
    total: int


class EntityDetail(BaseModel):
    """Details for a single entity."""

    name: str
    communities: List[int]


# --- Community Models ---
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


# --- Entity Endpoints ---
@app.get("/entities", response_model=EntitySearchResponse)
async def search_entities(
    q: Optional[str] = Query(None, description="Search term for entity names"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results to return"),
):
    """
    Search for entities in the knowledge graph.
    If no query is provided, returns all entities.
    """
    if not hasattr(app.state, "entity_info"):
        raise HTTPException(status_code=503, detail="Entity info not loaded")

    entity_info = app.state.entity_info

    if q:
        # Case-insensitive search
        q_lower = q.lower()
        matches = [
            {"name": name, "communities": list(set(communities))}
            for name, communities in entity_info.items()
            if q_lower in name.lower()
        ]
        # Sort by relevance (exact match first, then by name length)
        matches.sort(key=lambda x: (x["name"].lower() != q_lower, len(x["name"])))
    else:
        matches = [
            {"name": name, "communities": list(set(communities))}
            for name, communities in entity_info.items()
        ]
        matches.sort(key=lambda x: x["name"].lower())

    return {"entities": matches[:limit], "total": len(matches)}


@app.get("/entities/{name}", response_model=EntityDetail)
async def get_entity(name: str):
    """
    Get details for a specific entity by name.
    """
    if not hasattr(app.state, "entity_info"):
        raise HTTPException(status_code=503, detail="Entity info not loaded")

    entity_info = app.state.entity_info

    # Case-insensitive lookup
    for entity_name, communities in entity_info.items():
        if entity_name.lower() == name.lower():
            return {"name": entity_name, "communities": list(set(communities))}

    raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")


# --- Community Endpoints ---
@app.get("/communities", response_model=CommunityListResponse)
async def list_communities():
    """
    List all communities with entity counts.
    """
    if not hasattr(app.state, "community_summaries"):
        raise HTTPException(status_code=503, detail="Community summaries not loaded")

    summaries = app.state.community_summaries

    # Build community list with entity counts from entity_info
    entity_info = app.state.entity_info
    community_entities: Dict[int, List[str]] = {}

    for entity_name, communities in entity_info.items():
        for comm_id in communities:
            if comm_id not in community_entities:
                community_entities[comm_id] = []
            community_entities[comm_id].append(entity_name)

    communities = [
        {
            "id": comm_id,
            "entity_count": len(set(community_entities.get(comm_id, []))),
            "summary_preview": summaries.get(str(comm_id), "")[:100] + "...",
        }
        for comm_id in sorted(summaries.keys(), key=int)
    ]

    return {"communities": communities, "total": len(communities)}


@app.get("/communities/{id}", response_model=CommunityDetail)
async def get_community(id: int):
    """
    Get details for a specific community including its summary.
    """
    if not hasattr(app.state, "community_summaries"):
        raise HTTPException(status_code=503, detail="Community summaries not loaded")

    summaries = app.state.community_summaries
    summary = summaries.get(str(id))

    if summary is None:
        raise HTTPException(status_code=404, detail=f"Community {id} not found")

    # Count entities in this community
    entity_info = app.state.entity_info
    entity_count = sum(1 for communities in entity_info.values() if id in communities)

    return {"id": id, "summary": summary, "entity_count": entity_count}


@app.get("/communities/{id}/entities", response_model=CommunityEntitiesResponse)
async def get_community_entities(id: int):
    """
    Get all entities belonging to a specific community.
    """
    if not hasattr(app.state, "community_summaries"):
        raise HTTPException(status_code=503, detail="Community summaries not loaded")

    if str(id) not in app.state.community_summaries:
        raise HTTPException(status_code=404, detail=f"Community {id} not found")

    entity_info = app.state.entity_info
    entities = [name for name, communities in entity_info.items() if id in communities]

    return {
        "community_id": id,
        "entities": sorted(set(entities)),
        "total": len(entities),
    }


# --- Ingestion Endpoints ---
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


# Ingestion state tracking (for background tasks)
ingestion_status: Dict[str, dict] = {}


def parse_fn(response_str: str):
    """Parse LLM response for entity/relationship extraction."""
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, response_str, re.DOTALL)
    entities = []
    relationships = []
    if not match:
        return entities, relationships
    json_str = match.group(0)
    try:
        data = json.loads(json_str)
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
    except json.JSONDecodeError as e:
        logger.warning(f"Error parsing JSON: {e}")
        return entities, relationships


def run_full_ingestion(
    directory: str,
    files_to_process: List[str],
    task_id: str,
):
    """Run the complete ingestion pipeline (called as background task)."""
    try:
        ingestion_status[task_id] = {"status": "extracting_nodes", "progress": 0}

        # Initialize components
        ingester = DocumentIngestion()

        # Load LLM configuration
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            ingestion_status[task_id] = {
                "status": "error",
                "error": "OPENAI_API_KEY not set",
            }
            return

        llm = OpenAILike(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_base="https://api.groq.com/openai/v1",
            api_key=api_key,
            is_chat_model=True,
        )

        # Load extraction prompt
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(BASE_DIR, "prompts", "kg_extract_template.txt")
        with open(template_path, "r", encoding="utf-8") as f:
            KG_TRIPLET_EXTRACT_TMPL = f.read()

        # Extract nodes from documents
        logger.info(f"Extracting nodes from {len(files_to_process)} files...")
        all_nodes = []
        processed_files = []

        for file_path in files_to_process:
            try:
                logger.info(f"Processing: {file_path}")
                nodes = ingester.ingestion(os.path.dirname(file_path))
                # Filter to only nodes from this file
                from pathlib import Path

                file_nodes = [
                    n
                    for n in nodes
                    if n.metadata.get("file_path", "").endswith(Path(file_path).name)
                ]
                all_nodes.extend(file_nodes)
                processed_files.append(Path(file_path).name)
                logger.info(
                    f"Extracted {len(file_nodes)} nodes from {Path(file_path).name}"
                )
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                processed_files.append(f"{Path(file_path).name} (FAILED)")

        ingestion_status[task_id] = {
            "status": "building_knowledge_graph",
            "progress": 30,
            "total_nodes": len(all_nodes),
        }

        if not all_nodes:
            ingestion_status[task_id] = {
                "status": "error",
                "error": "No nodes extracted",
            }
            return

        # Create knowledge graph extractor
        logger.info("Creating GraphRAG extractor...")
        kg_extractor = GraphRAGExtractor(
            llm=llm,
            extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
            max_paths_per_chunk=2,
            parse_fn=parse_fn,
        )

        # Connect to Neo4j
        logger.info("Connecting to Neo4j...")

        graph_store = GraphRAGStore(
            username="neo4j",
            password=NEO4JPASSWORD,
            url=NEO4J_URL,
        )

        # Build property graph index
        logger.info(f"Building PropertyGraphIndex with {len(all_nodes)} nodes...")
        ingestion_status[task_id] = {"status": "extracting_entities", "progress": 50}

        index = PropertyGraphIndex(
            nodes=all_nodes,
            kg_extractors=[kg_extractor],
            property_graph_store=graph_store,
            show_progress=True,
        )

        ingestion_status[task_id] = {"status": "building_communities", "progress": 80}

        # Build communities and generate summaries
        logger.info("Building communities...")
        index.property_graph_store.get_community_summaries()

        # Save community summaries and entity info with timestamp
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = os.path.join(BASE_DIR, "..", "summaries")
        os.makedirs(output_dir, exist_ok=True)

        # Create timestamped filenames
        summary_path = os.path.join(output_dir, f"community_summaries_{timestamp}.json")
        entity_info_path = os.path.join(output_dir, f"entity_info_{timestamp}.json")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(index.property_graph_store.community_summary, f, indent=4)
        logger.info(f"Community summaries saved to {summary_path}")

        with open(entity_info_path, "w", encoding="utf-8") as f:
            json.dump(index.property_graph_store.entity_info, f, indent=4)
        logger.info(f"Entity info saved to {entity_info_path}")

        # Update current.json pointer
        current_path = os.path.join(output_dir, "current.json")
        current_info = {
            "version": timestamp,
            "created_at": datetime.now().isoformat(),
            "files": {
                "community_summaries": f"community_summaries_{timestamp}.json",
                "entity_info": f"entity_info_{timestamp}.json",
            },
            "stats": {
                "total_entities": len(index.property_graph_store.entity_info),
                "total_communities": len(index.property_graph_store.community_summary),
            },
        }
        with open(current_path, "w", encoding="utf-8") as f:
            json.dump(current_info, f, indent=4)
        logger.info(f"Current version updated to {timestamp}")

        # Calculate stats
        total_entities = len(index.property_graph_store.entity_info)
        total_communities = len(index.property_graph_store.community_summary)

        ingestion_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "total_nodes": len(all_nodes),
            "total_entities": total_entities,
            "total_communities": total_communities,
            "files_processed": processed_files,
        }

    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        import traceback

        traceback.print_exc()
        ingestion_status[task_id] = {"status": "error", "error": str(e)}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ingest documents from a directory into the knowledge graph.

    This runs the full pipeline:
    1. Extract nodes from documents
    2. Extract entities and relationships using LLM
    3. Build knowledge graph in Neo4j
    4. Generate community summaries

    If `files` is provided, only those specific files will be processed.
    If `files` is None, all supported documents in the directory will be processed.
    """
    import os
    import uuid
    from pathlib import Path

    # Validate directory exists
    dir_path = Path(request.directory)
    if not dir_path.exists():
        raise HTTPException(
            status_code=400, detail=f"Directory not found: {request.directory}"
        )
    if not dir_path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Path is not a directory: {request.directory}"
        )

    # Determine files to process
    if request.files:
        files_to_process = []
        for filename in request.files:
            file_path = dir_path / filename
            if not file_path.exists():
                raise HTTPException(
                    status_code=400, detail=f"File not found: {file_path}"
                )
            files_to_process.append(str(file_path))
    else:
        supported_extensions = {
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
        files_to_process = [
            str(dir_path / f)
            for f in os.listdir(dir_path)
            if Path(f).suffix.lower() in supported_extensions
        ]

    if not files_to_process:
        return IngestResponse(
            status="warning",
            directory=str(dir_path),
            files_processed=[],
            total_nodes=0,
            message="No supported files found to process",
        )

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable not set. Please set it in .env file.",
        )

    # Generate task ID and start background processing
    task_id = str(uuid.uuid4())
    ingestion_status[task_id] = {"status": "queued", "progress": 0}

    background_tasks.add_task(
        run_full_ingestion,
        str(dir_path),
        files_to_process,
        task_id,
    )

    # Return immediate response with task ID
    from pathlib import Path

    file_names = [Path(f).name for f in files_to_process]

    return IngestResponse(
        status="processing",
        directory=str(dir_path.absolute()),
        files_processed=file_names,
        total_nodes=0,
        message=f"Ingestion started in background. {len(files_to_process)} file(s) being processed. Task ID: {task_id}",
    )


@app.get("/ingest/status/{task_id}")
async def get_ingestion_status(task_id: str):
    """
    Get the status of a background ingestion task.
    """
    if task_id not in ingestion_status:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return ingestion_status[task_id]


@app.get("/ingest/preview")
async def preview_ingest(
    directory: str = Query(..., description="Directory path to preview")
):
    """
    Preview what files would be ingested from a directory.
    Returns list of supported files without actually ingesting them.
    """
    import os
    from pathlib import Path

    dir_path = Path(directory)
    if not dir_path.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")
    if not dir_path.is_dir():
        raise HTTPException(
            status_code=400, detail=f"Path is not a directory: {directory}"
        )

    supported_extensions = {
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

    files = []
    for f in sorted(os.listdir(dir_path)):
        file_path = dir_path / f
        if file_path.is_file() and Path(f).suffix.lower() in supported_extensions:
            files.append(
                {
                    "name": f,
                    "extension": Path(f).suffix.lower(),
                    "size_bytes": file_path.stat().st_size,
                }
            )

    return {
        "directory": str(dir_path.absolute()),
        "supported_extensions": list(supported_extensions),
        "files": files,
        "total_files": len(files),
    }


# --- Summaries Management Endpoints ---


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


SUMMARIES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "summaries"
)


def get_summaries_dir() -> str:
    """Get the summaries directory path, creating it if needed."""
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    return SUMMARIES_DIR


@app.get("/summaries", response_model=SummaryListResponse)
async def list_summaries():
    """
    List all available summary versions.
    Returns the current version and all available versions sorted by date (newest first).
    """
    import glob

    summaries_dir = get_summaries_dir()

    # Find all community_summaries files
    pattern = os.path.join(summaries_dir, "community_summaries_*.json")
    summary_files = glob.glob(pattern)

    # Extract versions
    versions = []
    for f in summary_files:
        filename = os.path.basename(f)
        # Extract timestamp from filename: community_summaries_2026-04-06_143000.json
        parts = filename.replace("community_summaries_", "").replace(".json", "")

        # Get file stats
        stat = os.stat(f)
        from datetime import datetime

        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()

        versions.append(
            {
                "version": parts,
                "filename": filename,
                "modified": mtime,
                "size_bytes": stat.st_size,
            }
        )

    # Sort by version (newest first)
    versions.sort(key=lambda x: x["version"], reverse=True)

    # Get current version
    current_path = os.path.join(summaries_dir, "current.json")
    current = None
    if os.path.exists(current_path):
        with open(current_path, "r") as f:
            current_data = json.load(f)
            current = SummaryVersion(**current_data)

    return SummaryListResponse(current=current, versions=versions)


@app.get("/summaries/current")
async def get_current_summary():
    """
    Get the current (active) summary version info.
    """
    current_path = os.path.join(get_summaries_dir(), "current.json")

    if not os.path.exists(current_path):
        raise HTTPException(status_code=404, detail="No current summary version found")

    with open(current_path, "r") as f:
        return json.load(f)


@app.get("/summaries/{version}")
async def get_summary_version(version: str):
    """
    Get a specific summary version's content.
    Returns both community_summaries and entity_info for the specified version.
    """
    summaries_dir = get_summaries_dir()

    summary_file = os.path.join(summaries_dir, f"community_summaries_{version}.json")
    entity_file = os.path.join(summaries_dir, f"entity_info_{version}.json")

    if not os.path.exists(summary_file):
        raise HTTPException(
            status_code=404, detail=f"Summary version '{version}' not found"
        )

    result = {"version": version}

    with open(summary_file, "r") as f:
        result["community_summaries"] = json.load(f)

    if os.path.exists(entity_file):
        with open(entity_file, "r") as f:
            result["entity_info"] = json.load(f)

    return result


@app.delete("/summaries")
async def cleanup_summaries(
    keep: int = Query(5, description="Number of versions to keep")
):
    """
    Delete old summary versions, keeping the N most recent.
    Also updates current.json if the current version is deleted.
    """
    import glob

    if keep < 1:
        raise HTTPException(status_code=400, detail="Must keep at least 1 version")

    summaries_dir = get_summaries_dir()

    # Find all versions
    pattern = os.path.join(summaries_dir, "community_summaries_*.json")
    summary_files = glob.glob(pattern)

    # Extract versions and sort
    versions = []
    for f in summary_files:
        filename = os.path.basename(f)
        version = filename.replace("community_summaries_", "").replace(".json", "")
        versions.append((version, f))

    # Sort by version (newest first)
    versions.sort(key=lambda x: x[0], reverse=True)

    # Determine which to delete
    to_delete = versions[keep:]
    to_keep = versions[:keep]

    deleted = []
    for version, filepath in to_delete:
        # Delete both community_summaries and entity_info
        summary_file = filepath
        entity_file = filepath.replace("community_summaries_", "entity_info_")

        try:
            os.remove(summary_file)
            deleted.append(os.path.basename(summary_file))
        except Exception as e:
            logger.warning(f"Failed to delete {summary_file}: {e}")

        if os.path.exists(entity_file):
            try:
                os.remove(entity_file)
                deleted.append(os.path.basename(entity_file))
            except Exception as e:
                logger.warning(f"Failed to delete {entity_file}: {e}")

    # Check if current version was deleted
    current_path = os.path.join(summaries_dir, "current.json")
    if os.path.exists(current_path):
        with open(current_path, "r") as f:
            current_data = json.load(f)

        current_version = current_data.get("version", "")
        deleted_versions = [v for v, _ in to_delete]

        if current_version in deleted_versions:
            # Update current to newest remaining version
            if to_keep:
                newest_version = to_keep[0][0]
                from datetime import datetime

                new_current = {
                    "version": newest_version,
                    "created_at": datetime.now().isoformat(),
                    "files": {
                        "community_summaries": f"community_summaries_{newest_version}.json",
                        "entity_info": f"entity_info_{newest_version}.json",
                    },
                }
                # Load stats from the file if possible
                entity_file = os.path.join(
                    summaries_dir, f"entity_info_{newest_version}.json"
                )
                if os.path.exists(entity_file):
                    with open(entity_file, "r") as f:
                        entity_data = json.load(f)
                        new_current["stats"] = {
                            "total_entities": len(entity_data),
                            "total_communities": 0,  # Would need to load summary file too
                        }

                with open(current_path, "w") as f:
                    json.dump(new_current, f, indent=4)
            else:
                # No versions left, remove current.json
                os.remove(current_path)

    kept_files = [os.path.basename(f) for _, f in to_keep]

    return SummaryCleanupResponse(
        deleted=deleted,
        kept=kept_files,
        message=f"Deleted {len(deleted)//2} version(s), keeping {len(to_keep)} version(s)",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
