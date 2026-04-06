import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.base.response.schema import Response
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from pydantic import BaseModel, Field

# Import custom classes from local directory
from core_classes import GraphQueryResponse, GraphRAGQueryEngine, GraphRAGStore

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
