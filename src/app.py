import json
import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.base.response.schema import Response
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from pydantic import BaseModel, Field

# Import custom classes from local directory
from core_classes import GraphQueryResponse, GraphRAGQueryEngine, GraphRAGStore

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants (mirroring main.py)
NEO4JPASSWORD = "Drewert237?"
NEO4J_URL = "bolt://localhost:7687"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARIES_PATH = os.path.join(BASE_DIR, "..", "summaries", "community_summaries.json")


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
        Settings.llm = Groq(model="meta-llama/llama-4-scout-17b-16e-instruct")

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

        # 3. Initialize Store and Index
        # We pass the loaded summaries directly to the store
        graph_store = GraphRAGStore(
            username="neo4j",
            password=NEO4JPASSWORD,
            url=NEO4J_URL,
            community_summary=community_summaries,
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
            "communities_detected": response.metadata.get("communities_consulted", []),
            "entities_found": response.metadata.get("entities_found", []),
        }
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
