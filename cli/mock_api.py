#!/usr/bin/env python3
"""
Mock FastAPI server for testing the CLI without thefull backend.

Usage:
    python mock_api.py [--port PORT]

This simulates the GraphRAG API responses for CLI development/testing.
"""

import argparse
import json
import random
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
import uvicorn

app = FastAPI(title="Study Buddy Mock API")

# Sample community summaries (from your actual data)
COMMUNITY_SUMMARIES = {
    12: "Knowledge graphs are crucial for various applications including data-driven approaches...",    18: "The University of Oslo and Oslo are closely related, with the university being located in the city...",
    2: "Fuzzing is a technique used to increase software quality...",
}

ENTITIES = [
    "Knowledge Graphs", "Fuzzing", "SMT Solvers", "Testing",
    "Neo4j", "LlamaIndex", "GraphRAG", "Miller et al.",
    "OWL Reasoners", "Compilers", "Compiler Testing",
]


class QueryRequest(BaseModel):
    query: str
    similarity_top_k: int = 20


class GraphQueryResponse(BaseModel):
    answer: str
    communities_consulted: List[Union[str, int]]
    entities_found: List[str]


@app.get("/")
async def root():
    return {"message": "Study Buddy GraphRAG API is online. Go to /docs for Swagger UI."}


@app.post("/query", response_model=GraphQueryResponse)
async def query_graph(request: QueryRequest):
    """Mock query endpoint that returns plausible responses."""
    # Generate a mock answer based on the query
    query_lower = request.query.lower()
    
    if "knowledge graph" in query_lower or "graph" in query_lower:
        answer = """Based on the knowledge graph analysis:

Knowledge graphs (KGs) are a crucial component of various applications. Their generation, interpretation, and reliability are essential aspects that need to be addressed.

Key points:
- Data-driven approaches utilize tabular database data to produce these graphs [test_data0.pdf]
- Knowledge graphs rely on the specific structure given by database management systems [test_data0.pdf]
- KGs often utilize RDF (Resource Description Framework) for representation [test_data0.pdf]"""
        entities = ["Knowledge Graphs", "RDF", "Database Management Systems", "Data-driven Approaches"]
        communities = [12, 20,10]
    elif "fuzz" in query_lower or "test" in query_lower:
        answer = """Fuzzing is a technique used to increase the quality of software:

Key findings:- Miller et al. developed one of the first fuzzers [test_data0.pdf]
- Fuzzing suffers from "fuzz blockers" - issues that prevent fuzzing from progressing [test_data0.pdf]
- Fuzzing has evolved into more sophisticated techniques like symbolic execution-based white box fuzzers [test_data0.pdf]
- Fuzzers are used to test complex tools such as compilers and reasoners [test_data0.pdf]"""
        entities = ["Fuzzing", "Miller et al.", "Fuzz Blockers", "Testing", "SMT Solvers"]
        communities = [2, 4, 25]
    else:
        # Generic response
        answer = f"""Based on the knowledge graph analysis:

The query "{request.query}" relates to several concepts in the graph.Communities consulted provide context about:
- Entity relationships and their significance
- Document sources for citation[Various sources]

This is a mock response for testing purposes."""
        entities = random.sample(ENTITIES, min(5, request.similarity_top_k))
        communities = random.sample(list(COMMUNITY_SUMMARIES.keys()), 3)
    
    return GraphQueryResponse(
        answer=answer,
        communities_consulted=communities,
        entities_found=entities,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock API server for CLI testing")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()
    print(f"Starting mock API server on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    uvicorn.run(app, host=args.host, port=args.port)