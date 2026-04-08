#!/usr/bin/env python3
"""
Mock FastAPI server for testing the CLI without the full backend.

Usage:
    python mock_api.py [--port PORT]

This simulates the GraphRAG API responses for CLI development/testing.
"""

import argparse
import json
import random
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
import uvicorn

app = FastAPI(title="Study Buddy Mock API")

# Sample community summaries with full text
COMMUNITY_SUMMARIES = {
    "10": """SMT solvers can operate on knowledge graphs to reason about the relationships between entities. This enables the solvers to analyze and draw conclusions about the entities and their relationships. SMT (Satisfiability Modulo Theories) solvers extend SAT solvers with the ability to reason about theories such as arrays, bit-vectors, and strings. When combined with knowledge graphs, they can verify properties and find counterexamples in complex systems.

Key concepts in this community:
- Constraint solving over graph structures
- Ontology reasoning and inference
- Integration with description logics
- Applications in verification and testing""",

    "12": """Knowledge graphs are crucial for various applications including data-driven approaches. They rely on specific structures given by database management systems and often utilize RDF (Resource Description Framework) for representation. Knowledge graphs enable semantic queries, inference, and reasoning over connected data.

This community covers:
- Graph databases and query languages (SPARQL, Cypher)
- Ontology engineering and knowledge representation
- Entity linking and resolution
- Graph neural networks and embeddings
- Integration with large language models for RAG systems""",

    "18": """The University of Oslo and Oslo are closely related, with the university being located in the city. This community contains entities related to academic institutions, research organizations, and geographic locations in Norway.

Entities include:
- University of Oslo (UiO) - Norway's oldest university
- Oslo - capital city and economic center
- Research groups and departments
- Academic collaborations and partnerships""",

    "19": """CISPA Helmholtz Center for Information Security is located in Saarbrücken, Germany. This community contains entities related to cybersecurity research, information security, and privacy.

Key areas:
- Security testing and fuzzing
- Formal verification
- Privacy-preserving technologies
- Secure systems design
- Cyber threat intelligence""",

    "20": """RDF and OWL are key technologies for knowledge representation on the semantic web. The OWL EL profile is optimized for expressive ontologies with efficient reasoning. This community covers standards and tools for semantic data.

Topics include:
- RDF (Resource Description Framework) data model
- OWL (Web Ontology Language) profiles and expressivity
- SPARQL query language
- Reasoning algorithms and complexity
- Tools like Protégé, Apache Jena, and RDF4J""",

    "21": """Grammar-based Fuzzing and Symbolic Execution are techniques used in software testing and verification. These approaches systematically explore program inputs to find bugs and security vulnerabilities.

This community covers:
- Grammar-based test generation
- Symbolic execution engines
- Constraint solving for path exploration
- Coverage-guided fuzzing
- Hybrid approaches combining static and dynamic analysis
- Tools like AFL, LibFuzzer, KLEE, and angr""",

    "22": """Software tools for testing and analysis encompass a wide range of utilities for ensuring software quality, correctness, and security.

Categories include:
- Static analysis tools (linters, type checkers)
- Dynamic analysis (profilers, sanitizers)
- Test frameworks and coverage tools
- Security scanners
- Continuous integration utilities""",
}

# Sample entity info (entity -> list of community IDs)
ENTITY_INFO = {
    "Knowledge Graphs": [12, 10],
    "SMT Solvers": [10, 10, 10],
    "Ontologies": [10, 20],
    "RDF": [20, 20, 20],
    "Fuzzing": [21, 2],
    "Grammar-based Fuzzing": [21],
    "Symbolic Execution": [21],
    "University of Oslo": [18],
    "Oslo": [18],
    "CISPA Helmholtz Center for Information Security": [19],
    "Saarbrucken": [19],
    "Testing": [2, 22],
    "Software Tools": [22],
    "Miller et al.": [2, 21],
    "Neo4j": [12],
    "LlamaIndex": [12],
    "GraphRAG": [12],
    "OWL Reasoners": [10, 20],
    "Compilers": [22],
}


class QueryRequest(BaseModel):
    query: str
    similarity_top_k: int = 20


class GraphQueryResponse(BaseModel):
    answer: str
    communities_consulted: List[Union[str, int]]
    entities_found: List[str]


# --- Entity Models ---
class EntitySearchResponse(BaseModel):
    entities: List[Dict[str, object]]
    total: int


class EntityDetail(BaseModel):
    name: str
    communities: List[int]


# --- Community Models ---
class CommunityListResponse(BaseModel):
    communities: List[Dict[str, object]]
    total: int


class CommunityDetail(BaseModel):
    id: int
    summary: str
    entity_count: int


class CommunityEntitiesResponse(BaseModel):
    community_id: int
    entities: List[str]
    total: int


@app.get("/")
async def root():
    return {"message": "Study Buddy GraphRAG API is online. Go to /docs for Swagger UI."}


# --- Entity Endpoints ---
@app.get("/entities", response_model=EntitySearchResponse)
async def search_entities(
    q: Optional[str] = Query(None, description="Search term for entity names"),
    limit: int = Query(50, ge=1, le=200, description="Maximum results to return")
):
    """Search for entities in the knowledge graph."""
    if q:
        q_lower = q.lower()
        matches = [
            {"name": name, "communities": list(set(communities))}
            for name, communities in ENTITY_INFO.items()
            if q_lower in name.lower()
        ]
        matches.sort(key=lambda x: (x["name"].lower() != q_lower, len(x["name"])))
    else:
        matches = [
            {"name": name, "communities": list(set(communities))}
            for name, communities in ENTITY_INFO.items()
        ]
        matches.sort(key=lambda x: x["name"].lower())
    
    return {"entities": matches[:limit], "total": len(matches)}


@app.get("/entities/{name}", response_model=EntityDetail)
async def get_entity(name: str):
    """Get details for a specific entity by name."""
    for entity_name, communities in ENTITY_INFO.items():
        if entity_name.lower() == name.lower():
            return {"name": entity_name, "communities": list(set(communities))}
    
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")


# --- Community Endpoints ---
@app.get("/communities", response_model=CommunityListResponse)
async def list_communities():
    """List all communities with entity counts."""
    # Build community list with entity counts
    community_entities: Dict[int, List[str]] = {}
    for entity_name, communities in ENTITY_INFO.items():
        for comm_id in communities:
            if comm_id not in community_entities:
                community_entities[comm_id] = []
            community_entities[comm_id].append(entity_name)
    
    communities = [
        {
            "id": int(comm_id),
            "entity_count": len(set(community_entities.get(int(comm_id), []))),
            "summary_preview": COMMUNITY_SUMMARIES.get(comm_id, "")[:100] + "..."
        }
        for comm_id in sorted(COMMUNITY_SUMMARIES.keys(), key=int)
    ]
    
    return {"communities": communities, "total": len(communities)}


@app.get("/communities/{id}", response_model=CommunityDetail)
async def get_community(id: int):
    """Get details for a specific community including its summary."""
    summary = COMMUNITY_SUMMARIES.get(str(id))
    if summary is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Community {id} not found")
    
    # Count entities in this community
    entity_count = sum(
        1 for communities in ENTITY_INFO.values()
        if id in communities
    )
    
    return {"id": id, "summary": summary, "entity_count": entity_count}


@app.get("/communities/{id}/entities", response_model=CommunityEntitiesResponse)
async def get_community_entities(id: int):
    """Get all entities belonging to a specific community."""
    if str(id) not in COMMUNITY_SUMMARIES:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Community {id} not found")
    
    entities = [
        name for name, communities in ENTITY_INFO.items()
        if id in communities
    ]
    
    return {"community_id": id, "entities": sorted(set(entities)), "total": len(entities)}


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

The query "{request.query}" relates to several concepts in the graph. Communities consulted provide context about:
- Entity relationships and their significance
- Document sources for citation [Various sources]

This is a mock response for testing purposes."""
        entities = random.sample(list(ENTITY_INFO.keys()), min(5, request.similarity_top_k))
        communities = random.sample([int(k) for k in COMMUNITY_SUMMARIES.keys()], 3)
    
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