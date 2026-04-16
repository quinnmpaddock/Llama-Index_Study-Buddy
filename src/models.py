"""Shared Pydantic models for Study Buddy API and services."""
import re
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


def slugify(name: str) -> str:
    """Convert a human-readable name to a URL-safe slug.
    
    Examples:
        "ML Research" -> "ml-research"
        "Biology Notes!" -> "biology-notes"
        "My  Cool   Project" -> "my-cool-project"
    """
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


# ---- Existing API models (will move from app.py later) ----

class QueryRequest(BaseModel):
    """Request to query the knowledge graph."""
    query: str = Field(..., description="The query to ask the knowledge graph")
    similarity_top_k: int = Field(default=20, ge=1, le=50)


class GraphQueryResponse(BaseModel):
    """Structured response from graph queries."""
    answer: str
    communities_consulted: List[int | str]
    entities_found: List[str]