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


class WorkspaceCreate(BaseModel):
    """Request to create a new workspace."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    slug: Optional[str] = Field(None, pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")

    def get_slug(self) -> str:
        """Return the slug, auto-generating from name if not provided."""
        return self.slug if self.slug else slugify(self.name)


class WorkspaceInfo(BaseModel):
    """Response with workspace details."""
    id: str
    name: str
    description: str
    neo4j_database: str
    created_at: str
    updated_at: str
    entity_count: int = 0
    community_count: int = 0


class WorkspaceListResponse(BaseModel):
    """Response listing all workspaces."""
    workspaces: List[WorkspaceInfo]
    total: int


class WorkspaceStatsResponse(BaseModel):
    """Response with workspace statistics."""
    id: str
    name: str
    entity_count: int
    relationship_count: int
    community_count: int
    document_count: int = 0
    last_ingestion: Optional[str] = None


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