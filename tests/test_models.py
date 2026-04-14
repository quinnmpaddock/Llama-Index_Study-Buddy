"""Tests for Pydantic models."""
from src.models import WorkspaceCreate, WorkspaceInfo, slugify, QueryRequest, GraphQueryResponse


def test_slugify_basic():
    assert slugify("ML Research") == "ml-research"


def test_slugify_special_chars():
    assert slugify("Biology Notes!") == "biology-notes"


def test_slugify_multiple_spaces():
    assert slugify("My  Cool   Project") == "my-cool-project"


def test_slugify_leading_trailing_spaces():
    assert slugify("  hello world  ") == "hello-world"


def test_slugify_unicode():
    result = slugify("Café Résumé")
    assert "cafe" in result or "caf" in result  # unicode normalization varies


def test_workspace_create_auto_slug():
    req = WorkspaceCreate(name="ML Research", description="My ML knowledge base")
    assert req.get_slug() == "ml-research"


def test_workspace_create_custom_slug():
    req = WorkspaceCreate(name="ML Research", slug="custom-slug", description="test")
    assert req.get_slug() == "custom-slug"


def test_workspace_create_default_description():
    req = WorkspaceCreate(name="Test")
    assert req.description == ""


def test_workspace_info_model():
    info = WorkspaceInfo(
        id="ml-research",
        name="ML Research",
        description="My ML knowledge base",
        neo4j_database="sb_ml_research",
        created_at="2026-04-13T00:00:00",
        updated_at="2026-04-13T00:00:00",
    )
    assert info.id == "ml-research"
    assert info.entity_count == 0  # default


def test_query_request_defaults():
    req = QueryRequest(query="What is AI?")
    assert req.similarity_top_k == 20