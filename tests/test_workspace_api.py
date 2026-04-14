"""Tests for the workspace-aware API endpoints (/kb prefix)."""
import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from contextlib import asynccontextmanager

# Ensure src/ is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Set env vars before importing app (needed by config)
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-api-tests")


@pytest.fixture
def client():
    """Create a test client with mocked lifespan that skips Neo4j/LLM init."""
    from workspace import WorkspaceRegistry
    from services.community import CommunityService
    from services.graph import GraphService
    from services.query import QueryService
    from services.ingestion import IngestionService

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = WorkspaceRegistry(data_dir=Path(tmpdir))
        registry.create(name="Default", description="Default workspace (auto-created)")

        # Create a no-op lifespan that sets up app.state with mocks
        @asynccontextmanager
        async def mock_lifespan(app):
            app.state.workspace_registry = registry
            app.state.community_svc = MagicMock(spec=CommunityService)
            app.state.graph_svc = MagicMock(spec=GraphService)
            app.state.query_svc = MagicMock(spec=QueryService)
            app.state.ingestion_svc = MagicMock(spec=IngestionService)
            app.state.engine = MagicMock()
            app.state.summaries_loaded = True
            app.state.community_summaries = {}
            app.state.entity_info = {}

            # Mock graph_svc responses for entity/community endpoints
            app.state.graph_svc.search_entities.return_value = {
                "entities": [],
                "total": 0,
            }
            app.state.graph_svc.get_entity.return_value = None
            app.state.graph_svc.list_communities.return_value = {
                "communities": [],
                "total": 0,
            }
            app.state.graph_svc.get_community.return_value = None
            app.state.graph_svc.get_community_entities.return_value = None

            # Mock community_svc for summary endpoints
            app.state.community_svc.list_versions.return_value = (None, [])
            app.state.community_svc.get_current_version.return_value = None
            app.state.community_svc.get_version.return_value = None

            yield

        # Patch the app's lifespan before creating TestClient
        import app as app_module
        original_lifespan = app_module.app.router.lifespan_context
        app_module.app.router.lifespan_context = mock_lifespan

        from fastapi.testclient import TestClient
        try:
            with TestClient(app_module.app, raise_server_exceptions=False) as c:
                yield c
        finally:
            # Restore original lifespan
            app_module.app.router.lifespan_context = original_lifespan


# ------------------------------------------------------------------
# Workspace CRUD endpoints
# ------------------------------------------------------------------

def test_create_workspace(client):
    """POST /kb creates a new workspace."""
    resp = client.post("/kb", json={"name": "Biology", "description": "Bio notes"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == "biology"
    assert data["name"] == "Biology"
    assert data["description"] == "Bio notes"
    assert data["neo4j_database"] == "sb_biology"


def test_create_workspace_duplicate_returns_409(client):
    """POST /kb with duplicate name returns 409."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.post("/kb", json={"name": "Biology"})
    assert resp.status_code == 409


def test_list_workspaces(client):
    """GET /kb lists workspaces including the auto-created default."""
    resp = client.get("/kb")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1  # at least the default workspace
    ids = [ws["id"] for ws in data["workspaces"]]
    assert "default" in ids


def test_get_workspace(client):
    """GET /kb/{id} returns workspace info."""
    # The default workspace should exist
    resp = client.get("/kb/default")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "default"
    assert data["name"] == "Default"


def test_get_workspace_not_found(client):
    """GET /kb/{id} returns 404 for nonexistent workspace."""
    resp = client.get("/kb/nonexistent-workspace")
    assert resp.status_code == 404


def test_delete_workspace(client):
    """DELETE /kb/{id} deletes a non-default workspace."""
    client.post("/kb", json={"name": "To Delete"})
    resp = client.delete("/kb/to-delete")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Verify it's gone
    resp = client.get("/kb/to-delete")
    assert resp.status_code == 404


def test_delete_default_workspace_returns_403(client):
    """DELETE /kb/default returns 403."""
    resp = client.delete("/kb/default")
    assert resp.status_code == 403


def test_delete_nonexistent_workspace_returns_404(client):
    """DELETE /kb/{id} returns 404 for nonexistent workspace."""
    resp = client.delete("/kb/nonexistent-workspace")
    assert resp.status_code == 404


# ------------------------------------------------------------------
# Workspace-scoped endpoints — 501 for non-default workspaces
# ------------------------------------------------------------------

def test_query_non_default_workspace_returns_501(client):
    """POST /kb/{workspace_id}/query returns 501 for non-default workspace."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.post("/kb/biology/query", json={"query": "test"})
    assert resp.status_code == 501


def test_entities_non_default_workspace_returns_501(client):
    """GET /kb/{workspace_id}/entities returns 501 for non-default workspace."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.get("/kb/biology/entities")
    assert resp.status_code == 501


def test_entity_detail_non_default_workspace_returns_501(client):
    """GET /kb/{workspace_id}/entities/{name} returns 501 for non-default."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.get("/kb/biology/entities/test-entity")
    assert resp.status_code == 501


def test_communities_non_default_workspace_returns_501(client):
    """GET /kb/{workspace_id}/communities returns 501 for non-default."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.get("/kb/biology/communities")
    assert resp.status_code == 501


def test_community_detail_non_default_workspace_returns_501(client):
    """GET /kb/{workspace_id}/communities/{id} returns 501 for non-default."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.get("/kb/biology/communities/1")
    assert resp.status_code == 501


def test_community_entities_non_default_workspace_returns_501(client):
    """GET /kb/{workspace_id}/communities/{id}/entities returns 501."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.get("/kb/biology/communities/1/entities")
    assert resp.status_code == 501


def test_ingest_non_default_workspace_returns_501(client):
    """POST /kb/{workspace_id}/ingest returns 501 for non-default workspace."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.post("/kb/biology/ingest", json={"directory": "/tmp"})
    assert resp.status_code == 501


def test_ingest_status_non_default_workspace_returns_501(client):
    """GET /kb/{workspace_id}/ingest/status/{task_id} returns 501."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.get("/kb/biology/ingest/status/some-task")
    assert resp.status_code == 501


def test_ingest_preview_non_default_workspace_returns_501(client):
    """GET /kb/{workspace_id}/ingest/preview returns 501."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.get("/kb/biology/ingest/preview?directory=/tmp")
    assert resp.status_code == 501


def test_summaries_non_default_workspace_returns_501(client):
    """GET /kb/{workspace_id}/summaries returns 501 for non-default."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.get("/kb/biology/summaries")
    assert resp.status_code == 501


def test_summaries_current_non_default_workspace_returns_501(client):
    """GET /kb/{workspace_id}/summaries/current returns 501."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.get("/kb/biology/summaries/current")
    assert resp.status_code == 501


def test_summaries_version_non_default_workspace_returns_501(client):
    """GET /kb/{workspace_id}/summaries/{version} returns 501."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.get("/kb/biology/summaries/v1")
    assert resp.status_code == 501


def test_summaries_cleanup_non_default_workspace_returns_501(client):
    """DELETE /kb/{workspace_id}/summaries returns 501."""
    client.post("/kb", json={"name": "Biology"})
    resp = client.delete("/kb/biology/summaries")
    assert resp.status_code == 501


# ------------------------------------------------------------------
# Workspace-scoped endpoints — 404 for nonexistent workspace
# ------------------------------------------------------------------

def test_query_nonexistent_workspace_returns_404(client):
    """POST /kb/{workspace_id}/query returns 404 for unknown workspace."""
    resp = client.post("/kb/does-not-exist/query", json={"query": "test"})
    assert resp.status_code == 404


def test_entities_nonexistent_workspace_returns_404(client):
    """GET /kb/{workspace_id}/entities returns 404 for unknown workspace."""
    resp = client.get("/kb/does-not-exist/entities")
    assert resp.status_code == 404


def test_communities_nonexistent_workspace_returns_404(client):
    """GET /kb/{workspace_id}/communities returns 404 for unknown workspace."""
    resp = client.get("/kb/does-not-exist/communities")
    assert resp.status_code == 404


# ------------------------------------------------------------------
# Legacy endpoints still reachable (don't 404)
# ------------------------------------------------------------------

def test_legacy_query_endpoint_reachable(client):
    """POST /query should still be reachable (not removed)."""
    # It will fail with 503 because mocked engine can't really query,
    # but it should NOT return 404
    resp = client.post("/query", json={"query": "test"})
    assert resp.status_code != 404


def test_legacy_entities_endpoint_reachable(client):
    """GET /entities should still be reachable."""
    resp = client.get("/entities")
    assert resp.status_code != 404


def test_legacy_communities_endpoint_reachable(client):
    """GET /communities should still be reachable."""
    resp = client.get("/communities")
    assert resp.status_code != 404


def test_root_still_works(client):
    """GET / should still return the root message."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()