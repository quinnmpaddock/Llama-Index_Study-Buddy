"""Tests for legacy API endpoints (no /kb prefix)."""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock
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
    from services.community import CommunityService
    from services.graph import GraphService
    from services.query import QueryService
    from services.ingestion import IngestionService

    # Create a no-op lifespan that sets up app.state with mocks
    @asynccontextmanager
    async def mock_lifespan(app):
        app.state.community_svc = MagicMock(spec=CommunityService)
        app.state.graph_svc = MagicMock(spec=GraphService)
        app.state.query_svc = MagicMock(spec=QueryService)
        app.state.ingestion_svc = MagicMock(spec=IngestionService)
        mock_response = MagicMock()
        mock_response.response = "test answer"
        mock_response.metadata = {"communities_consulted": [], "entities_found": []}
        app.state.engine = AsyncMock()
        app.state.engine.acustom_query.return_value = mock_response
        app.state.summaries_loaded = True
        app.state.community_summaries = {}
        app.state.entity_info = {}
        app.state.data_dir = "/tmp/test-data"

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
# Legacy endpoints still reachable (don't 404)
# ------------------------------------------------------------------

def test_legacy_query_endpoint_reachable(client):
    """POST /query should still be reachable (not removed)."""
    resp = client.post("/query", json={"query": "test"})
    assert resp.status_code == 200


def test_legacy_entities_endpoint_reachable(client):
    """GET /entities should still be reachable."""
    resp = client.get("/entities")
    assert resp.status_code == 200


def test_legacy_communities_endpoint_reachable(client):
    """GET /communities should still be reachable."""
    resp = client.get("/communities")
    assert resp.status_code == 200


def test_root_still_works(client):
    """GET / should still return the root message."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


# ------------------------------------------------------------------
# Removed workspace-scoped routes return 404
# ------------------------------------------------------------------

def test_kb_workspace_routes_removed(client):
    """Workspace-scoped /kb routes should return 404 (removed)."""
    # POST /kb/default/query should not exist
    resp = client.post("/kb/default/query", json={"query": "test"})
    assert resp.status_code == 404

    # GET /kb/default/entities should not exist
    resp = client.get("/kb/default/entities")
    assert resp.status_code == 404

    # GET /kb/default/communities should not exist
    resp = client.get("/kb/default/communities")
    assert resp.status_code == 404