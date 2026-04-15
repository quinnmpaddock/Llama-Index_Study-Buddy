"""Tests for service modules extracted from app.py."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src/ is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from services.community import CommunityService
from services.graph import GraphService, _make_summary_preview
from services.query import QueryService
from services.ingestion import extract_json, parse_fn, IngestionService


# ---------------------------------------------------------------------------
# CommunityService tests
# ---------------------------------------------------------------------------


class TestCommunityServiceFindSnapshot:
    """Tests for CommunityService.find_most_recent_snapshot."""

    def test_no_snapshots(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        result = svc.find_most_recent_snapshot(workspace_id="test-ws")
        assert result is None

    def test_single_snapshot(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        ws_dir = tmp_path / "test-ws" / "summaries"
        ws_dir.mkdir(parents=True)

        # Write a matching pair
        with open(ws_dir / "community_summaries_2026-04-10_120000.json", "w") as f:
            json.dump({}, f)
        with open(ws_dir / "entity_info_2026-04-10_120000.json", "w") as f:
            json.dump({}, f)

        version = svc.find_most_recent_snapshot(workspace_id="test-ws")
        assert version == "2026-04-10_120000"

    def test_multiple_snapshots_returns_newest(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        ws_dir = tmp_path / "ws1" / "summaries"
        ws_dir.mkdir(parents=True)

        for ts in ["2026-01-01_000000", "2026-06-15_120000", "2026-03-01_060000"]:
            with open(ws_dir / f"community_summaries_{ts}.json", "w") as f:
                json.dump({}, f)
            with open(ws_dir / f"entity_info_{ts}.json", "w") as f:
                json.dump({}, f)

        version = svc.find_most_recent_snapshot(workspace_id="ws1")
        assert version == "2026-06-15_120000"

    def test_incomplete_pair_ignored(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        ws_dir = tmp_path / "ws2" / "summaries"
        ws_dir.mkdir(parents=True)

        # Only community_summaries, no entity_info
        with open(ws_dir / "community_summaries_2026-01-01_000000.json", "w") as f:
            json.dump({}, f)

        version = svc.find_most_recent_snapshot(workspace_id="ws2")
        assert version is None


class TestCommunityServiceLoadSave:
    """Tests for CommunityService.load/save round-trip."""

    def test_save_and_load(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        summaries = {"1": "A summary", "2": "Another summary"}
        entity_info = {"entity1": [1, 2], "entity2": [2]}

        svc.save_summaries("ws-test", summaries, entity_info, version="v1")
        loaded_summaries, loaded_entities = svc.load_summaries_and_entity_info("ws-test")

        assert loaded_summaries == {1: "A summary", 2: "Another summary"}
        assert loaded_entities == entity_info

    def test_load_empty_returns_empty(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        summaries, entities = svc.load_summaries_and_entity_info("empty-ws")
        assert summaries == {}
        assert entities == {}

    def test_load_via_current_json_pointer(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        summaries = {"1": "Only version"}
        entity_info = {"e1": [1]}

        svc.save_summaries("pointer-ws", summaries, entity_info, version="v_pin")

        # Load in a new instance — should find data via current.json
        svc2 = CommunityService(data_dir=str(tmp_path))
        loaded_s, loaded_e = svc2.load_summaries_and_entity_info("pointer-ws")
        assert loaded_s == {1: "Only version"}


class TestCommunityServiceVersions:
    """Tests for list/get/cleanup versions."""

    def test_list_versions_empty(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        current, versions = svc.list_versions("ws-list")
        assert current is None
        assert versions == []

    def test_list_versions_with_data(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        svc.save_summaries("ws-list", {"1": "s1"}, {"e1": [1]}, version="v_list")

        current, versions = svc.list_versions("ws-list")
        assert current is not None
        assert current["version"] == "v_list"
        assert len(versions) == 1

    def test_get_version(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        svc.save_summaries("ws-get", {"1": "summary"}, {"e1": [1]}, version="v_get")

        data = svc.get_version("v_get", "ws-get")
        assert data is not None
        assert data["version"] == "v_get"

    def test_get_version_not_found(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        assert svc.get_version("nonexistent", "ws-get") is None

    def test_cleanup_keeps_newest(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        # Create 3 versions
        for v in ["v_old", "v_mid", "v_new"]:
            svc.save_summaries("ws-cleanup", {"1": f"summary_{v}"}, {"e1": [1]}, version=v)

        deleted, kept = svc.cleanup_versions("ws-cleanup", keep=2)
        assert len(deleted) >= 2  # both entity_info and community_summaries files deleted

    def test_cleanup_invalid_keep(self, tmp_path):
        svc = CommunityService(data_dir=str(tmp_path))
        with pytest.raises(ValueError, match="at least 1"):
            svc.cleanup_versions("ws-bad", keep=0)


# ---------------------------------------------------------------------------
# GraphService tests
# ---------------------------------------------------------------------------


class TestMakeSummaryPreview:
    """Tests for the _make_summary_preview helper."""

    def test_short_summary(self):
        assert _make_summary_preview("Hello world") == "Hello world"

    def test_long_summary_truncated(self):
        text = "A" * 200
        result = _make_summary_preview(text)
        assert len(result) <= 104  # 100 + "..."

    def test_strips_intro_sentence(self):
        text = "This community is about. The real content is here."
        result = _make_summary_preview(text)
        assert result.startswith("The real content")


class TestGraphServiceEntities:
    """Tests for GraphService entity search."""

    def _make_state(self):
        state = MagicMock()
        state.entity_info = {"Apple": [1, 2], "Banana": [3], "Cherry": [1, 3]}
        state.community_summaries = {"1": "Community 1", "2": "Community 2", "3": "Community 3"}
        return state

    def test_search_all(self):
        svc = GraphService(state=self._make_state())
        result = svc.search_entities(query=None)
        assert result["total"] == 3
        assert len(result["entities"]) == 3

    def test_search_with_query(self):
        svc = GraphService(state=self._make_state())
        result = svc.search_entities(query="app")
        assert result["total"] == 1
        assert result["entities"][0]["name"] == "Apple"

    def test_get_entity_found(self):
        svc = GraphService(state=self._make_state())
        result = svc.get_entity("apple")
        assert result is not None
        assert result["name"] == "Apple"

    def test_get_entity_not_found(self):
        svc = GraphService(state=self._make_state())
        result = svc.get_entity("durian")
        assert result is None

    def test_no_state(self):
        svc = GraphService(state=None)
        result = svc.search_entities(query=None)
        assert result["total"] == 0


class TestGraphServiceCommunities:
    """Tests for GraphService community endpoints."""

    def _make_state(self):
        state = MagicMock()
        state.entity_info = {"Alpha": [1], "Beta": [1, 2], "Gamma": [2]}
        state.community_summaries = {
            "1": "Community about Alpha and Beta.",
            "2": "Community about Beta and Gamma.",
        }
        return state

    def test_list_communities(self):
        svc = GraphService(state=self._make_state())
        result = svc.list_communities()
        assert result["total"] == 2
        assert result["communities"][0]["id"] == 1

    def test_get_community(self):
        svc = GraphService(state=self._make_state())
        result = svc.get_community(id=1)
        assert result is not None
        assert result["summary"] == "Community about Alpha and Beta."
        assert result["entity_count"] == 2

    def test_get_community_not_found(self):
        svc = GraphService(state=self._make_state())
        result = svc.get_community(id=999)
        assert result is None

    def test_get_community_entities(self):
        svc = GraphService(state=self._make_state())
        result = svc.get_community_entities(id=1)
        assert result is not None
        assert "Alpha" in result["entities"]
        assert "Beta" in result["entities"]


# ---------------------------------------------------------------------------
# QueryService tests
# ---------------------------------------------------------------------------


class TestQueryService:
    def test_query_no_engine(self):
        svc = QueryService(state=None)
        with pytest.raises(RuntimeError, match="Engine not initialized"):
            svc.query("test")

    def test_engine_not_loaded(self):
        state = MagicMock()
        del state.engine  # will cause hasattr to return False
        svc = QueryService(state=state)
        with pytest.raises(RuntimeError, match="Engine not initialized"):
            svc.query("test")

    def test_format_response(self):
        response = MagicMock()
        response.response = "The answer is 42."
        response.metadata = {"communities_consulted": [1, 2], "entities_found": ["A", "B"]}
        result = QueryService.format_response(response)
        assert result["answer"] == "The answer is 42."
        assert result["communities_consulted"] == [1, 2]


# ---------------------------------------------------------------------------
# Ingestion helpers tests
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_valid_json(self):
        result = extract_json('Some text {"key": "value"} more text')
        assert result == {"key": "value"}

    def test_no_json(self):
        result = extract_json("No JSON here")
        assert result is None

    def test_nested_json(self):
        result = extract_json('{"entities": [], "relationships": []}')
        assert result == {"entities": [], "relationships": []}


class TestParseFn:
    def test_valid_response(self):
        response = json.dumps({
            "entities": [{"entity_name": "A", "entity_type": "B", "entity_description": "C"}],
            "relationships": [{"source_entity": "A", "target_entity": "B", "relation": "R", "relationship_description": "D"}],
        })
        entities, relationships = parse_fn(response)
        assert len(entities) == 1
        assert len(relationships) == 1

    def test_invalid_response(self):
        entities, relationships = parse_fn("not valid json at all")
        assert entities == []
        assert relationships == []


class TestIngestionServicePreview:
    def test_preview_nonexistent_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            IngestionService.preview_directory(str(tmp_path / "nonexistent"))

    def test_preview_not_directory(self, tmp_path):
        file_path = tmp_path / "file.txt"
        file_path.write_text("hello")
        with pytest.raises(ValueError):
            IngestionService.preview_directory(str(file_path))

    def test_preview_directory(self, tmp_path):
        # Create some files
        (tmp_path / "test.md").write_text("# Hello")
        (tmp_path / "test.txt").write_text("World")
        (tmp_path / "test.xyz").write_text("Unsupported")

        result = IngestionService.preview_directory(str(tmp_path))
        assert result["total_files"] == 2  # .md and .txt, not .xyz
        assert any(f["name"] == "test.md" for f in result["files"])


class TestPathResolution:
    """Verify that __file__-based paths resolve correctly from services/ subdir.

    The services/ module is two levels deep (src/services/), so any path
    built from __file__ must account for this extra nesting. These tests
    guard against regressions like the kg_extract_template.txt FileNotFoundError
    where both path attempts resolved to wrong directories.
    """

    def test_prompts_dir_resolves_from_services(self):
        """src/prompts/kg_extract_template.txt must be reachable from services/."""
        # Same logic as ingestion.py: Path(__file__).resolve().parent.parent
        from services.ingestion import __file__ as ingestion_file
        _src_dir = Path(ingestion_file).resolve().parent.parent
        template_path = _src_dir / "prompts" / "kg_extract_template.txt"
        assert template_path.exists(), (
            f"Template not found at {template_path}. "
            "Path resolution from services/ must reach src/prompts/."
        )

    def test_project_root_resolves_from_services(self):
        """Project root must be reachable from services/ (3 levels up)."""
        from services.community import _PROJECT_ROOT
        assert _PROJECT_ROOT.exists(), f"Project root {_PROJECT_ROOT} does not exist"
        # The project root should contain src/ and summaries/
        assert (_PROJECT_ROOT / "src").is_dir(), f"src/ not found under {_PROJECT_ROOT}"
        assert (_PROJECT_ROOT / "summaries").is_dir() or (_PROJECT_ROOT / "app_data").is_dir(), (
            f"Neither summaries/ nor app_data/ found under {_PROJECT_ROOT}"
        )


class TestIngestionServiceStart:
    def test_start_ingestion_no_directory(self):
        config = MagicMock()
        config.llm.api_key = "test-key"
        svc = IngestionService(config=config)
        with pytest.raises(FileNotFoundError):
            svc.start_ingestion("/nonexistent/path")

    def test_get_status_unknown_task(self):
        config = MagicMock()
        svc = IngestionService(config=config)
        assert svc.get_status("nonexistent-task-id") is None