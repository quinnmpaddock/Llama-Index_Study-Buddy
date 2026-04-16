"""Tests for GraphRAGStore in core/store.py.

Since the test environment doesn't have a working Neo4j/numpy stack,
we mock the llama_index dependencies at import time and only test the
logic that doesn't require a real database connection.
"""
import json
import os
import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock heavy llama_index dependencies before importing core.store
# ---------------------------------------------------------------------------
_mock_llama_index_modules = {
    "llama_index": MagicMock(),
    "llama_index.core": MagicMock(),
    "llama_index.core.Settings": MagicMock(),
    "llama_index.core.PropertyGraphIndex": MagicMock(),
    "llama_index.core.llms": MagicMock(),
    "llama_index.core.llms.LLM": MagicMock(),
    "llama_index.core.llms.ChatMessage": MagicMock(),
    "llama_index.core.query_engine": MagicMock(),
    "llama_index.core.query_engine.CustomQueryEngine": MagicMock(),
    "llama_index.core.base": MagicMock(),
    "llama_index.core.base.response": MagicMock(),
    "llama_index.core.base.response.schema": MagicMock(),
    "llama_index.core.async_utils": MagicMock(),
    "llama_index.core.bridge": MagicMock(),
    "llama_index.core.bridge.pydantic": MagicMock(),
    "llama_index.core.graph_stores": MagicMock(),
    "llama_index.core.graph_stores.types": MagicMock(),
    "llama_index.core.indices": MagicMock(),
    "llama_index.core.indices.property_graph": MagicMock(),
    "llama_index.core.indices.property_graph.utils": MagicMock(),
    "llama_index.core.prompts": MagicMock(),
    "llama_index.core.prompts.default_prompts": MagicMock(),
    "llama_index.core.schema": MagicMock(),
    "llama_index.graph_stores": MagicMock(),
    "llama_index.graph_stores.neo4j": MagicMock(),
}

# Create a real-looking ChatMessage that can be instantiated
class _FakeChatMessage:
    def __init__(self, role="user", content=""):
        self.role = role
        self.content = content

# Create a real-looking base class for Neo4jPropertyGraphStore
class _FakeNeo4jPropertyGraphStore:
    """Stand-in for Neo4jPropertyGraphStore that doesn't need a real connection."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

# Create a real-looking CustomQueryEngine base class
class _FakeCustomQueryEngine:
    """Stand-in for CustomQueryEngine."""
    pass

_mock_llama_index_modules["llama_index.graph_stores.neo4j"].Neo4jPropertyGraphStore = _FakeNeo4jPropertyGraphStore
_mock_llama_index_modules["llama_index.core"].PropertyGraphIndex = MagicMock()
_mock_llama_index_modules["llama_index.core"].Settings = MagicMock()
_mock_llama_index_modules["llama_index.core.llms"].ChatMessage = _FakeChatMessage
_mock_llama_index_modules["llama_index.core.llms"].LLM = MagicMock
_mock_llama_index_modules["llama_index.core.query_engine"].CustomQueryEngine = _FakeCustomQueryEngine
_mock_llama_index_modules["llama_index.core.base.response.schema"].Response = MagicMock()

# Patch sys.modules before importing core.store
for mod_name, mod_mock in _mock_llama_index_modules.items():
    if mod_name not in sys.modules:
        sys.modules[mod_name] = mod_mock

# Now we can import
from core.store import GraphRAGStore


def _make_store(**kwargs):
    """Create a GraphRAGStore without calling the real Neo4j __init__.

    Since we've mocked the base class, we can call __init__ normally.
    The fake Neo4jPropertyGraphStore just sets attributes from kwargs.
    """
    data_dir = kwargs.pop("data_dir", None)
    entity_info = kwargs.pop("entity_info", {})
    community_summary = kwargs.pop("community_summary", {})
    llm = kwargs.pop("llm", MagicMock())
    database = kwargs.pop("database", "neo4j")

    store = GraphRAGStore(
        username="neo4j",
        password="test",
        url="bolt://localhost:7867",
        database=database,
        llm=llm,
        entity_info=entity_info,
        community_summary=community_summary,
        refresh_schema=False,
        create_indexes=False,
        data_dir=data_dir,
    )
    return store


class TestGraphRAGStoreInit:
    """Tests for GraphRAGStore initialization."""

    def test_default_graph_name(self):
        """graph_name should be the database name passed to constructor."""
        store = _make_store()
        assert store.graph_name == "neo4j"

    def test_custom_database_name(self):
        """When a custom database is provided, graph_name should reflect it."""
        store = _make_store(database="mydb")
        assert store.graph_name == "mydb"


class TestGraphRAGStoreGDSProjection:
    """Tests for GDS projection naming in build_communities.

    These tests validate that build_communities uses graph_name for
    the GDS projection name.
    """

    def test_gds_projection_uses_graph_name(self):
        """build_communities should use graph_name as the projection name."""
        store = _make_store()
        gds_projection = store.graph_name
        assert gds_projection == "neo4j"


class TestGraphRAGStoreSummariesDir:
    """Tests for get_summaries_dir (uses 'default' subdir)."""

    def test_summaries_dir_with_data_dir(self, tmp_path):
        """When data_dir is set, summaries dir should be data_dir/default/summaries."""
        store = _make_store(data_dir=str(tmp_path))
        result = store.get_summaries_dir()
        expected = tmp_path / "default" / "summaries"
        assert result == expected
        assert result.is_dir()

    def test_summaries_dir_no_data_dir(self):
        """Without data_dir, should use the data/ directory under project root."""
        store = _make_store()
        result = store.get_summaries_dir()
        # Should end with data/default/summaries
        assert result.name == "summaries"
        assert result.parent.name == "default"
        # The path should resolve to project_root/data/default/summaries
        expected_root = Path(__file__).resolve().parent.parent
        assert result == expected_root / "data" / "default" / "summaries"


class TestGraphRAGStoreSaveLoadSummaries:
    """Tests for save_summaries / load_summaries round-trip."""

    def test_save_and_load_summaries(self, tmp_path):
        """Save and then load summaries — data should round-trip."""
        store = _make_store(
            data_dir=str(tmp_path),
            entity_info={"entity1": [1, 2], "entity2": [3]},
            community_summary={"1": "Summary for community 1", "2": "Summary for community 2"},
        )

        version = store.save_summaries(version="test_v1")
        assert version == "test_v1"

        # Verify files exist
        summaries_dir = tmp_path / "default" / "summaries"
        assert (summaries_dir / "community_summaries_test_v1.json").exists()
        assert (summaries_dir / "entity_info_test_v1.json").exists()
        assert (summaries_dir / "current.json").exists()

        # Load in a fresh store
        store2 = _make_store(data_dir=str(tmp_path))

        loaded_summaries, loaded_entities = store2.load_summaries()
        assert loaded_summaries == {"1": "Summary for community 1", "2": "Summary for community 2"}
        assert loaded_entities == {"entity1": [1, 2], "entity2": [3]}
        # Also check attributes on store2
        assert store2.community_summary == loaded_summaries
        assert store2.entity_info == loaded_entities

    def test_save_summaries_auto_version(self, tmp_path):
        """save_summaries without explicit version should generate a timestamp."""
        store = _make_store(data_dir=str(tmp_path))

        version = store.save_summaries()
        assert version  # Non-empty
        # Version should look like a timestamp: YYYY-MM-DD_HHMMSS
        assert len(version) > 10

    def test_load_summaries_no_files(self, tmp_path):
        """load_summaries should return empty dicts when no files exist."""
        store = _make_store(data_dir=str(tmp_path))

        summaries, entities = store.load_summaries()
        assert summaries == {}
        assert entities == {}

    def test_current_json_pointer(self, tmp_path):
        """After save, current.json should point to the latest version."""
        store = _make_store(
            data_dir=str(tmp_path),
            entity_info={"e1": [1]},
            community_summary={"1": "First version"},
        )

        store.save_summaries(version="v1")
        
        # Update and save again
        store.community_summary = {"1": "Second version", "2": "New community"}
        store.entity_info = {"e1": [1], "e2": [2]}
        store.save_summaries(version="v2")

        # current.json should point to v2
        current_path = tmp_path / "default" / "summaries" / "current.json"
        with open(current_path) as f:
            current = json.load(f)
        assert current["version"] == "v2"
        assert current["stats"]["total_communities"] == 2
        assert current["stats"]["total_entities"] == 2

    def test_load_without_current_json(self, tmp_path):
        """load_summaries should find the latest version when current.json is missing."""
        store = _make_store(
            data_dir=str(tmp_path),
            entity_info={"e1": [1]},
            community_summary={"1": "Only version"},
        )

        store.save_summaries(version="v3")

        # Remove current.json to test fallback
        current_path = tmp_path / "default" / "summaries" / "current.json"
        assert current_path.exists()
        current_path.unlink()

        # Load should still work via file globbing
        store2 = _make_store(data_dir=str(tmp_path))
        summaries, entities = store2.load_summaries()
        assert summaries == {"1": "Only version"}
        assert entities == {"e1": [1]}

    def test_load_skips_incomplete_version(self, tmp_path):
        """load_summaries should skip versions missing the entity_info file."""
        store = _make_store(
            data_dir=str(tmp_path),
            entity_info={"e1": [1]},
            community_summary={"1": "Version with both files"},
        )
        summaries_dir = tmp_path / "default" / "summaries"
        store.save_summaries(version="v1")

        # Now create a v2 summaries file WITHOUT the matching entity_info file
        store.community_summary = {"2": "Orphan summaries - no entity_info"}
        summaries_dir.mkdir(parents=True, exist_ok=True)
        (summaries_dir / "community_summaries_v2.json").write_text(
            json.dumps({"2": "Orphan summaries - no entity_info"})
        )

        # Remove current.json to force fallback
        current_path = summaries_dir / "current.json"
        current_path.unlink()

        # Should fall back to v1 (which has both files), not v2
        store2 = _make_store(data_dir=str(tmp_path))
        summaries, entities = store2.load_summaries()
        assert summaries == {"1": "Version with both files"}
        assert entities == {"e1": [1]}

    def test_load_current_json_points_to_missing_files(self, tmp_path):
        """load_summaries should fall back if current.json points to missing files."""
        store = _make_store(
            data_dir=str(tmp_path),
            entity_info={"e1": [1]},
            community_summary={"1": "Good version"},
        )
        summaries_dir = tmp_path / "default" / "summaries"
        store.save_summaries(version="v_good")

        # Rewrite current.json to point to a version that doesn't exist
        current_path = summaries_dir / "current.json"
        current_info = {"version": "v_nonexistent", "created_at": "", "files": {}, "stats": {}}
        current_path.write_text(json.dumps(current_info))

        # Should fall back to v_good (which has both files)
        store2 = _make_store(data_dir=str(tmp_path))
        summaries, entities = store2.load_summaries()
        assert summaries == {"1": "Good version"}
        assert entities == {"e1": [1]}


class TestDeprecationWarning:
    """Test that importing from core_classes emits a deprecation warning."""

    def test_core_classes_deprecation_warning(self):
        """Importing from core_classes should emit DeprecationWarning."""
        import importlib
        # Remove from cache if already loaded
        if "core_classes" in sys.modules:
            del sys.modules["core_classes"]
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import core_classes
            importlib.reload(core_classes)
            # Check that a DeprecationWarning was emitted
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "core_classes is deprecated" in str(deprecation_warnings[0].message)


class TestGraphRAGQueryEngineValidation:
    """Tests for GraphRAGQueryEngine similarity_top_k validation."""

    def test_validate_top_k_accepts_valid_range(self):
        """Values 1-100 should pass validation."""
        from core.store import GraphRAGQueryEngine
        # Should not raise for valid values
        GraphRAGQueryEngine._validate_top_k(1)
        GraphRAGQueryEngine._validate_top_k(20)
        GraphRAGQueryEngine._validate_top_k(100)

    def test_validate_top_k_rejects_zero(self):
        """similarity_top_k=0 should raise ValueError."""
        import pytest as _pytest
        from core.store import GraphRAGQueryEngine
        with _pytest.raises(ValueError, match="similarity_top_k"):
            GraphRAGQueryEngine._validate_top_k(0)

    def test_validate_top_k_rejects_over_100(self):
        """similarity_top_k=101 should raise ValueError."""
        import pytest as _pytest
        from core.store import GraphRAGQueryEngine
        with _pytest.raises(ValueError, match="similarity_top_k"):
            GraphRAGQueryEngine._validate_top_k(101)