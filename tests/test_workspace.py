"""Tests for Workspace and WorkspaceRegistry."""
import json
import pytest
from pathlib import Path
from src.workspace import Workspace, WorkspaceRegistry, neo4j_db_name


def test_workspace_from_dict():
    data = {
        "id": "ml-research",
        "name": "ML Research",
        "description": "Knowledge base for ML papers",
        "neo4j_database": neo4j_db_name("ml-research"),
        "created_at": "2026-04-13T00:00:00",
        "updated_at": "2026-04-13T00:00:00",
    }
    ws = Workspace.from_dict(data)
    assert ws.id == "ml-research"
    assert ws.neo4j_database == neo4j_db_name("ml-research")


def test_workspace_to_dict():
    ws = Workspace(
        id="ml-research",
        name="ML Research",
        description="ML papers",
        neo4j_database=neo4j_db_name("ml-research"),
        created_at="2026-04-13T00:00:00",
        updated_at="2026-04-13T00:00:00",
    )
    d = ws.to_dict()
    assert d["id"] == "ml-research"
    assert d["neo4j_database"] == neo4j_db_name("ml-research")


def test_neo4j_db_name():
    result = neo4j_db_name("ml-research")
    assert result.startswith("sb_")
    assert "ml_research" in result
    assert len(result) <= 63
    # Deterministic: same input always gives same output
    assert neo4j_db_name("ml-research") == result
    
    result_bio = neo4j_db_name("bio")
    assert result_bio.startswith("sb_")
    assert "bio" in result_bio


def test_neo4j_db_name_long():
    # Neo4j db names max 63 chars
    long_id = "a" * 70
    result = neo4j_db_name(long_id)
    assert len(result) <= 63


def test_registry_create(tmp_path):
    registry = WorkspaceRegistry(data_dir=tmp_path)
    ws = registry.create(name="ML Research", description="ML papers")
    assert ws.id == "ml-research"
    assert ws.neo4j_database == neo4j_db_name("ml-research")
    assert (tmp_path / "ml-research").is_dir()
    assert (tmp_path / "ml-research" / "config.yaml").exists()


def test_registry_list(tmp_path):
    registry = WorkspaceRegistry(data_dir=tmp_path)
    registry.create(name="ML Research")
    registry.create(name="Biology Notes")
    workspaces = registry.list()
    assert len(workspaces) == 2


def test_registry_get(tmp_path):
    registry = WorkspaceRegistry(data_dir=tmp_path)
    created = registry.create(name="ML Research")
    fetched = registry.get("ml-research")
    assert fetched.id == created.id


def test_registry_get_not_found(tmp_path):
    registry = WorkspaceRegistry(data_dir=tmp_path)
    assert registry.get("nonexistent") is None


def test_registry_delete(tmp_path):
    registry = WorkspaceRegistry(data_dir=tmp_path)
    ws = registry.create(name="ML Research")
    assert registry.delete(ws.id)
    assert registry.get("ml-research") is None
    assert not (tmp_path / "ml-research").exists()


def test_registry_duplicate_create_raises(tmp_path):
    registry = WorkspaceRegistry(data_dir=tmp_path)
    registry.create(name="ML Research")
    with pytest.raises(ValueError, match="already exists"):
        registry.create(name="ML Research")


def test_registry_persistence(tmp_path):
    """Registry state persists across instances."""
    registry1 = WorkspaceRegistry(data_dir=tmp_path)
    registry1.create(name="ML Research")
    
    # Create a new instance pointing to the same directory
    registry2 = WorkspaceRegistry(data_dir=tmp_path)
    assert registry2.get("ml-research") is not None
    assert registry2.get("ml-research").name == "ML Research"