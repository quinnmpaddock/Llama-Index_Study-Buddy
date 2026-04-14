"""Workspace model and registry for multi-workspace support."""
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from models import slugify


logger = logging.getLogger(__name__)


def neo4j_db_name(workspace_id: str) -> str:
    """Convert a workspace ID to a valid Neo4j database name.
    
    Neo4j database names: alphanumeric + underscore, 3-63 chars, must start with letter.
    We prefix with 'sb_' to namespace, replace hyphens with underscores, and append
    a short hash to prevent collisions from truncation.
    """
    import hashlib
    import re
    
    sanitized = workspace_id.replace("-", "_")
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "", sanitized)
    
    hash_suffix = hashlib.sha256(workspace_id.encode()).hexdigest()[:8]
    
    prefix = "sb_"
    hash_part = "_" + hash_suffix
    
    max_sanitized_len = 63 - len(prefix) - len(hash_part)
    if len(sanitized) > max_sanitized_len:
        sanitized = sanitized[:max_sanitized_len]
    
    return prefix + sanitized + hash_part


@dataclass
class Workspace:
    """A knowledge base workspace with its own Neo4j database and data directory."""
    id: str
    name: str
    description: str
    neo4j_database: str
    created_at: str
    updated_at: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Workspace":
        """Create a Workspace from a dictionary (e.g., loaded from JSON)."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            neo4j_database=data["neo4j_database"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize workspace to a dictionary for JSON storage."""
        return asdict(self)


class WorkspaceRegistry:
    """Manages workspace lifecycle — create, list, get, delete.
    
    Persists workspace metadata to data_dir/workspaces.json.
    Each workspace also gets a data_dir/{workspace_id}/ directory.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.data_dir / "workspaces.json"
        self._workspaces: Dict[str, Workspace] = self._load()

    def _load(self) -> Dict[str, Workspace]:
        """Load workspaces from the registry file."""
        if not self._registry_path.exists():
            return {}
        try:
            with open(self._registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                ws_id: Workspace.from_dict(ws_data)
                for ws_id, ws_data in data.items()
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to load workspaces.json: %s", e)
            return {}

    def _save(self):
        """Persist workspaces to the registry file."""
        data = {ws_id: ws.to_dict() for ws_id, ws in self._workspaces.items()}
        with open(self._registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def create(self, name: str, description: str = "") -> Workspace:
        """Create a new workspace.
        
        Args:
            name: Human-readable name.
            description: What this workspace covers.
        Returns:
            The created Workspace.
        Raises:
            ValueError: If a workspace with the derived slug already exists.
        """
        workspace_id = slugify(name)
        if workspace_id in self._workspaces:
            raise ValueError(f"Workspace '{workspace_id}' already exists")

        now = datetime.now(timezone.utc).isoformat()
        workspace = Workspace(
            id=workspace_id,
            name=name,
            description=description,
            neo4j_database=neo4j_db_name(workspace_id),
            created_at=now,
            updated_at=now,
        )

        # Create workspace data directory
        ws_dir = self.data_dir / workspace_id
        ws_dir.mkdir(parents=True, exist_ok=True)

        # Write default workspace config
        config_path = ws_dir / "config.yaml"
        if not config_path.exists():
            config_path.write_text(
                f"# Workspace config overrides for '{name}'\n"
                f"# Any field left commented out will use the global default.\n\n"
                f"# llm_model: meta-llama/llama-4-scout-17b-16e-instruct\n"
                f"# embedding_model: KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5\n"
                f"# max_paths_per_chunk: 2\n"
            )

        self._workspaces[workspace_id] = workspace
        self._save()
        logger.info("Created workspace '%s' (id=%s)", name, workspace_id)
        return workspace

    def get(self, workspace_id: str) -> Optional[Workspace]:
        """Get a workspace by ID. Returns None if not found."""
        return self._workspaces.get(workspace_id)

    def list(self) -> List[Workspace]:
        """List all workspaces."""
        return list(self._workspaces.values())

    def delete(self, workspace_id: str) -> bool:
        """Delete a workspace.
        
        Removes from registry and deletes its data directory.
        Does NOT drop the Neo4j database (that must be done separately).
        Returns True if deleted, False if not found.
        """
        if workspace_id not in self._workspaces:
            return False

        # Remove workspace data directory
        ws_dir = self.data_dir / workspace_id
        if ws_dir.exists():
            import shutil
            shutil.rmtree(ws_dir)

        del self._workspaces[workspace_id]
        self._save()
        logger.info("Deleted workspace '%s'", workspace_id)
        return True

    def get_or_create_database(self, workspace_id: str) -> str:
        """Get the Neo4j database name for a workspace.
        
        Creates the Neo4j database if it doesn't exist yet.
        """
        ws = self.get(workspace_id)
        if ws is None:
            raise ValueError(f"Workspace '{workspace_id}' not found")
        return ws.neo4j_database