# Phase 0: Foundation Refactor — Multi-Workspace Support

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Restructure study-buddy to support multiple independent knowledge bases (workspaces), each with its own Neo4j database, config overrides, and data directory.

**Architecture:** Service layer pattern — thin FastAPI routes delegate to service classes that accept `workspace_id` as first parameter. Each workspace gets a Neo4j database and a `data/{workspace_id}/` directory. Single `workspaces.json` registry tracks all workspaces. Config resolution: workspace override → global config → hardcoded default.

**Tech Stack:** Python 3.11+, FastAPI, LlamaIndex, Neo4j (multi-DB), Pydantic v2, pytest

---

## Current State Summary

| File | Lines | Role |
|------|-------|------|
| `src/app.py` | 1281 | Monolithic: lifespan, API models, all endpoints, ingestion pipeline |
| `src/core_classes.py` | 525 | `GraphRAGExtractor`, `GraphRAGStore`, `GraphRAGQueryEngine` |
| `src/config.py` | 228 | Singleton `Config` from `study_buddy.yaml` |
| `src/ingestion.py` | 129 | `DocumentIngestion` file parser/router |
| `src/main.py` | 168 | Legacy standalone script (to be deleted) |
| `summaries/` | — | Flat JSON files for community summaries/entity info |

**Key issues:**
- `Config` is a singleton via `get_config()` — can't support per-workspace overrides
- `app.state.*` holds global engine/summaries/entity_info — single workspace only
- `GraphRAGStore.community_summary` and `entity_info` are in-memory dicts, not persisted to Neo4j
- `GraphRAGStore._run_cypher()` already passes `database_=self.graph_name` — good, but `build_communities()` uses `self.graph_name` for GDS projection names which will collide across workspaces
- No test infrastructure exists

---

## Task 0.0: Test Infrastructure Setup

**Objective:** Create a test foundation before refactoring. No code changes — just project structure.

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `requirements_test.txt` or update `pyproject.toml`

**Step 1: Create test directory and conftest**

Create `tests/conftest.py`:
```python
"""Shared test fixtures for study-buddy tests."""
import os
import pytest
from pathlib import Path

# Test data directory
TESTS_DIR = Path(__file__).parent
FIXTURES_DIR = TESTS_DIR / "fixtures"


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary data directory for workspace tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_workspace_config():
    """Provide a sample workspace config dict for tests."""
    return {
        "llm": {"model": "test-model"},
        "embedding": {"model": "test-embedding"},
        "graphrag": {"max_paths_per_chunk": 5},
    }
```

**Step 2: Add test dependencies**

Add to `requirements_minimal.txt` (or create `requirements_test.txt`):
```
pytest>=8.0
pytest-asyncio>=0.23
```

**Step 3: Verify tests run**

```bash
cd src && python -m pytest ../tests/ -v --co
```
Expected: collects 0 tests (no test files yet), no import errors.

**Step 4: Commit**

```bash
git add tests/ requirements_test.txt
git commit -m "chore: add test infrastructure for phase 0"
```

---

## Task 0.1: Refactor Config — Remove Singleton, Add WorkspaceConfig

**Objective:** Replace the singleton `Config` with an instantiable class that supports workspace-level overrides. Keep backward compatibility via `get_config()` that returns the global config.

**Files:**
- Modify: `src/config.py` (refactored)
- Create: `tests/test_config.py`

### Sub-task 0.1a: Add WorkspaceConfig dataclass

**Step 1: Write failing test for WorkspaceConfig**

Create `tests/test_config.py`:
```python
"""Tests for config module — workspace overrides, resolution chain."""
import pytest
from src.config import Config, WorkspaceConfig


def test_workspace_config_defaults():
    """WorkspaceConfig with no overrides returns global defaults."""
    global_config = Config.__new__(Config)
    global_config.llm = type("LLM", (), {"model": "gpt-4", "api_base": "https://api.openai.com/v1", "api_key": "test-key"})()
    global_config.embedding = type("Emb", (), {"model": "text-embedding-3-small"})()
    global_config.graphrag = type("GR", (), {"max_paths_per_chunk": 10, "extraction_prompt": "default.txt"})()

    ws = WorkspaceConfig()
    assert ws.llm_model is None  # None means "use global"


def test_workspace_config_override():
    """WorkspaceConfig overrides take precedence over global config."""
    ws = WorkspaceConfig(llm_model="ollama/llama3")
    resolved = ws.resolve({"llm_model": "gpt-4", "embedding_model": "bge-small"})
    assert resolved["llm_model"] == "ollama/llama3"
    assert resolved["embedding_model"] == "bge-small"  # fallback to global


def test_config_get_returns_same_instance():
    """get_config() still returns a consistent global Config."""
    from src.config import get_config, reset_config
    reset_config()
    # Config will fail without OPENAI_API_KEY in tests, so we skip if not set
    # This test validates the pattern, not full config loading
```

**Step 2: Run test to verify failure**

```bash
cd src && python -m pytest ../tests/test_config.py -v -k "workspace"
```
Expected: FAIL — `ImportError: cannot import name 'WorkspaceConfig'`

**Step 3: Implement WorkspaceConfig**

Add to `src/config.py` (append after existing classes, before `Config`):
```python
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class WorkspaceConfig:
    """Per-workspace config overrides.

    Any field set to None falls back to the global config value.
    """
    llm_model: Optional[str] = None
    llm_api_base: Optional[str] = None
    embedding_model: Optional[str] = None
    max_paths_per_chunk: Optional[int] = None
    extraction_prompt: Optional[str] = None
    neo4j_database: Optional[str] = None

    def resolve(self, global_defaults: dict) -> dict:
        """Merge workspace overrides with global defaults.

        Workspace value takes precedence if not None.
        """
        return {
            key: (getattr(self, key) if getattr(self, key) is not None else global_defaults.get(key))
            for key in [
                "llm_model", "llm_api_base", "embedding_model",
                "max_paths_per_chunk", "extraction_prompt", "neo4j_database",
            ]
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "WorkspaceConfig":
        """Load workspace config from a YAML file."""
        if not path.exists():
            return cls()
        try:
            import yaml
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
        except ImportError:
            data = cls._parse_simple_yaml(path)
        return cls(
            llm_model=data.get("llm_model"),
            llm_api_base=data.get("llm_api_base"),
            embedding_model=data.get("embedding_model"),
            max_paths_per_chunk=data.get("max_paths_per_chunk"),
            extraction_prompt=data.get("extraction_prompt"),
            neo4j_database=data.get("neo4j_database"),
        )

    @staticmethod
    def _parse_simple_yaml(path: Path) -> dict:
        """Basic YAML parser (mirrors Config._parse_simple_yaml)."""
        data = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.rstrip()
                    if not line or line.strip().startswith("#"):
                        continue
                    if ":" in line and not line.startswith(" "):
                        key, value = line.split(":", 1)
                        data[key.strip()] = value.strip().strip('"').strip("'")
        except OSError:
            pass
        return data
```

Also modify `Config` class to add a `workspace_defaults()` method:
```python
    def workspace_defaults(self) -> dict:
        """Return global config values as a dict for WorkspaceConfig.resolve()."""
        return {
            "llm_model": self.llm.model,
            "llm_api_base": self.llm.api_base,
            "embedding_model": self.embedding.model,
            "max_paths_per_chunk": self.graphrag.max_paths_per_chunk,
            "extraction_prompt": self.graphrag.extraction_prompt,
        }
```

**Step 4: Run test to verify pass**

```bash
cd src && python -m pytest ../tests/test_config.py -v -k "workspace"
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat(config): add WorkspaceConfig with override resolution chain"
```

---

## Task 0.2: Create Workspace Model and Registry

**Objective:** Create `Workspace` dataclass and `WorkspaceRegistry` that manages workspaces in `data/workspaces.json`, including Neo4j database creation.

**Files:**
- Create: `src/workspace.py`
- Create: `src/models.py` (Pydantic models)
- Create: `tests/test_workspace.py`

### Sub-task 0.2a: Create Pydantic models

**Step 1: Write failing test**

Create `tests/test_models.py`:
```python
"""Tests for Pydantic models."""
from src.models import WorkspaceCreate, WorkspaceInfo


def test_workspace_create_model():
    req = WorkspaceCreate(name="ML Research", description="My ML knowledge base")
    assert req.name == "ML Research"
    assert req.slug is None  # auto-generated from name


def test_workspace_info_model():
    info = WorkspaceInfo(
        id="ml-research",
        name="ML Research",
        description="My ML knowledge base",
        neo4j_database="sb_ml_research",
        created_at="2026-04-13T00:00:00",
        updated_at="2026-04-13T00:00:00",
        entity_count=0,
        community_count=0,
    )
    assert info.id == "ml-research"
```

**Step 2: Create `src/models.py`**

```python
"""Shared Pydantic models for Study Buddy API."""
from datetime import datetime
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
import re


def slugify(name: str) -> str:
    """Convert a name to a URL-safe slug."""
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
        return self.slug or slugify(self.name)


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


# Keep existing API models from app.py for backward compat
class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to ask the knowledge graph")
    similarity_top_k: int = Field(default=20, ge=1, le=50)


class GraphQueryResponse(BaseModel):
    """to enforce structured metadata returns"""
    answer: str
    communities_consulted: List[str | int]
    entities_found: List[str]
```

**Step 3: Run test**

```bash
cd src && python -m pytest ../tests/test_models.py -v
```
Expected: PASS

**Step 4: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat(models): shared Pydantic models for workspaces"
```

### Sub-task 0.2b: Create Workspace dataclass and registry

**Step 1: Write failing test**

Create `tests/test_workspace.py`:
```python
"""Tests for Workspace and WorkspaceRegistry."""
import json
import pytest
from pathlib import Path
from src.workspace import Workspace, WorkspaceRegistry


def test_workspace_from_dict():
    data = {
        "id": "ml-research",
        "name": "ML Research",
        "description": "Knowledge base for ML papers",
        "neo4j_database": "sb_ml_research",
        "created_at": "2026-04-13T00:00:00",
        "updated_at": "2026-04-13T00:00:00",
    }
    ws = Workspace.from_dict(data)
    assert ws.id == "ml-research"
    assert ws.neo4j_database == "sb_ml_research"


def test_workspace_to_dict():
    ws = Workspace(
        id="ml-research",
        name="ML Research",
        description="ML papers",
        neo4j_database="sb_ml_research",
        created_at="2026-04-13T00:00:00",
        updated_at="2026-04-13T00:00:00",
    )
    d = ws.to_dict()
    assert d["id"] == "ml-research"
    assert d["neo4j_database"] == "sb_ml_research"


def test_registry_create(tmp_path):
    registry = WorkspaceRegistry(data_dir=tmp_path)
    ws = registry.create(name="ML Research", description="ML papers")
    assert ws.id == "ml-research"
    assert ws.neo4j_database == "sb_ml_research"
    # Workspace directory created
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
    registry.delete(ws.id)
    assert registry.get("ml-research") is None


def test_neo4j_database_naming():
    """Database names must be valid for Neo4j (alphanumeric + underscore, 3-63 chars)."""
    # Expressions like "ml-research" → database "sb_ml_research"
    from src.workspace import neo4j_db_name
    assert neo4j_db_name("ml-research") == "sb_ml_research"
    assert neo4j_db_name("bio") == "sb_bio"  # short slug
```

**Step 2: Create `src/workspace.py`**

```python
"""Workspace model and registry for multi-workspace support."""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


def neo4j_db_name(workspace_id: str) -> str:
    """Convert a workspace ID to a valid Neo4j database name.

    Neo4j database names: alphanumeric + underscore, 3-63 chars, must start with letter.
    We prefix with 'sb_' to namespace and replace hyphens with underscores.
    """
    db_name = "sb_" + workspace_id.replace("-", "_")
    # Ensure it starts with a letter (sb_ prefix handles this)
    # Truncate to 63 chars max
    return db_name[:63]


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
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            neo4j_database=data["neo4j_database"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    def to_dict(self) -> Dict[str, Any]:
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
            with open(self._registry_path, "r") as f:
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
        with open(self._registry_path, "w") as f:
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
        from src.models import slugify

        workspace_id = slugify(name)
        if workspace_id in self._workspaces:
            raise ValueError(f"Workspace '{workspace_id}' already exists")

        now = datetime.utcnow().isoformat()
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
```

**Step 3: Create `src/__init__.py` update to make imports work**

Current `src/__init__.py` is empty. No change needed — imports use `from src.config import ...` style which works with pytest from project root.

**Step 4: Run tests**

```bash
cd /path/to/worktree && python -m pytest tests/test_workspace.py tests/test_models.py -v
```
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add src/workspace.py src/models.py tests/test_workspace.py tests/test_models.py
git commit -m "feat(workspace): Workspace model, registry, and Pydantic models"
```

---

## Task 0.3: Refactor GraphRAGStore for Workspace Context

**Objective:** Make `GraphRAGStore` workspace-aware — accept a `workspace_id`, namespace GDS projections, and store summaries/entity_info in per-workspace data directories.

**Files:**
- Create: `src/core/__init__.py`
- Create: `src/core/store.py` (extracted and refactored from core_classes.py)
- Modify: `src/core_classes.py` (import from new location, emit deprecation)
- Create: `tests/test_store.py`

### Sub-task 0.3a: Extract GraphRAGStore to src/core/store.py

**Step 1: Create package structure**

```bash
mkdir -p src/core
touch src/core/__init__.py
```

**Step 2: Create `src/core/store.py`**

Copy `GraphRAGStore` and `GraphRAGQueryEngine` from `core_classes.py` with these changes:

1. `GraphRAGStore.__init__` gains `workspace_id: Optional[str] = None` and `data_dir: Optional[str] = None` params
2. `self.graph_name` set to `neo4j_db_name(workspace_id)` if `workspace_id` given, else falls back to `database` param (backward compat)
3. `build_communities()` uses `f"{self.workspace_id}_graph"` as GDS projection name (avoiding `self.graph_name` which is the DB name, not a GDS name)
4. New method `save_summaries()` and `load_summaries()` that read/write to `data/{workspace_id}/` instead of flat `summaries/`

Key changes in `build_communities`:
```python
# BEFORE: GDS projection uses self.graph_name (which is the database name)
# AFTER: Use workspace-scoped projection name
if self.workspace_id:
    gds_projection = f"{self.workspace_id}_graph"
else:
    gds_projection = self.graph_name  # backward compat
```

Key changes in save/load:
```python
def _summaries_dir(self) -> Path:
    """Return the directory for this workspace's summaries."""
    if self.data_dir:
        d = Path(self.data_dir) / (self.workspace_id or "default") / "summaries"
    else:
        # Legacy behavior: flat summaries/ directory
        d = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "summaries"
    d.mkdir(parents=True, exist_ok=True)
    return d
```

**Step 3: Update `src/core_classes.py` to re-export from new locations**

Add deprecation imports at top of `core_classes.py`:
```python
# Backward compatibility — these classes now live in src.core modules
import warnings
warnings.warn(
    "Importing from core_classes is deprecated. Use src.core.extractor and src.core.store.",
    DeprecationWarning,
    stacklevel=2,
)
from src.core.extractor import GraphRAGExtractor
from src.core.store import GraphRAGStore, GraphRAGQueryEngine
```

**Step 4: Write tests for workspace-scoped store**

```python
# tests/test_store.py
"""Tests for GraphRAGStore workspace awareness."""
import pytest
from src.core.store import GraphRAGStore


def test_default_graph_name():
    """Without workspace_id, graph_name defaults to database param."""
    store = GraphRAGStore.__new__(GraphRAGStore)
    store.workspace_id = None
    store.graph_name = "neo4j"
    assert store.graph_name == "neo4j"


def test_workspace_scoped_gds_projection():
    """With workspace_id, GDS projection uses workspace prefix."""
    store = GraphRAGStore.__new__(GraphRAGStore)
    store.workspace_id = "ml-research"
    store.graph_name = "sb_ml_research"
    # The build_communities method should use ml-research_graph
    # for the GDS projection name, not sb_ml_research
    gds_name = f"{store.workspace_id}_graph"
    assert gds_name == "ml-research_graph"
```

**Step 5: Verify existing imports still work**

```bash
cd src && python -c "from core_classes import GraphRAGStore; print('backward compat OK')"
```
Expected: prints warning, then OK.

**Step 6: Commit**

```bash
git add src/core/__init__.py src/core/store.py src/core_classes.py tests/test_store.py
git commit -m "refactor(store): extract GraphRAGStore to core/store.py with workspace awareness"
```

---

## Task 0.4: Refactor GraphRAGExtractor to core/extractor.py

**Objective:** Move `GraphRAGExtractor` to its own module in `src/core/extractor.py`, with support for per-workspace extraction prompts.

**Files:**
- Create: `src/core/extractor.py`
- Create: `src/core/query.py` (move `GraphRAGQueryEngine` here too)
- Modify: `src/core_classes.py` (update deprecation import)

### Sub-task 0.4a: Extract GraphRAGExtractor

**Step 1: Create `src/core/extractor.py`**

Move the `GraphRAGExtractor` class verbatim from `core_classes.py` into `src/core/extractor.py`. Include the imports it needs.

No functional changes — just file reorganization. The `parse_fn` stays in `app.py` for now (it will move to a service later).

**Step 2: Update `src/core_classes.py` deprecation imports**

Update to import from both new modules:
```python
from src.core.extractor import GraphRAGExtractor
from src.core.store import GraphRAGStore, GraphRAGQueryEngine
```

**Step 3: Verify**

```bash
cd src && python -c "from core_classes import GraphRAGExtractor, GraphRAGStore, GraphRAGQueryEngine; print('OK')"
```

**Step 4: Commit**

```bash
git add src/core/extractor.py src/core/query.py src/core_classes.py
git commit -m "refactor(extractor): extract GraphRAGExtractor and QueryEngine to core/"
```

---

## Task 0.5: Refactor app.py into Service Modules

**Objective:** Split the 1281-line `app.py` into focused service modules. `app.py` becomes a thin FastAPI router.

This is the largest task. We'll do it as a multi-step extraction.

**Files:**
- Create: `src/services/__init__.py`
- Create: `src/services/ingestion.py` (from `DocumentIngestion` + `run_full_ingestion`)
- Create: `src/services/graph.py` (entity/community lookup logic)
- Create: `src/services/community.py` (summary management, versioning)
- Create: `src/services/query.py` (query engine setup + execution)
- Rewrite: `src/app.py` (thin router)

### Sub-task 0.5a: Create services package

```bash
mkdir -p src/services
touch src/services/__init__.py
```

### Sub-task 0.5b: Extract ingestion service

Move `DocumentIngestion` class, `run_full_ingestion` function, `parse_fn`, `extract_json`, and the ingestion-status tracking dict into `src/services/ingestion.py`.

The service class:
```python
# src/services/ingestion.py
class IngestionService:
    def __init__(self, config, workspace_registry):
        self.config = config
        self.workspace_registry = workspace_registry
        self._status: Dict[str, dict] = {}

    def start_ingestion(self, workspace_id: str, directory: str, files: List[str] | None = None) -> str:
        """Start background ingestion. Returns task_id."""
        ...

    def get_status(self, task_id: str) -> dict | None:
        ...

    def _run_ingestion(self, workspace_id: str, directory: str, files: List[str], task_id: str):
        """Background task that runs the full pipeline."""
        ...
```

Key change: `_run_ingestion` receives `workspace_id` and uses workspace-scoped graph store and data directories.

### Sub-task 0.5c: Extract community service

Move `load_summaries_and_entity_info()`, `find_most_recent_snapshot()`, summary CRUD, and version management into `src/services/community.py`.

```python
# src/services/community.py
class CommunityService:
    def __init__(self, workspace_registry, config):
        self.workspace_registry = workspace_registry
        self.config = config

    def load_summaries(self, workspace_id: str) -> tuple[dict, dict]:
        """Load community summaries and entity info for a workspace."""
        ...

    def save_summaries(self, workspace_id: str, summaries: dict, entity_info: dict) -> str:
        """Save summaries to workspace data dir. Returns version timestamp."""
        ...

    def list_versions(self, workspace_id: str) -> list[dict]:
        ...

    def cleanup_versions(self, workspace_id: str, keep: int = 5) -> list[str]:
        ...
```

### Sub-task 0.5d: Extract graph service

Move entity search, community listing, and entity detail logic into `src/services/graph.py`.

```python
# src/services/graph.py
class GraphService:
    def __init__(self, workspace_registry, config):
        ...

    def search_entities(self, workspace_id: str, query: str | None = None, limit: int = 50) -> dict:
        ...

    def get_entity(self, workspace_id: str, name: str) -> dict | None:
        ...

    def list_communities(self, workspace_id: str) -> dict:
        ...

    def get_community(self, workspace_id: str, community_id: int) -> dict | None:
        ...

    def get_community_entities(self, workspace_id: str, community_id: int) -> dict:
        ...
```

### Sub-task 0.5e: Extract query service

```python
# src/services/query.py
class QueryService:
    def __init__(self, workspace_registry, config):
        ...
        self._engines: Dict[str, GraphRAGQueryEngine] = {}

    def get_engine(self, workspace_id: str) -> GraphRAGQueryEngine:
        """Get or create a query engine for the workspace."""
        ...

    async def query(self, workspace_id: str, query: str, similarity_top_k: int = 20) -> dict:
        ...
```

### Sub-task 0.5f: Rewrite app.py as thin router

The new `app.py` should be ~200 lines:
1. Import services, models, config
2. Setup lifespan: initialize `Config`, `WorkspaceRegistry`, services
3. Define API routes that delegate to service methods
4. Keep all `@app.get/post` definitions, just thinner

**Step: Commit each sub-task**

```bash
git add src/services/ingestion.py && git commit -m "refactor(services): extract IngestionService"
git add src/services/community.py && git commit -m "refactor(services): extract CommunityService"
git add src/services/graph.py && git commit -m "refactor(services): extract GraphService"
git add src/services/query.py && git commit -m "refactor(services): extract QueryService"
git add src/app.py && git commit -m "refactor(app): thin router delegating to services"
```

---

## Task 0.6: Workspace-Aware API Endpoints

**Objective:** Add `/kb` route prefix for workspace management, and namespace existing endpoints under `/kb/{workspace_id}`.

**Files:**
- Modify: `src/app.py` (new routes)

### Step-by-step

**API surface:**

```
POST   /kb                     — create workspace
GET    /kb                      — list workspaces
GET    /kb/{workspace_id}      — get workspace info
DELETE /kb/{workspace_id}      — delete workspace
GET    /kb/{workspace_id}/stats — workspace statistics

GET    /kb/{workspace_id}/query          — query graph (was /query)
POST   /kb/{workspace_id}/ingest         — ingest documents (was /ingest)
GET    /kb/{workspace_id}/entities        — search entities (was /entities)
GET    /kb/{workspace_id}/entities/{name} — get entity (was /entities/{name})
GET    /kb/{workspace_id}/communities     — list communities (was /communities)
...etc

# Legacy routes (no workspace) point to a "default" workspace
GET    /query          → /kb/default/query
POST   /ingest         → /kb/default/ingest
...
```

**Implementation approach:**

1. Add a `DEFAULT_WORKSPACE_ID = "default"` constant
2. On first startup, auto-create the "default" workspace if it doesn't exist
3. Legacy routes call the same service methods with `workspace_id="default"`
4. New `/kb/{workspace_id}/...` routes use the service layer directly

**Commit:**

```bash
git add src/app.py src/models.py
git commit -m "feat(api): workspace-aware endpoints with /kb prefix and legacy compat"
```

---

## Task 0.7: Workspace-Aware Data Storage

**Objective:** Move from flat `summaries/` to `data/{workspace_id}/summaries/` directories. Version snapshots per-workspace.

**Files:**
- Modify: `src/services/community.py` (workspace-scoped paths)
- Modify: `src/services/ingestion.py` (workspace-scoped output)
- Create: migration script for existing data

### Key changes

1. `CommunityService.load_summaries()` reads from `data/{workspace_id}/summaries/` instead of `summaries/`
2. `CommunityService.save_summaries()` writes to per-workspace directories
3. On startup, if "default" workspace has no summaries data but `summaries/` exists, auto-migrate existing data
4. `SUMMARIES_DIR` constant in `app.py` is replaced by `workspace_registry.data_dir / workspace_id / "summaries"`

### Migration

```python
# In lifespan startup, after creating "default" workspace:
data_dir = Path(BASE_DIR) / "data" / "default" / "summaries"
old_dir = Path(BASE_DIR) / ".." / "summaries"
if old_dir.exists() and not data_dir.exists():
    import shutil
    shutil.copytree(old_dir, data_dir)
    logger.info("Migrated existing summaries to default workspace data directory")
```

**Commit:**

```bash
git add src/services/ src/app.py
git commit -m "feat(storage): workspace-scoped data directories, auto-migrate legacy summaries/"
```

---

## Task 0.8: Update Rust CLI for Multi-Workspace (Deferred)

**Objective:** Update the Rust CLI to support workspace selection. This is lower priority and can be done in a follow-up since the REST API is the primary interface.

**Files:**
- Modify: `cli/src/api/models.rs` — add workspace request/response models
- Modify: `cli/src/api/client.rs` — add workspace API calls
- Create: `cli/src/commands/workspace.rs` — workspace subcommand
- Modify: `cli/src/commands/mod.rs` — register workspace module

**This task is deferred** — the REST API changes in Task 0.6 are the foundation. The Rust CLI can be updated whenever convenient since it's just an HTTP client.

---

## Verification Plan

After all tasks are complete:

1. **Unit tests pass**: `pytest tests/ -v`
2. **Existing API still works**: Start server with Neo4j, hit legacy endpoints (`/query`, `/entities`, `/communities`, `/ingest`) — they should work with auto-migrated "default" workspace
3. **Workspace isolation**:
   ```
   POST /kb  {"name": "Test WS 1", "description": "First workspace"}
   POST /kb  {"name": "Test WS 2", "description": "Second workspace"}
   # Ingest same doc into both
   POST /kb/test-ws-1/ingest  {"directory": "/path/to/docs"}
   POST /kb/test-ws-2/ingest  {"directory": "/path/to/docs"}
   # Verify they have separate Neo4j databases
   GET /kb  # lists both workspaces
   # Query one, verify no cross-contamination
   GET /kb/test-ws-1/query  {"query": "what is..."}
   ```
4. **Delete workspace**: `DELETE /kb/test-ws-2` — verify test-ws-1 unaffected
5. **Backward compat**: Legacy `/query` works (maps to default workspace)

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| LlamaIndex `Settings` singleton conflicts with per-workspace LLM/embedding | Phase 0 uses global Settings for all workspaces; per-workspace model selection is Phase 4 |
| Refactoring app.py may break existing endpoints | Keep legacy routes that delegate to service layer; add integration test |
| Neo4j Community Edition limits on concurrent databases | Test with 3-5 workspaces; add resource monitoring |
| GDS projection name collisions | Use workspace-specific projection names (`{workspace_id}_graph`) |
| Data migration from flat summaries/ | Auto-migrate on first startup if "default" workspace has no data but summaries/ exists |

---

## Dependency Order

```
0.0 Test infra
 ↓
0.1 Config refactor
 ↓
0.2 Workspace model + registry ← depends on models from 0.1
 ↓
0.3 Store refactor ← depends on workspace registry for database names
 ↓
0.4 Extractor refactor ← independent, pure file move
 ↓
0.5 Service extraction ← depends on 0.2, 0.3, 0.4
 ↓
0.6 API endpoints ← depends on 0.5 services
 ↓
0.7 Data storage ← depends on 0.5, 0.6
 ↓
0.8 Rust CLI ← depends on 0.6 API surface
```

Tasks 0.3 and 0.4 can be done in parallel. Task 0.5 is the critical path.

---

## Open Questions

1. **Should we store community summaries in Neo4j (as nodes) instead of JSON files?** The PROJECT_SCOPE says yes (section 5.5), but this is a bigger change. For Phase 0, we should keep the JSON-sidecar approach but make it per-workspace, then migrate to Neo4j storage in Phase 1 or later.

2. **Should the default workspace be auto-created on startup?** Yes — this ensures backward compatibility. If `summaries/` exists, its data migrates into `data/default/summaries/`.

3. **Thread safety for WorkspaceRegistry?** Since it's a JSON file, concurrent writes could corrupt it. For Phase 0 (single-process FastAPI), a `threading.Lock` is sufficient. Later phases may need SQLite or a more robust store.