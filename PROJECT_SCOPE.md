# Study Buddy — Project Scope & Implementation Roadmap

> **Created:** 2026-04-13  
> **Status:** Living document — update as implementation progresses

---

## 1. Vision

Study Buddy is an **agent-interactable knowledge graph application** that:

1. **Intakes documents and builds property graphs** — entities, relationships, communities — using LLM extraction and Neo4j storage.
2. **Supports multiple separate knowledge bases** (workspaces) on a single device — each with its own graph, config, and metadata.
3. **Provides built-in tools for agent interaction** via MCP (Model Context Protocol) — any agent that speaks MCP can query, ingest, and manage knowledge bases.
4. **Displays select graph data via a TUI** — a Textual-based terminal interface for browsing entities, communities, and query results.

The target user is not a human with a browser — it's an AI agent (like Hermes) manipulating knowledge graphs programmatically, with an optional TUI for human inspection.

---

## 2. Current Architecture Summary

```
Documents → DocumentIngestion → LLM extraction → Neo4j (PropertyGraphIndex)
                                                          │
                                                Leiden Community Detection (GDS)
                                                          │
                                                Community Summaries (in-memory dicts + JSON files)
                                                          │
Query → GraphRAGQueryEngine → Vector Search → Entity Lookup → Community Summaries → LLM → Answer
```

**What exists today:**
- `src/app.py` — FastAPI server with single-workspace endpoints
- `src/core_classes.py` — `GraphRAGExtractor`, `GraphRAGStore(Neo4jPropertyGraphStore)`, `GraphRAGQueryEngine`
- `src/ingestion.py` — `DocumentIngestion` routing files to parsers
- `src/config.py` — Singleton `Config` from `study_buddy.yaml` + env vars
- `src/main.py` — Legacy standalone ingestion script
- `cli/` — Rust CLI client (HTTP client wrapping the API)
- `summaries/` — Flat JSON files for community summaries + entity info per version

**What's missing for the vision:**
- Multi-workspace support (one graph per knowledge base)
- MCP server (agent-connectable tools)
- TUI (Textual-based terminal interface)
- Synthesis/compounding layer (knowledge accumulation across queries)
- Per-workspace schema configuration
- Incremental ingestion without full rebuild
- Lint/health-check system

---

## 3. Architecture Redesign

### 3.1 Workspace Model

A **workspace** is the top-level organizing unit. Each workspace is an independent knowledge base with its own graph, metadata, and configuration.

```python
@dataclass
class Workspace:
    id: str                    # Unique identifier (slug)
    name: str                  # Human-readable name
    description: str           # What this workspace covers
    created_at: datetime
    updated_at: datetime
    neo4j_database: str        # Neo4j database name for this workspace
    config: WorkspaceConfig   # Per-workspace config overrides (LLM, embedding, etc.)
    schema: WorkspaceSchema   # Entity types, relationship types, extraction rules
    stats: WorkspaceStats     # Entity count, relationship count, community count, etc.
```

**Storage:** Each workspace gets:
- Its own Neo4j database (e.g., `sb_ml_research`, `sb_biology_notes`)
- Its own directory under `data/{workspace_id}/` for metadata, raw documents, and synthesis pages
- Its own entry in a central workspace registry (`data/workspaces.json` or SQLite)

**Neo4j isolation:** Use one database per workspace. Neo4j supports multi-database in both Community (4.x+) and Enterprise editions. The `database_` parameter on `execute_query()` and `session()` calls routes queries to the correct database. The GraphRAGStore's `graph_name` parameter (currently always `"neo4j"`) is the database name — we just need to set it per-workspace.

### 3.2 Data Directory Layout

```
study-buddy/
├── study_buddy.yaml                 # Global config (Neo4j connection, default LLM/embedding)
├── src/                              # Python backend (refactored)
│   ├── app.py                        # FastAPI app (thin — delegates to services)
│   ├── workspace.py                  # Workspace CRUD, registry, lifecycle
│   ├── services/
│   │   ├── ingestion.py              # DocumentIngestion (from current ingestion.py)
│   │   ├── graph.py                  # GraphRAG operations (extract, build, query)
│   │   ├── community.py              # Community detection + summarization
│   │   ├── query.py                  # Query engine (from GraphRAGQueryEngine)
│   │   └── synthesis.py              # Synthesis layer (NEW)
│   ├── core/
│   │   ├── extractor.py              # GraphRAGExtractor (from core_classes.py)
│   │   ├── store.py                  # GraphRAGStore (from core_classes.py)
│   │   └── config.py                 # Config (refactored — no singleton)
│   ├── mcp_server.py                 # MCP server (NEW)
│   ├── tui/
│   │   ├── app.py                    # Textual TUI app (NEW)
│   │   ├── screens/                  # TUI screens
│   │   └── widgets/                  # TUI custom widgets
│   └── models.py                     # Pydantic models (shared)
├── data/                             # Runtime data (gitignored)
│   ├── workspaces.json               # Workspace registry
│   └── {workspace_id}/               # Per-workspace data
│       ├── config.yaml               # Per-workspace config overrides
│       ├── schema.yaml               # Entity/relationship types, extraction rules
│       ├── raw/                      # Original documents (immutable)
│       ├── synthesis/                # Synthesis pages (markdown + frontmatter)
│       ├── index.md                  # Workspace content catalog
│       └── log.md                    # Workspace action log
└── cli/                              # Rust CLI (updated for multi-workspace)
```

### 3.3 Service Architecture

Instead of the current monolithic `app.py` (1281 lines), split into focused services:

```
FastAPI (thin HTTP layer)
    │
    ├── WorkspaceService      — create, list, get, delete workspaces
    ├── IngestionService      — document intake, parsing, extraction pipeline
    ├── GraphService          — entity/relationship CRUD, graph queries
    ├── CommunityService      — community detection, summaries, rebuild
    ├── QueryService          — natural language query via GraphRAG
    ├── SynthesisService      — create/update/list/query synthesis pages
    └── LintService           — health check, orphan detection, stats

MCP Server (agent interface)
    │
    └── Same services, exposed as MCP tools

TUI (human interface)
    │
    └── Same services, accessed via HTTP client
```

All services operate on a workspace context. The workspace ID is resolved from the request (HTTP header, MCP parameter, or TUI selection) and used to load workspace-specific state (Neo4j database, summaries, config overrides).

---

## 4. Implementation Phases

### Phase 0: Foundation Refactor — Multi-Workspace Support

**Goal:** Restructure the codebase to support multiple independent knowledge bases.

**Why first:** Everything else (MCP, TUI, synthesis) depends on workspace isolation. Can't build on a single-workspace foundation.

**Tasks:**

0.1 **Refactor config to support workspace-level overrides**
- Remove `Config` singleton. Make `WorkspaceConfig` loadable from `data/{workspace_id}/config.yaml`
- Global config provides defaults; workspace config overrides specific fields (LLM model, embedding model, extraction prompt, etc.)
- `WorkspaceConfig` resolves: workspace override → global config → hardcoded default
- Files: `src/core/config.py` (refactored)

0.2 **Create `Workspace` model and registry**
- `Workspace` dataclass with id, name, description, neo4j_database, created_at, updated_at
- `WorkspaceRegistry` that manages the workspace list in `data/workspaces.json`
- Methods: `create(name, description)`, `get(id)`, `list()`, `delete(id)`, `get_or_create_database(id)`
- Database creation: `CREATE DATABASE {id} IF NOT EXISTS` via Neo4j system session
- Files: `src/workspace.py` (new), `src/models.py` (new)

0.3 **Refactor `GraphRAGStore` to accept workspace context**
- Constructor takes `database` parameter (the workspace's Neo4j database name)
- All `_run_cypher()` calls pass `database_=self.graph_name` (already partially there)
- `build_communities()` uses `{workspace_id}_graph` as GDS projection name to avoid collisions
- Community summaries and entity info stored in workspace data dir, not flat `summaries/`
- Files: `src/core/store.py` (refactored from `core_classes.py`)

0.4 **Refactor `GraphRAGExtractor` into its own module**
- Move from `core_classes.py` to `core/extractor.py`
- Support per-workspace extraction prompts (loaded from workspace schema)
- The `parse_fn` lives alongside it
- Files: `src/core/extractor.py` (extracted)

0.5 **Refactor `app.py` into service modules**
- Split the 1281-line `app.py` into:
  - `src/services/ingestion.py` — document intake pipeline
  - `src/services/graph.py` — graph construction and entity operations
  - `src/services/community.py` — community detection and summary management
  - `src/services/query.py` — query engine
- `app.py` becomes a thin FastAPI router that delegates to services
- Each service method accepts a `workspace_id` parameter
- Files: `src/app.py` (rewritten thin), `src/services/*.py` (new)

0.6 **Workspace-aware API endpoints**
- All endpoints prefixed with `/kb/{workspace_id}/...`
- Keep legacy endpoints (without prefix) pointing to a "default" workspace for backward compat
- New endpoints:
  - `POST /kb` — create workspace
  - `GET /kb` — list workspaces
  - `GET /kb/{id}` — get workspace info
  - `DELETE /kb/{id}` — delete workspace (and database)
  - `GET /kb/{id}/stats` — workspace statistics
- Files: `src/app.py` (updated routes), `src/workspace.py`

0.7 **Workspace-aware data storage**
- Move from flat `summaries/` to `data/{workspace_id}/` directories
- Each workspace stores its own community summaries, entity info, raw docs, and metadata
- Version snapshots work the same way, just per-workspace
- Files: `src/services/community.py`, `src/workspace.py`

0.8 **Update Rust CLI for multi-workspace**
- Add `workspace` subcommand: `sb workspace create`, `sb workspace list`, `sb workspace use`
- All existing commands take optional `--workspace` flag (defaults to active workspace)
- Add workspace selection to TUI placeholder
- Files: `cli/src/commands/workspace.rs` (new), `cli/src/api/client.rs` (updated)

**Verification:**
- Create two workspaces, ingest different docs into each, verify they're isolated
- Query one workspace, confirm no cross-contamination
- Delete one workspace, confirm the other is unaffected

---

### Phase 1: MCP Server — Agent Tools

**Goal:** Make Study Buddy callable by any MCP-compatible agent (Hermes, Claude, GPT, etc.).

**Why next:** MCP integration is the primary agent interface. It's what makes the app "agent-interactable" as opposed to just "human-interactable via API." This is the core differentiator.

**Tasks:**

1.1 **Create MCP server module**
- Use `FastMCP` from `mcp` package (v1.27.0+)
- Server name: `"study-buddy"`
- Transport: `streamable-http` (can mount onto existing FastAPI app)
- Files: `src/mcp_server.py` (new)

1.2 **Implement MCP tools**
- `ingest_documents(workspace_id, directory, files)` — start ingestion
- `query_graph(workspace_id, query, similarity_top_k)` — query the knowledge graph
- `search_entities(workspace_id, query, limit)` — search entities
- `get_entity(workspace_id, name)` — get entity details
- `list_communities(workspace_id)` — list communities
- `get_community(workspace_id, community_id)` — get community details
- `list_workspaces()` — list all workspaces
- `create_workspace(name, description)` — create a workspace
- `delete_workspace(workspace_id)` — delete a workspace
- `get_workspace_stats(workspace_id)` — workspace statistics
- Each tool delegates to the corresponding service method
- Return structured Pydantic models (not just strings)
- Files: `src/mcp_server.py`

1.3 **Implement MCP resources**
- `kg://workspace/{workspace_id}/schema` — workspace schema/config
- `kg://workspace/{workspace_id}/stats` — workspace statistics
- `kg://workspace/{workspace_id}/communities` — community summaries overview
- These are read-only data that agents can discover and inspect
- Files: `src/mcp_server.py`

1.4 **Implement MCP prompts**
- `explore_workspace` — prompt template for exploring a knowledge base
- `compare_entities` — prompt template for comparing entities across communities
- Files: `src/mcp_server.py`

1.5 **Mount MCP server onto FastAPI**
- At startup, create MCP server and mount at `/mcp` path
- Both HTTP API and MCP share the same service layer
- Single process, single port
- Files: `src/app.py` (updated), `src/mcp_server.py`

1.6 **Progress reporting for long operations**
- MCP tools that trigger ingestion should report progress via `ctx.report_progress()`
- This lets agents show progress to users
- Files: `src/mcp_server.py`, `src/services/ingestion.py`

**Verification:**
- Start the server, connect with an MCP client (e.g., Hermes's native MCP client)
- Create a workspace, ingest documents, query entities — all via MCP tools
- Verify progress reporting works for ingestion

---

### Phase 2: TUI — Terminal User Interface

**Goal:** A terminal interface for browsing and querying knowledge graphs, replacing/supplementing the Rust CLI.

**Why here:** TUI requires the multi-workspace and service foundation from Phase 0. It's a human-facing interface that makes the system inspectable and debuggable.

**Tasks:**

2.1 **Create Textual app scaffold**
- `src/tui/app.py` — Main `StudyBuddyApp(App)` class
- Screens: `WorkspaceScreen`, `EntityScreen`, `CommunityScreen`, `QueryScreen`
- Shared `APIClient` that talks to the FastAPI server (reuse the Rust CLI's API models)
- Files: `src/tui/app.py`, `src/tui/screens/`, `src/tui/widgets/`

2.2 **Workspace selection screen**
- List all workspaces with stats (entity count, community count, last updated)
- Create/delete workspaces
- Select active workspace (sets context for all other screens)
- Files: `src/tui/screens/workspace.py`

2.3 **Entity browser screen**
- Search entities by name (substring match)
- Display entity details: name, label, description, communities, relationships
- Navigate to related entities and communities via keybindings
- Files: `src/tui/screens/entity.py`

2.4 **Community browser screen**
- List all communities with summaries and entity counts
- Drill into a community to see its entities and relationships
- View community summary in detail
- Files: `src/tui/screens/community.py`

2.5 **Query screen**
- Input field for natural language queries
- Display results with source entities and communities
- Show similarity scores and community context
- Files: `src/tui/screens/query.py`

2.6 **Graph visualization (text-based)**
- ASCII/rich representation of entity relationships within a community
- Use `Rich.Tree` for hierarchical display
- Files: `src/tui/widgets/graph_tree.py`

2.7 **Ingestion screen**
- Select a directory, preview files, start ingestion
- Show progress bar with entity/relationship counts
- Post-ingestion summary
- Files: `src/tui/screens/ingest.py`

2.8 **Update Rust CLI**
- Add `sb tui` command that launches the Python TUI (via `python -m study_buddy.tui`)
- Or: add `--tui` flag that shells out to the Python TUI
- Files: `cli/src/commands/tui.rs` (updated from placeholder)

**Verification:**
- Launch TUI, create a workspace, ingest documents, browse entities and communities, run queries
- Switch between workspaces, verify data isolation
- Test keyboard navigation and responsive layout

---

### Phase 3: Synthesis Layer — Compounding Knowledge

**Goal:** Add a wiki-like synthesis layer where knowledge accumulates across queries and ingestions.

**Why here:** This is the biggest conceptual upgrade from the current system. It transforms Study Buddy from a retrieval engine into a knowledge base that compounds over time. Requires stable multi-workspace support first.

**Tasks:**

3.1 **Synthesis data model**
- `SynthesisPage` dataclass: id, title, content (markdown), type (concept, comparison, timeline, analysis), tags, source_entities, source_communities, source_documents, created_at, updated_at
- Stored as markdown files with YAML frontmatter in `data/{workspace_id}/synthesis/`
- Compatible with Obsidian (wikilinks, frontmatter, Dataview)
- Files: `src/models.py`, `src/services/synthesis.py`

3.2 **Index and log per workspace**
- `data/{workspace_id}/index.md` — content catalog of all entities + synthesis pages
- `data/{workspace_id}/log.md` — append-only action log
- Updated automatically on every mutation (ingest, query, create/update synthesis)
- Files: `src/services/index.py`, `src/services/log.py`

3.3 **Synthesis CRUD API and MCP tools**
- `POST /kb/{id}/synthesis` — create a synthesis page
- `GET /kb/{id}/synthesis` — list synthesis pages
- `GET /kb/{id}/synthesis/{page_id}` — get a synthesis page
- `PUT /kb/{id}/synthesis/{page_id}` — update a synthesis page
- `DELETE /kb/{id}/synthesis/{page_id}` — delete a synthesis page
- MCP tools: `create_synthesis`, `list_syntheses`, `get_synthesis`, `update_synthesis`
- Files: `src/app.py` (routes), `src/mcp_server.py` (tools), `src/services/synthesis.py`

3.4 **Query filing**
- After a `query_graph` call, option to file the answer as a synthesis page
- Metadata links back to the entities and communities consulted
- MCP tool: `file_query_as_synthesis(workspace_id, query, answer, entities, communities)`
- Files: `src/services/synthesis.py`, `src/mcp_server.py`

3.5 **Ingestion synthesis updates**
- After ingesting new documents, detect which existing synthesis pages are affected
- Flag pages for review when new information contradicts or supplements existing content
- Don't auto-overwrite — flag for human/agent review
- Files: `src/services/synthesis.py`, `src/services/ingestion.py`

3.6 **Contradiction tracking**
- When new entity descriptions conflict with existing ones (same entity, different descriptions), record both with source citations and timestamps
- Add `contradictions` field to entity metadata
- Surface contradictions in MCP tools and TUI
- Files: `src/services/graph.py`, `src/models.py`

**Verification:**
- Create a workspace, ingest documents, file a query answer as synthesis
- Ingest more documents, verify that affected synthesis pages are flagged
- Create a second workspace, verify synthesis pages are isolated
- View synthesis pages in Obsidian (markdown + frontmatter compatibility)

---

### Phase 4: Advanced Features

**Goal:** Polish, reliability, and power-user features.

4.1 **Lint and health-check**
- `GET /kb/{id}/lint` endpoint and `lint_workspace` MCP tool
- Checks: orphan entities, disconnected components, stale communities, missing properties, contradiction flags
- Returns structured report with severity levels
- Files: `src/services/lint.py`, `src/mcp_server.py`

4.2 **Configurable extraction schema**
- Per-workspace entity types and relationship types defined in `schema.yaml`
- Generate extraction prompt dynamically from schema
- Support both open extraction (current default) and constrained extraction (schema-enforced)
- Files: `src/core/extractor.py`, `src/workspace.py`

4.3 **Incremental ingestion**
- Ingest new documents without rebuilding the entire graph
- Use `PropertyGraphIndex.insert_nodes()` for incremental inserts
- Rebuild communities only for affected subgraphs (or on-demand)
- Detect and skip already-ingested documents (file hash tracking)
- Files: `src/services/ingestion.py`, `src/services/community.py`

4.4 **Workspace import/export**
- Export a workspace as a portable archive (Neo4j dump + metadata + synthesis pages)
- Import a workspace from an archive (creates Neo4j database + restores files)
- Enables sharing knowledge bases between devices
- Files: `src/services/import_export.py`

4.5 **Observability and stats**
- `GET /kb/{id}/stats` — entity count, relationship count, community count, document count, last ingestion, etc.
- Dashboard-style stats in TUI
- Accumulation metrics: total synthesis pages, contradictions flagged, queries filed
- Files: `src/workspace.py`, `src/tui/screens/stats.py`

4.6 **Configurable community detection**
- Support different algorithms beyond Leiden (Louvain, Label Propagation)
- Per-workspace algorithm selection in config
- Adjustable parameters (max levels, tolerance, seed)
- Files: `src/services/community.py`

4.7 **Per-workspace embedding model selection**
- Allow different embedding models per workspace (currently global `Settings.embed_model`)
- Requires per-workspace vector stores or namespace isolation within a shared store
- Files: `src/core/config.py`, `src/services/graph.py`

---

## 5. Technical Decisions

### 5.1 Neo4j Multi-Database Strategy

**Decision:** One Neo4j database per workspace.

**Rationale:**
- Clean isolation — no risk of cross-contamination between knowledge bases
- Simpler Cypher queries (no `WHERE n.workspace_id = $workspace_id` on every query)
- Can drop a workspace by dropping its database
- Neo4j Community Edition (4.x+) supports multiple databases (with resource constraints)
- The project's `GraphRAGStore` already has a `database` parameter — just need to set it per-workspace

**Trade-off:** More memory usage per database. For local use, this is fine. For cloud deployment, Neo4j Enterprise has better multi-database support.

### 5.2 MCP vs REST API for Agent Interaction

**Decision:** Both. FastAPI REST for programmatic clients, MCP for agent clients.

**Rationale:**
- REST API is stable, well-documented, and works with any HTTP client
- MCP is the emerging standard for agent tool discovery — agents can discover Study Buddy's capabilities dynamically
- Both share the same service layer, so no logic duplication
- MCP server mounts onto the FastAPI app at `/mcp`, single process

### 5.3 TUI: Python (Textual) vs Rust

**Decision:** Python/Textual for the TUI.

**Rationale:**
- The backend is Python. Reusing models, API client logic, and data structures saves significant development time.
- Textual v8.2.3 is mature, supports reactive state, async workers, rich widgets, and CSS styling.
- The Rust CLI already has its own API client. The TUI is a separate interface — it doesn't need to share code with the Rust CLI.
- The Rust CLI stays for scripting and quick commands. The TUI is for interactive browsing.

### 5.4 Synthesis Storage: Markdown Files

**Decision:** Synthesis pages stored as markdown files with YAML frontmatter.

**Rationale:**
- Compatible with Obsidian, VS Code, and any text editor
- Human-readable without any tool
- Wikilinks (`[[entity-name]]`) cross-reference to graph entities
- YAML frontmatter enables structured metadata (tags, dates, sources)
- Git-trackable for version history
- No additional database dependency

### 5.5 Community Summaries: Move to Neo4j

**Decision:** Store community summaries as Neo4j nodes instead of JSON files.

**Rationale:**
- Current JSON sidecar files are fragile (can become stale, no transaction safety, hard to query)
- Neo4j can store `Community` nodes with `summary` property — queryable, persistent, transactional
- `entity_info` can be stored as entity properties (`community_ids` list on `__Entity__` nodes)
- Version snapshots can still be exported as JSON for backup, but the source of truth is Neo4j

### 5.6 Service Layer Pattern

**Decision:** Service classes that accept `workspace_id` as first parameter, return structured result objects.

**Rationale:**
- Services are the single source of truth for business logic
- Both FastAPI routes and MCP tools call into the same service methods
- TUI calls service methods directly (no HTTP overhead when running in same process)
- Easy to test in isolation
- Workspace context passed explicitly, never from global state

---

## 6. Dependency Changes

### Add
- `mcp>=1.27.0` — MCP server SDK (FastMCP)
- `textual>=0.86.0` — TUI framework (v8.x line)
- `rich>=13.0.0` — Rich text rendering (Textual dependency, also used for TUI output)

### Modify
- `neo4j>=5.28.0` — Keep at 5.x (don't upgrade to 6.x yet, breaking changes)
- `llama-index-*` — Keep at current versions, upgrade incrementally after refactoring
- `pydantic>=2.0` — Already using; models shared across API/MCP/TUI

### Remove (eventually)
- `src/main.py` — Legacy standalone script, replaced by API + services
- `summaries/` flat directory — Replaced by per-workspace data directories

---

## 7. What NOT to Change

These aspects of the current system work well and should be preserved:

- **LlamaIndex PropertyGraphIndex** — the graph construction and retrieval API is solid
- **GraphRAGExtractor** — custom extraction with configurable prompt and parse function
- **Leiden community detection** via Neo4j GDS — works correctly with Cypher-based API
- **DocumentIngestion routing** — file extension → parser mapping is extensible and correct
- **Versioned community summaries** — the snapshot pattern is good, just needs per-workspace scoping
- **Rust CLI** — keep for scripting; it wraps the HTTP API and doesn't need to be rewritten

---

## 8. File Change Map

### New Files
| File | Phase | Purpose |
|------|-------|---------|
| `src/workspace.py` | 0 | Workspace model, registry, lifecycle |
| `src/models.py` | 0 | Shared Pydantic models |
| `src/core/extractor.py` | 0 | GraphRAGExtractor (extracted from core_classes.py) |
| `src/core/store.py` | 0 | GraphRAGStore (extracted, workspace-aware) |
| `src/core/config.py` | 0 | Config (refactored, no singleton) |
| `src/services/ingestion.py` | 0 | DocumentIngestion service |
| `src/services/graph.py` | 0 | Graph operations service |
| `src/services/community.py` | 0 | Community detection service |
| `src/services/query.py` | 0 | Query engine service |
| `src/mcp_server.py` | 1 | MCP server with tools/resources/prompts |
| `src/tui/app.py` | 2 | Textual TUI main app |
| `src/tui/screens/workspace.py` | 2 | Workspace selection screen |
| `src/tui/screens/entity.py` | 2 | Entity browser screen |
| `src/tui/screens/community.py` | 2 | Community browser screen |
| `src/tui/screens/query.py` | 2 | Query screen |
| `src/tui/screens/ingest.py` | 2 | Ingestion screen |
| `src/tui/widgets/graph_tree.py` | 2 | Graph tree widget |
| `src/services/synthesis.py` | 3 | Synthesis pages service |
| `src/services/index.py` | 3 | Workspace index service |
| `src/services/log.py` | 3 | Workspace log service |
| `src/services/lint.py` | 4 | Lint/health-check service |
| `src/services/import_export.py` | 4 | Import/export service |

### Modified Files
| File | Phase | Changes |
|------|-------|---------|
| `src/app.py` | 0,1 | Restructure into thin router, mount MCP server |
| `src/core_classes.py` | 0 | Split into core/extractor.py + core/store.py, then delete |
| `src/ingestion.py` | 0 | Move to services/ingestion.py, add workspace param |
| `src/config.py` | 0 | Move to core/config.py, remove singleton, add WorkspaceConfig |
| `cli/src/api/models.rs` | 0,8 | Add workspace-related request/response models |
| `cli/src/api/client.rs` | 0,8 | Add workspace-related API calls |
| `cli/src/commands/` | 0,8 | Add workspace subcommand |
| `study_buddy.yaml` | 0 | Add workspace defaults section |

### Deleted Files
| File | Phase | Reason |
|------|-------|--------|
| `src/main.py` | 0 | Replaced by API + services |
| `src/core_classes.py` | 0 | Split into extractor.py + store.py |
| `AGENT_KNOWLEDGE_BASE_VISION.md` | 0 | Replaced by this document |

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Neo4j Community edition limits multi-database performance | Medium | Medium | Test with 5-10 workspaces; if constrained, add option for single-DB with workspace labels |
| LlamaIndex Settings singleton conflicts with per-workspace config | High | High | Remove Settings dependency; instantiate LLM/embedding models per-workspace, pass explicitly |
| Refactoring 1281-line app.py introduces regressions | High | Medium | Incremental refactoring with tests; keep old endpoints working during transition |
| MCP server slows down FastAPI | Low | Low | MCP runs on separate path (`/mcp`); no interference with REST endpoints |
| Synthesis pages grow unbounded | Medium | Medium | Implement page size limits in schema; auto-split large pages; lint detects oversized pages |
| GDS projection names collide between workspaces | Medium | High | Use workspace-specific projection names (`{workspace_id}_graph`) instead of database name |

---

## 10. Success Metrics

| Metric | Target | Phase |
|--------|--------|-------|
| Multiple workspaces can coexist without cross-contamination | Verify with integration test | 0 |
| MCP tools callable from Hermes agent | End-to-end test | 1 |
| TUI can create workspace, browse entities, run query | Manual test | 2 |
| Synthesis pages viewable in Obsidian | Verify markdown + frontmatter compatibility | 3 |
| Lint detects orphan entities | Integration test | 4 |
| Incremental ingestion doesn't require full rebuild | Performance test | 4 |