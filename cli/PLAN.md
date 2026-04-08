# Rust CLI Tool for Study Buddy GraphRAG

## Overview

A Rust-based TUI/CLI frontend that interfaces with the existing FastAPI GraphRAG backend. The CLI will provide interactive and scriptable access to knowledge graph operations.

## Architecture

```
cli/
├── Cargo.toml
├── src/
│   ├── main.rs              # Entry point, CLI arg parsing
│   ├── lib.rs                # Library exports
│   ├── api/
│   │   ├── mod.rs
│   │   ├── client.rs        # HTTP client for FastAPI backend
│   │   └── models.rs        # Request/response structs (serde)
│   ├── commands/
│   │   ├── mod.rs
│   │   ├── query.rs         # General graph querying
│   │   ├── ingest.rs        # Document ingestion
│   │   ├── search.rs        # Node surfing/search
│   │   └── community.rs     # Community evaluation summary
│   ├── tui/
│   │   ├── mod.rs
│   │   ├── app.rs          # TUI application state
│   │   ├── ui.rs           # UI rendering (ratatui)
│   │   └── event.rs        # Event handling
│   ├── config/
│   │   ├── mod.rs
│   │   └── settings.rs     # Configuration management
│   └── utils/
│       ├── mod.rs
│       ├── output.rs       # Output formatting (table, json, yaml)
│       └── error.rs        # Error types
└── tests/
    └── integration_tests.rs
```

## Dependencies

```toml
[dependencies]
# Async runtime
tokio = { version = "1", features = ["full"] }

# HTTP client
reqwest = { version = "0.12", features = ["json", "rustls-tls"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# CLI argument parsing
clap = { version = "4", features = ["derive", "color"] }

# TUI framework
ratatui = "0.28"
crossterm = "0.28"

# Configuration
config = "0.14"
directories = "5"

# Error handling
thiserror = "1"
anyhow = "1"

# Output formatting
tabled = "0.16"

# Optional: for file upload
mime = "0.3"
mime_guess = "2"
```

## CLI Commands

### 1. General Graph Querying (`query`)

```bash
# Interactive query
study-buddy query "What are the main topics in the knowledge graph?"

# With options
study-buddy query "Explain fuzzing techniques" --top-k 30 --format json

# Pipe mode (reads from stdin)
echo "What is SMT solving?" | study-buddy query --stdin
```

**API Interaction:**
- `POST /query` with `{ query: string, similarity_top_k: int }`
- Returns `{ answer, communities_consulted, entities_found }`

### 2. Document Ingestion (`ingest`)

```bash
# Ingest a single file
study-buddy ingest ./input/document.pdf

# Ingest a directory
study-buddy ingest ./input/ --recursive

# With options
study-buddy ingest ./data/ --format csv --wait-for-completion
```

**API Interaction:** 
- Requires new FastAPI endpoint: `POST /ingest`
- Request: multipart/form-data file upload
- Response: `{ job_id: string, status: string, nodes_created: int }`

### 3. Node Surfing/Search (`search`)

```bash
# Search for entities
study-buddy search "fuzzing" --type entity

# Search for relationships
study-buddy search "test*" --type relation --limit 20

# Get entity details including relationships
study-buddy search "Miller et al." --depth 2

# List all entities in a community
study-buddy search --community 12 --list-entities
```

**API Interaction:** 
- Requires new FastAPI endpoints:
- `GET /entities?search={query}&limit={n}` - Entity search
- `GET /entities/{name}` - Entity details with relationships
- `GET /communities/{id}/entities` - Entities in community

### 4. Full Graph Community Evaluation (`community`)

```bash
# List all communities
study-buddy community list

# Get community summary
study-buddy community show 12

# Get full community summary (all communities)
study-buddy community summary --format markdown

# Evaluate community coverage
study-buddy community stats

# Export community data
study-buddy community export --output communities.json
```

**API Interaction:**
- `GET /communities` - List all communities
- `GET /communities/{id}` - Single community details
- `GET /communities/{id}/summary` - Community summary text

### 5. Interactive TUI Mode (`tui`)

```bash
study-buddy tui
```

**TUI Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Study Buddy - GraphRAG CLI                           [? Help] │
├─────────────────────────────────────────────────────────────────┤
│ Query: [____________________] [Enter to submit]                  │
├────────────────────────┬────────────────────────────────────────┤
│ Entities Found         │ Answer                                   │
│ ┌────────────────────┐ │ ┌──────────────────────────────────────┐│
│ │ • Fuzzing          │ │ │ Based on the knowledge graph...      ││
│ │ • SMT Solvers      │ │ │                                      ││
│ │ • Knowledge Graphs │ │ │ Miller et al. developed the first    ││
│ │ • Testing          │ │ │ fuzzer... [test_data0.pdf]           ││
│ └────────────────────┘ │ └──────────────────────────────────────┘│
├────────────────────────┴────────────────────────────────────────┤
│ Communities: [18, 19, 12]  |  Similarity: 0.85  |  Status: OK   │
└─────────────────────────────────────────────────────────────────┘
```

**Features:**
- Query input with history (arrow keys)
- Results panel with scrollable answer
- Entity list sidebar
- Community badges
- Keyboard shortcuts (Ctrl+C to quit, Tab to switch panels)

## API Extensions Required

The CLI requires additional FastAPI endpoints not currently implemented:

### New Endpoints for `src/app.py`

```python
# === Ingestion ===
class IngestRequest(BaseModel):
    file_path: str
    recursive: bool = False

class IngestResponse(BaseModel):
    job_id: str
    status: str
    nodes_created: Optional[int] = None

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest documents from specified path."""
    ...

# === Entity Search ===
class EntitySearchResponse(BaseModel):
    entities: List[str]
    total: int

@app.get("/entities", response_model=EntitySearchResponse)
async def search_entities(
    query: Optional[str] = None,
    limit: int = 50
):
    """Search for entities in the knowledge graph."""
    ...

class EntityDetail(BaseModel):
    name: str
    communities: List[int]
    relationships: List[Dict[str, str]]

@app.get("/entities/{name}", response_model=EntityDetail)
async def get_entity(name: str):
    """Get entity details with relationships."""
    ...

# === Communities ===
class CommunityList(BaseModel):
    communities: List[int]
    total: int

@app.get("/communities", response_model=CommunityList)
async def list_communities():
    """List all community IDs."""
    ...

class CommunityDetail(BaseModel):
    id: int
    summary: str
    entities: List[str]
    entity_count: int

@app.get("/communities/{id}", response_model=CommunityDetail)
async def get_community(id: int):
    """Get community details."""
    ...

@app.get("/communities/{id}/summary")
async def get_community_summary(id: int):
    """Get raw community summary text."""
    ...
```

## Configuration

**File:** `~/.config/study-buddy/config.toml`

```toml
[api]
base_url = "http://localhost:8000"
timeout_seconds = 30

[display]
default_format = "table"  # table, json, yaml
color = true
pager = "less"

[tui]
history_size = 100
theme = "dark"
```

## Implementation Phases

### Phase 1: Core CLI Foundation
1. Project setup with Cargo.toml
2. API client module with reqwest
3. Basic `query` command implementation
4. Configuration management
5. Output formatting (table/json/yaml)

### Phase 2: Search & Community Commands
1. Entity search functionality
2. Community listing and details
3. Add required FastAPI endpoints
4. Connection pooling for performance

### Phase 3: Document Ingestion
1. File upload handling
2. Ingestion endpoint implementation
3. Progress tracking
4. Batch ingestion support

### Phase 4: Interactive TUI
1. ratatui application setup
2. Query input with history
3. Results display panel
4. Entity navigation
5. Keyboard shortcuts

### Phase 5: Polish & Testing
1. Integration tests
2. Error handling improvements
3. Documentation
4. Performance optimization

## Key Design Decisions

### 1. API-First Design
- All operations go through FastAPI backend
- CLI is a thin client that orchestrates API calls
- Enables future mobile/web frontends

### 2. Async Architecture
- Use tokio for async I/O
- Non-blocking HTTP requests
- Concurrent API calls where beneficial

### 3. Modular Command Structure
- Each command in separate module
- Shared API client instance
- Consistent error handling

### 4. Multiple Output Formats
- Table (default for human readability)
- JSON (for scripting/piping)
- YAML (for configuration export)

### 5. Optional TUI
- TUI mode for interactive exploration
- CLI mode for scripting/automation
- Same underlying operations

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum StudyBuddyError {
    #[error("API connection failed: {0}")]
    ApiConnection(#[from] reqwest::Error),
    
    #[error("API error: {status} - {message}")]
    ApiError { status: u16, message: String },
    
    #[error("Configuration error: {0}")]
    Config(#[from] config::ConfigError),
    
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("Ingestion failed: {0}")]
    IngestionFailed(String),
}
```

## Usage Examples

```bash
# Basic query
study-buddy query "What is knowledge graph synthesis?"

# JSON output for scripting
study-buddy query "Explain fuzzing" -f json | jq '.answer'

# Search for an entity
study-buddy search "OWL" --type entity

# Get full details on an entity
study-buddy search "Miller et al." --depth 2 -f yaml

# List all communities
study-buddy community list

# Get community 12 summary
study-buddy community show 12

# Export all community summaries
study-buddy community export -o summaries.md

# Ingest new documents
study-buddy ingest ./papers/ --recursive

# Launch interactive mode
study-buddy tui
```

## Testing Strategy

1. **Unit Tests:** API model serialization, config parsing
2. **Integration Tests:** Mock HTTP server for API testing
3. **E2E Tests:** Run against live FastAPI backend (optional)
4. **TUI Tests:** Snapshot testing for UI rendering

## Future Enhancements

- WebSocket support for real-time updates
- Graph visualization (terminal-based with svg	fi)
- Export to common formats (Neo4j dump, GraphML)
- Query history persistence
- Saved queries/favorites
- Bulk operations (multi-file ingest, batch queries)