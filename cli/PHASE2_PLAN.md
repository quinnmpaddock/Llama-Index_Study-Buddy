# Phase 2 Implementation Plan: Search & Community Commands

## Overview

Extend both the FastAPI backend and Rust CLI to support entity/concept search and community exploration.

## Part A: FastAPI Backend Extensions

### 1. Entity Search Endpoints

**`GET /entities`** - Search entities by name
- Query params: `q` (search term), `limit` (default50)
- Returns: list of matching entity names with their community IDs
- Use fuzzy/prefix matching on `entity_info.json`

**`GET /entities/{name}`** - Get entitydetails
- Returns: entity name, list of community IDs it belongs to
- Future: include relationships extracted from Neo4j

### 2. Community Endpoints

**`GET /communities`** - List all communities
- Returns: list of community IDs with entity counts

**`GET /communities/{id}`** - Get community details
- Returns: community ID, summary text, list of entities in that community

**`GET /communities/{id}/entities`** - Entities in community
- Returns: list of entity names belonging to the community

## Part B: Rust CLI Commands

### 1. `search` Command

```bash
# Search for entities matching a term
sb search "knowledge"                # fuzzy search entities
sb search "SMT" --type entity       # explicit entity search
sb search "fuzzing" -l 20           # limit to20 results
sb search "graphs" -f json          # JSON output
```

Output (table format):
```
ENTITY                    COMMUNITIES
Knowledge Graphs          10, 12
Knowledge Graphs Synthesis2
...
```

### 2. `community` Command

```bash
# List all communities
sb community list

# Show community summary
sb community show12

# Show community with entities
sb community show12 --entities

# Export all summaries
sb community export -o summaries.md
```

Output (table format):
```
ID    ENTITIES    SUMMARY PREVIEW
10    15          SMT solvers can operate on knowledge graphs...
12    23          Knowledge graphs are structured representations...
```

## Part C: Data Flow

```
entity_info.json structure:
{
    "ENTITY_NAME": [community_id, community_id, ...],
    ...
}

community_summaries.json structure:
{
    "10": "Summary text for community 10...",
    "12": "Summary text for community 12...",
    ...
}
```

## Implementation Order

1. **Backend: Entity endpoints** - Simple read from `entity_info.json`
2. **Backend: Community endpoints** - Read from both JSON files
3. **CLI: search command** - Call entity endpoint, format output
4. **CLI: community command** - Call community endpoints, format output
5. **Mock API** - Add endpoints to `mock_api.py` for testing
6. **Testing** - Build and verify all commands work

## Files to Modify/Create

### Backend (src/app.py)
- Add `EntityInfo` model
- Add `CommunitySummary` model
- Add `/entities` endpoints
- Add `/communities` endpoints

### CLI (cli/src/)
- `commands/search.rs` - New file
- `commands/community.rs` - New file
- `commands/mod.rs` - Export new commands
- `main.rs` - Add new subcommands
- `api/client.rs` - Add API methods
- `api/models.rs` - Add response structs

### Mock API (cli/mock_api.py)
- Add `/entities` endpoint mock
- Add `/communities` endpoint mock

## Future Considerations

- Neo4j integration for relationship data
- Document ingestion CLI control
- Incremental community creation