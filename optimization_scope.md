# Study Buddy — Optimization Scope

> **Purpose:** Reference document for agentic optimization of the study-buddy codebase. Each section identifies specific issues with file/line references, explains the risk or impact, and suggests concrete fixes. Priority levels (P0–P3) guide implementation order.

---

## Table of Contents

1. [Security Vulnerabilities](#1-security-vulnerabilities)
2. [Performance Optimizations](#2-performance-optimizations)
3. [Code Quality & Architecture](#3-code-quality--architecture)
4. [Reliability & Resilience](#4-reliability--resilience)
5. [Scalability](#5-scalability)
6. [Missing Features & Gaps](#6-missing-features--gaps)
7. [Prioritized Recommendations](#7-prioritized-recommendations)

---

## 1. Security Vulnerabilities

### 1.1 Path Traversal in `/ingest/preview` — **P0 CRITICAL**

**File:** `src/app.py:982–1030`

The `/ingest/preview` endpoint accepts an arbitrary `directory` query parameter with no path restriction:

```python
@app.get("/ingest/preview")
async def preview_ingest(directory: str = Query(...)):
    dir_path = Path(directory)  # RAW user input → filesystem path
```

An attacker can pass `/ingest/preview?directory=/etc` or any path to enumerate files and sizes on the server. This is both a path traversal and information disclosure vulnerability.

**Fix:** Restrict `directory` to a whitelist of allowed base paths (e.g., only subdirectories of a configured `INPUT_DIR`). Validate with `resolve()` and `relative_to()`:

```python
ALLOWED_DIRS = [Path(config.ingestion.input_dir).resolve()]
resolved = Path(directory).resolve()
if not any(resolved == d or resolved.is_relative_to(d) for d in ALLOWED_DIRS):
    raise HTTPException(status_code=403, detail="Directory not allowed")
```

### 1.2 Unrestricted Directory Access in `/ingest` POST — **P0 CRITICAL**

**File:** `src/app.py:858–866`

While individual file paths within the request are checked for traversal (lines 888–903), the base `directory` parameter itself is only checked for existence and `is_dir()`. An attacker can ingest files from any directory on the server (e.g., `/etc`, `/home`).

**Fix:** Apply the same whitelist restriction as above to the `directory` parameter in the `/ingest` endpoint.

### 1.3 No API Authentication — **P0 CRITICAL**

**File:** `src/app.py:274–279`

The entire FastAPI app has zero authentication. All endpoints — including destructive ones like `DELETE /summaries` and expensive ones like `POST /ingest` — are publicly accessible.

- `POST /ingest` → triggers expensive LLM API calls (costs money)
- `POST /query` → exfiltrates all knowledge graph data
- `DELETE /summaries` → destroys stored data
- `GET /entities`, `/communities` → data exposure

**Fix:** Add API key authentication middleware:

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != os.environ.get("STUDY_BUDDY_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")

# Apply to all routes:
app = FastAPI(dependencies=[Depends(verify_api_key)])
```

### 1.4 Cypher Injection via String Interpolation — **P0 HIGH**

**File:** `src/core_classes.py:235–268`

The `build_communities` method interpolates `self.graph_name` directly into Cypher queries:

```python
self._run_cypher(
    f"MATCH (n:__Entity__)-[r]->(m:__Entity__)
     Return gds.graph.project('{self.graph_name}', n, m, ...)"
)
```

And line 268–269:

```python
self._run_cypher(
    f"CALL gds.graph.drop('{self.graph_name}', false) YIELD graphName"
)
```

While `graph_name` currently comes from config, this pattern is unsafe. The `_run_cypher` method accepts `params` but these interpolations bypass parameterized queries.

**Fix:** Use parameterized Cypher where possible. For GDS procedures that don't support parameters, validate `graph_name` with a strict regex:

```python
import re
if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', self.graph_name):
    raise ValueError(f"Invalid graph name: {self.graph_name}")
```

### 1.5 Hardcoded Neo4j Password — **P1 HIGH**

**File:** `study_buddy.yaml:63`

```yaml
password: "neo4j2026"
```

The default config ships with a plaintext password committed to the repo. While `config.py:52` supports `NEO4J_PASSWORD` env var override, the default leaks via git history.

**Fix:**
1. Replace the default password with a placeholder: `password: "${NEO4J_PASSWORD}"`
2. Add validation in `Neo4jConfig.__init__` that rejects default passwords in production
3. Add `study_buddy.yaml` to `.gitignore` (or provide `study_buddy.yaml.example` instead)

### 1.6 No CORS Configuration — **P1 MEDIUM**

**File:** `src/app.py:274–279`

No `CORSMiddleware` is configured on the FastAPI app. This means:
- Browser-based clients are blocked by same-origin policy
- If fronted by a proxy, there's no CORS policy enforcement

**Fix:**

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 1.7 No Rate Limiting — **P1 MEDIUM**

No rate limiting on any endpoint. Particularly dangerous for `POST /ingest` (costs money per call) and `POST /query` (triggers multiple LLM calls).

**Fix:** Add `slowapi` or custom middleware:

```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query_graph(request: Request, ...):
```

### 1.8 LLM Prompt Injection — **P1 MEDIUM**

**File:** `src/core_classes.py:415–419`

```python
prompt = (
    f"Given the community summary: {community_summary}, "
    f"how would you answer the following query? Query: {query}\n\n"
)
```

User input (`query`) is directly interpolated into LLM prompts with no sanitization. Malicious queries can contain instructions that manipulate LLM output.

**Fix:** Add input sanitization and length limits:

```python
# In QueryRequest model
query: str = Field(..., min_length=1, max_length=1000, description="...")

# Sanitize at query entry
sanitized_query = re.sub(r'[^\w\s?.,!;:\-()]', '', query.strip())[:1000]
```

### 1.9 Summary Version Parameter Path Traversal — **P2 LOW**

**File:** `src/app.py:1136–1160`

```python
@app.get("/summaries/{version}")
async def get_summary_version(version: str):
    summary_file = os.path.join(summaries_dir, f"community_summaries_{version}.json")
```

The `version` parameter is user-supplied and not validated. While the `community_summaries_` prefix format limits exploitation, explicit validation is missing.

**Fix:** Validate `version` format:

```python
import re
if not re.match(r'^\d{4}-\d{2}-\d{2}_\d{6}$', version):
    raise HTTPException(status_code=400, detail="Invalid version format")
```

---

## 2. Performance Optimizations

### 2.1 Synchronous LLM Calls Block the Event Loop — **P1 HIGH**

**File:** `src/core_classes.py:207, 428, 468`

Three synchronous `self.llm.chat()` calls exist in `GraphRAGStore.generate_community_summary`, `GraphRAGQueryEngine.generate_answer_from_summary`, and `GraphRAGQueryEngine.aggregate_answers`. These block the FastAPI event loop during queries.

The async variants (`achat`, `aaggregate_answers`) exist but the sync versions are still used in the `custom_query` method and `build_communities`.

**Fix:** Replace all sync LLM calls with async equivalents. Ensure `build_communities` is called in a threadpool if sync is unavoidable.

### 2.2 `asyncio.run()` Inside Transform May Cause Event Loop Conflicts — **P1 HIGH**

**File:** `src/core_classes.py:89`

```python
def __call__(self, nodes, show_progress=False, **kwargs):
    return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))
```

`asyncio.run()` creates a new event loop. When called from within an already-running async context (like during background ingestion), this raises `RuntimeError: This event loop is already running`.

**Fix:** Use `asyncio.get_event_loop().run_until_complete()` or, better, schedule the coroutine properly within the existing event loop using `asyncio.ensure_future()` or `loop.create_task()`.

### 2.3 Sequential Cypher Queries in `build_communities` — **P2 MEDIUM**

**File:** `src/core_classes.py:235–270`

Three separate Cypher queries execute sequentially (project, leiden, drop). These could be batched into a single transaction for reduced round-trip overhead.

**Fix:**

```python
def build_communities(self):
    with self._driver.session() as session:
        session.run("MATCH (n:__Entity__)-[r]->(m:__Entity__) ...")
        session.run("CALL gds.leiden.write(...) ...")
    # Drop is separate (must be after write commits)
    self._run_cypher(f"CALL gds.graph.drop('{self.graph_name}', false) YIELD graphName")
```

### 2.4 Community Entity Lookup Is O(n×m) Per Request — **P2 MEDIUM**

**File:** `src/app.py:426–443, 501, 518`

The `/communities` endpoint rebuilds the community→entity mapping on every request by iterating over all entities. Similarly, `/communities/{id}` counts entities with a linear scan, and `/communities/{id}/entities` also linear scans.

```python
# app.py:501 — linear scan
entity_count = sum(1 for communities in entity_info.values() if id in communities)

# app.py:518 — another linear scan
entities = [name for name, communities in entity_info.items() if id in communities]
```

**Fix:** Pre-compute reverse indexes on ingestion and cache them:

```python
# Build reverse index once
community_to_entities: Dict[int, List[str]] = defaultdict(list)
for entity_name, communities in entity_info.items():
    for comm_id in communities:
        community_to_entities[comm_id].append(entity_name)
```

Store this in `app.state` and update after ingestion.

### 2.5 Embedding Model Startup Blocking — **P2 MEDIUM**

**File:** `src/app.py:199–202`

Loading the HuggingFace embedding model is synchronous and can take 10–30 seconds. The entire lifespan runs sequentially.

**Fix:** Load the embedding model in a background thread while continuing other initialization:

```python
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor() as executor:
    embed_future = executor.submit(HuggingFaceEmbedding, model_name=config.embedding.model)
    # Load summaries from disk concurrently
    community_summaries, entity_info = load_summaries_and_entity_info()
    Settings.embed_model = embed_future.result()
```

### 2.6 No Caching for Entity/Community Endpoints — **P3 LOW**

All GET endpoints (`/entities`, `/communities`, `/communities/{id}`) compute results from in-memory dicts on every request. For read-heavy workloads, even simple caching with invalidation-on-ingestion would reduce CPU overhead.

**Fix:** Add `functools.lru_cache` or a lightweight cache with TTL and invalidation after ingestion completes.

---

## 3. Code Quality & Architecture

### 3.1 `app.py` Is 1268 Lines — Needs Modularization — **P2 MEDIUM**

`src/app.py` contains API routes, business logic, ingestion pipeline, file I/O, JSON parsing, and background task management all in one file.

**Proposed structure:**

```
src/
├── app.py              # FastAPI app factory + lifespan
├── routes/
│   ├── query.py         # /query endpoints
│   ├── entities.py      # /entities endpoints
│   ├── communities.py    # /communities endpoints
│   ├── ingest.py         # /ingest endpoints
│   └── summaries.py      # /summaries endpoints
├── services/
│   ├── ingestion.py      # DocumentIngestion + ingestion pipeline
│   ├── snapshot.py       # load_summaries_and_entity_info, find_most_recent_snapshot
│   └── kg_store.py      # GraphRAGStore, GraphRAGExtractor, etc.
├── models/
│   ├── config.py         # Config classes
│   └── schemas.py        # Pydantic request/response models
└── utils/
    ├── parsing.py        # extract_json, parse_fn
    └── text.py           # _make_summary_preview
```

### 3.2 Duplicated Code — **P2 MEDIUM**

| Duplication | Locations | Fix |
|---|---|---|
| `parse_fn` / `extract_json` | `app.py:555–615` vs `main.py:55–86` | Extract to `utils/parsing.py` |
| Entity search logic | `app.py:374–391` | Single function in `services/entities.py` |
| `supported_extensions` set | `app.py:870` vs `app.py:1001` | Define once in `config.py` or constants |
| `SUMMARIES_DIR` constant | `app.py:32` vs `app.py:1060` | Define once as module constant |
| Sync/async prompt generation | `core_classes.py:413–450` vs `432–494` | Extract prompt template, share logic |
| `datetime` imports | `app.py:48,741,1094,1228` | Top-level import |

### 3.3 `print()` Instead of `logger` in Core Classes — **P2 MEDIUM**

**File:** `src/core_classes.py:106, 210, 264, 309, 317`

Five `print()` statements mixed with proper `logger` calls in `app.py`. In production, these bypass the logging system and may not be captured.

**Fix:** Replace all `print()` with `logger.info()` or `logger.debug()`.

### 3.4 Inconsistent Error Handling — **P2 MEDIUM**

| Issue | Location | Fix |
|---|---|---|
| Bare `except Exception` silences `build_communities` failures | `core_classes.py:263` | Log and re-raise or handle gracefully |
| Missing API key returns 500 (should be 401/503) | `app.py:939` | Return 401 Unauthorized |
| Engine reload failure leaves partial state | `app.py:811–813` | Rollback old state if reload fails |
| No error handling for corrupt JSON in `/summaries/{version}` | `app.py:1153` | Wrap in try/except, return 400/500 |
| Only `ValueError` caught in `_aextract` | `core_classes.py:105` | Catch broader exception types |

### 3.5 Typo: `enitites` → `entities` — **P3 LOW**

**File:** `src/core_classes.py:379`

```python
enitites = set()
```

Functionally harmless but reduces readability. Rename all occurrences in the method.

### 3.6 Two Singleton Patterns for Config — **P3 LOW**

**Files:** `src/config.py:90` (class-level `_instance`) and `config.py:210` (module-level `_config`)

Both `Config._instance` and `get_config()` implement separate singleton patterns. This is confusing and could lead to two instances.

**Fix:** Remove `Config._instance`/`Config.get()`/`Config.reset()` and rely solely on `get_config()` / `_config` global.

---

## 4. Reliability & Resilience

### 4.1 Race Condition: Engine State Swap During Queries — **P1 HIGH**

**File:** `src/app.py:796–808`

After ingestion completes, the engine is hot-swapped:

```python
app.state.engine = GraphRAGQueryEngine(...)
app.state.community_summaries = {str(k): v for k, v in ...}
app.state.entity_info = index.property_graph_store.entity_info
app.state.summaries_loaded = True
```

A concurrent query could read partially-updated state (e.g., new `community_summaries` but old `entity_info`).

**Fix:** Use atomic swap — build the complete state dict first, then assign all attributes at once, or use a lock:

```python
new_state = {
    "engine": GraphRAGQueryEngine(...),
    "community_summaries": {str(k): v for k, v in ...},
    "entity_info": ...,
    "summaries_loaded": True,
}
for key, value in new_state.items():
    setattr(app.state, key, value)
```

Even better, use an `asyncio.Lock` around state reads/writes.

### 4.2 Ingestion Status Dict Never Evicted — **P2 MEDIUM**

**File:** `src/app.py:551`

```python
ingestion_status: Dict[str, Dict] = {}
```

Completed ingestion task entries persist for the lifetime of the process. Long-running servers will accumulate stale entries.

**Fix:** Add TTL-based cleanup:

```python
import time

# In run_full_ingestion, after completion:
ingestion_status[task_id]["completed_at"] = time.time()

# Periodic cleanup (or on access):
CLEANUP_THRESHOLD = 3600  # 1 hour
for tid, status in list(ingestion_status.items()):
    if status.get("completed_at", 0) < time.time() - CLEANUP_THRESHOLD:
        del ingestion_status[tid]
```

### 4.3 No Atomic Write for `current.json` — **P2 MEDIUM**

**File:** `src/app.py:772–774`

```python
with open(current_path, "w", encoding="utf-8") as f:
    json.dump(current_info, f, indent=4)
```

If the process crashes mid-write, `current.json` is corrupted. Use write-then-rename:

```python
import tempfile
fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".json")
with os.fdopen(fd, "w") as f:
    json.dump(current_info, f, indent=4)
os.replace(tmp_path, current_path)  # Atomic on POSIX
```

### 4.4 No Retry Logic for LLM Calls — **P2 MEDIUM**

**File:** `src/core_classes.py:96–108`

```python
try:
    llm_response = await self.llm.apredict(...)
except ValueError as e:
    print(f"DEBUG ValueError: {e}")
    entities = []
    entities_relationship = []
```

Only `ValueError` is caught. Network errors, rate limits (429), timeouts, and other LLM API failures propagate as unhandled exceptions, silently dropping entities.

**Fix:** Add retry with exponential backoff for transient failures:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _aextract_with_retry(self, node):
    try:
        return await self._aextract(node)
    except (ValueError, RateLimitError) as e:
        logger.warning(f"LLM extraction failed for node: {e}")
        return node
```

### 4.5 Startup Failure Crashes the App — **P2 MEDIUM**

**File:** `src/app.py:265–267`

```python
except Exception as e:
    logger.error(f"Failed to initialize engine: {str(e)}")
    raise e
```

If Neo4j is unreachable at startup, the entire app fails. No fallback mode.

**Fix:** Add a degraded startup mode that serves cached summaries without Neo4j:

```python
except Exception as e:
    logger.warning(f"Engine init failed: {e}. Starting in degraded mode.")
    app.state.summaries_loaded = False
    app.state.engine = None
```

And return 503 for query endpoints when in degraded mode (which already exists at `app.py:298`).

### 4.6 `_sanitize_metadata` Modifies Dict In-Place — **P3 LOW**

**File:** `src/ingestion.py:71`

```python
def _sanitize_metadata(self, metadata: dict) -> dict:
```

This modifies the input dict in-place (side effect) AND returns it. Should either copy first or document the in-place behavior.

**Fix:**

```python
def _sanitize_metadata(self, metadata: dict) -> dict:
    sanitized = metadata.copy()
    # ... modify sanitized instead of metadata
    return sanitized
```

---

## 5. Scalability

### 5.1 `app.state` Dictionaries Grow Unboundedly — **P2 MEDIUM**

**File:** `src/app.py:252–256`

`app.state.community_summaries` and `app.state.entity_info` are loaded fully into memory. For knowledge graphs with millions of entities, this becomes prohibitive.

**Fix:** For large graphs:
1. Page community summaries on demand
2. Use a reverse-index data structure that's more memory-efficient
3. Consider storing entity_info in Redis or an in-process LRU cache

### 5.2 Horizontal Scaling Blocked by In-Process State — **P3 LOW**

The current architecture stores all state in `app.state` (in-process dict). Running multiple FastAPI workers (via `uvicorn --workers`) would result in inconsistent state across workers.

**Fix options:**
- Use Redis for shared state (summaries, entity_info, ingestion_status)
- Use a database for ingestion_status tracking
- For single-worker deployments (current), document this limitation

### 5.3 Synchronous Neo4j Driver Blocks Thread — **P3 LOW**

**File:** `src/core_classes.py:213–220`

`_run_cypher` uses the synchronous Neo4j driver, which blocks whichever thread it runs in. During ingestion (a background task), this holds a threadpool worker for the entire duration.

**Fix:** Migrate to `neo4j` async driver, or run `_run_cypher` calls in `asyncio.to_thread()`.

---

## 6. Missing Features & Gaps

### 6.1 No API Authentication — **P0 CRITICAL**
(See Section 1.3)

### 6.2 No Health Check for Neo4j Dependency — **P1 HIGH**

The `/` endpoint only checks that FastAPI is running. There's no check that Neo4j is reachable.

**Fix:** Add a `/health` endpoint that checks Neo4j connectivity:

```python
@app.get("/health")
async def health_check():
    checks = {"api": "ok"}
    try:
        app.state.engine.graph_store._driver.verify_connectivity()
        checks["neo4j"] = "ok"
    except Exception as e:
        checks["neo4j"] = f"error: {e}"
    return checks
```

### 6.3 No Structured Logging — **P2 MEDIUM**

Logging uses `logging.basicConfig(level=...)` with plain text. For production deployments, structured JSON logging is essential for searchability and monitoring.

**Fix:**

```python
import structlog
structlog.configure(processors=[structlog.processors.JSONRenderer()])
```

Or at minimum, configure Python's `logging` with JSON formatter.

### 6.4 No Metrics / Observability — **P2 MEDIUM**

No metrics are exposed (request counts, latency histograms, LLM call counts/costs, ingestion durations).

**Fix:** Add Prometheus metrics:

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

### 6.5 No Test Coverage — **P1 HIGH**

The only test file found is `cli/mock_api.py` (a mock server for CLI testing, not unit tests). There are zero Python unit tests for the backend.

**Minimum test coverage needed:**
- `test_config.py` — Config loading, env var overrides, YAML parsing
- `test_app_routes.py` — FastAPI TestClient tests for all endpoints
- `test_parsing.py` — `extract_json()` and `parse_fn()` edge cases
- `test_ingestion.py` — DocumentIngestion routing and metadata sanitization
- `test_snapshot.py` — Snapshot loading, version detection, cleanup

### 6.6 No Graceful Shutdown — **P3 LOW**

The `lifespan` context manager has a `yield` with no cleanup after it. Active ingestion tasks are not awaited or cancelled on shutdown.

**Fix:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... startup ...
    yield
    # Graceful shutdown
    logger.info("Shutting down...")
    # Cancel any running background tasks
    # Clean up Neo4j connections
    app.state.engine.graph_store._driver.close()
```

### 6.7 No Config Validation Beyond API Key — **P3 LOW**

**File:** `src/config.py:174–181`

Only the API key is validated. Neo4j URL format, server port range, model name format, and timeout bounds are not validated.

**Fix:** Add Pydantic-style validation to config classes or use `pydantic-settings`:

```python
from pydantic import field_validator

class Neo4jConfig(BaseModel):
    url: str
    @field_validator('url')
    def validate_url(cls, v):
        if not v.startswith(('bolt://', 'neo4j://')):
            raise ValueError('Neo4j URL must start with bolt:// or neo4j://')
        return v
```

### 6.8 TUI Mode Not Implemented — **P3 LOW**

**File:** `cli/src/main.rs:222–225`

```rust
Commands::Tui => {
    print_error("TUI mode not yet implemented. Use 'query' command for now.");
    std::process::exit(1);
}
```

The TUI command is defined but not implemented.

### 6.9 `main.py` Is Legacy Code — **P3 LOW**

**File:** `src/main.py` is described as a "legacy artifact" in the README. It duplicates `app.py`'s `parse_fn` and has interactive `input()` calls. It should be removed or clearly marked as deprecated.

---

## 7. Prioritized Recommendations

### P0 — Critical / Security (Fix Immediately)

| # | Issue | Impact | Effort |
|---|---|---|---|
| 1 | Add API key authentication to all endpoints | Unauthorized access, cost abuse | Small |
| 2 | Restrict `/ingest/preview` and `/ingest` directory to whitelist | Server file enumeration, data exfiltration | Small |
| 3 | Validate `/summaries/{version}` version parameter format | Path traversal risk | Tiny |
| 4 | Add CORS middleware with configurable origins | Browser access blocked | Tiny |

### P1 — High Priority (Fix Soon)

| # | Issue | Impact | Effort |
|---|---|---|---|
| 5 | Fix `asyncio.run()` in `GraphRAGExtractor.__call__` | Runtime crash during background ingestion | Medium |
| 6 | Add atomic state swap during engine reload | Stale/partial state during queries | Small |
| 7 | Add Neo4j health check endpoint | No way to detect dependency failure | Small |
| 8 | Replace `print()` with `logger` in core_classes.py | Lost debugging data in production | Tiny |
| 9 | Add LLM call retry with exponential backoff | Silent entity loss on transient failures | Medium |
| 10 | Add rate limiting on `/ingest` and `/query` | Cost abuse via repeated LLM calls | Small |
| 11 | Add basic test suite (config, routes, parsing) | Zero test coverage, regressions undetected | Medium |
| 12 | Move Neo4j password out of committed config file | Credential leak via git history | Small |

### P2 — Medium Priority (Plan for Next Iteration)

| # | Issue | Impact | Effort |
|---|---|---|---|
| 13 | Refactor `app.py` into modular structure | 1268-line god module | Large |
| 14 | Deduplicate `parse_fn`, `supported_extensions`, `SUMMARIES_DIR` | Maintenance burden, bug surface | Small |
| 15 | Pre-compute community→entity reverse index | O(n×m) per request for community endpoints | Small |
| 16 | Fix inconsistent error codes (500 vs 401/503) | Incorrect client behavior | Tiny |
| 17 | Add TTL-based cleanup for `ingestion_status` | Memory leak on long-running servers | Small |
| 18 | Use atomic write-then-rename for `current.json` | Data corruption on crash | Small |
| 19 | Add graceful shutdown in lifespan context | Orphaned connections, dropped tasks | Small |
| 20 | Add degraded startup mode (no Neo4j) | App crash on startup if Neo4j is down | Medium |
| 21 | Add structured JSON logging | Poor observability in production | Small |
| 22 | Add Prometheus metrics | No monitoring capability | Small |
| 23 | Sanitize Cypher query parameters (graph_name) | Injection risk if config is user-controllable | Small |

### P3 — Low Priority (Nice to Have)

| # | Issue | Impact | Effort |
|---|---|---|---|
| 24 | Fix `enitites` typo to `entities` | Readability | Tiny |
| 25 | Remove `main.py` legacy code | Developer confusion | Tiny |
| 26 | Consolidate config singleton pattern | Confusion | Tiny |
| 27 | Add Pydantic config validation | Runtime misconfiguration | Small |
| 28 | Migrate `_run_cypher` to async Neo4j driver | Threadpool blocking | Large |
| 29 | Move `app.state` to Redis for multi-worker scaling | Horizontal scaling blocked | Large |
| 30 | Implement TUI mode in CLI CLI | Incomplete feature | Large |
| 31 | Add caching for read-heavy endpoints | Minor CPU overhead | Small |
| 32 | Implement graceful shutdown for background tasks | Orphaned ingestion | Medium |