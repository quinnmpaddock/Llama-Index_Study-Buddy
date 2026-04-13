# Study Buddy — Optimization Scope

> **Purpose:** Reference document for agentic optimization of the study-buddy codebase. Each section identifies specific issues with file/line references, explains the risk or impact, and suggests concrete fixes.
>
> **Context:** This application is designed to run locally without internet exposure. Security concerns related to remote attackers (API auth, CORS, rate limiting) are deprioritized. The focus is on **code correctness, reliability, performance, and developer experience**. Only genuine code-safety issues (like path traversal that could corrupt data) are highlighted.

---

## Table of Contents

1. [Critical Code Issues](#1-critical-code-issues)
2. [Performance Optimizations](#2-performance-optimizations)
3. [Code Quality & Architecture](#3-code-quality--architecture)
4. [Reliability & Resilience](#4-reliability--resilience)
5. [Missing Features & Gaps](#5-missing-features--gaps)
6. [Prioritized Recommendations](#6-prioritized-recommendations)

---

## 1. Critical Code Issues

### 1.1 `asyncio.run()` Inside Async Context Will Crash — **P0 CRITICAL**

**File:** `src/core_classes.py:89`

```python
def __call__(self, nodes, show_progress=False, **kwargs):
    return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))
```

`asyncio.run()` creates a **new event loop**. When called from within an already-running async context (like during FastAPI's background ingestion via `run_full_ingestion`), this raises:

```
RuntimeError: This event loop is already running
```

This is a hard crash — ingestion will fail. The `__call__` method is the sync entry point that LlamaIndex's `PropertyGraphIndex` uses internally.

**Fix:** Detect the running loop and schedule the coroutine properly:

```python
def __call__(self, nodes, show_progress=False, **kwargs):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop — schedule and wait
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, self.acall(nodes, show_progress=show_progress, **kwargs))
            return future.result()
    else:
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))
```

Or better, ensure ingestion always calls the async path directly via `await extractor.acall(...)`.

### 1.2 Race Condition: Engine State Swap During Queries — **P0 CRITICAL**

**File:** `src/app.py:796–808`

After ingestion completes, the engine is hot-swapped across four separate attribute assignments:

```python
app.state.engine = GraphRAGQueryEngine(...)
app.state.community_summaries = {str(k): v for k, v in ...}
app.state.entity_info = index.property_graph_store.entity_info
app.state.summaries_loaded = True
```

A concurrent query could read partially-updated state — e.g., new `community_summaries` but old `entity_info` — producing incorrect results or a crash.

**Fix:** Build the complete state first, then swap atomically:

```python
# Build all new state first
new_engine = GraphRAGQueryEngine(...)
new_summaries = {str(k): v for k, v in index.property_graph_store.community_summary.items()}
new_entity_info = index.property_graph_store.entity_info

# Swap all at once (still not perfectly atomic in asyncio, but minimizes window)
app.state.engine = new_engine
app.state.community_summaries = new_summaries
app.state.entity_info = new_entity_info
app.state.summaries_loaded = True
```

Even better, wrap the swap + all reads in an `asyncio.Lock`.

### 1.3 Bare `except Exception` Silences `build_communities` Failures — **P0 HIGH**

**File:** `src/core_classes.py:263`

```python
except Exception as e:
    print(f"DEBUG: build_communities failed: {type(e).__name__}: {e}")
```

This catches **all** exceptions from the Leiden algorithm + graph projection, then silently continues to `_collect_community_info()` which will likely fail on corrupted/missing graph state. The result is either empty communities or a confusing downstream error.

**Fix:** Log the error properly and propagate or handle gracefully:

```python
except Exception as e:
    logger.error(f"build_communities failed: {type(e).__name__}: {e}")
    # Don't collect community info if the graph projection failed
    return
```

### 1.4 Hardcoded Neo4j Password in Committed Config — **P1 MEDIUM**

**File:** `study_buddy.yaml:63`

```yaml
password: "neo4j2026"
```

While this is a local app, the default password is committed to git history. Anyone cloning the repo gets it. Even locally, it's a hygiene issue — if the machine is shared or the repo goes public, the credential is exposed.

**Fix:**
1. Replace the default in `study_buddy.yaml` with a placeholder: `password: "CHANGE_ME"`
2. Already works: `config.py:52` reads `NEO4J_PASSWORD` env var
3. Add `study_buddy.yaml` to `.gitignore` and provide `study_buddy.yaml.example` instead
4. Add a startup warning if the default password is detected

### 1.5 Only `ValueError` Caught in LLM Extraction — **P1 HIGH**

**File:** `src/core_classes.py:96–108`

```python
try:
    llm_response = await self.llm.apredict(...)
except ValueError as e:
    print(f"DEBUG ValueError: {e}")
    entities = []
    entities_relationship = []
```

Network errors, rate limits, timeouts, and API errors from LlamaIndex all inherit from different exception classes. Only `ValueError` is caught — everything else propagates as an unhandled exception that crashes the entire extraction loop, losing all entities from that chunk.

**Fix:** Catch broader categories with proper logging:

```python
except (ValueError, asyncio.TimeoutError, Exception) as e:
    logger.warning(f"LLM extraction failed for node: {type(e).__name__}: {e}")
    entities = []
    entities_relationship = []
```

Or better, use `tenacity` for retry with backoff on transient errors.

---

## 2. Performance Optimizations

### 2.1 Synchronous LLM Calls Block the Event Loop — **P1 HIGH**

**File:** `src/core_classes.py:207, 428, 468`

Three synchronous `self.llm.chat()` calls exist:

| Method | Line | Call |
|---|---|---|
| `GraphRAGStore.generate_community_summary` | 207 | `self.llm.chat(messages)` |
| `GraphRAGQueryEngine.generate_answer_from_summary` | 428 | `self.llm.chat(messages)` |
| `GraphRAGQueryEngine.aggregate_answers` | 468 | `self.llm.chat(messages)` |

These block the FastAPI event loop during queries. Async variants already exist (`achat`, `agenerate_answer_from_summary`, `aaggregate_answers`) but the sync versions are called from `custom_query` and `build_communities`.

**Fix:** Ensure all query paths use the async variants. Wrap any remaining sync calls in `asyncio.to_thread()`.

### 2.2 Sequential Cypher Queries in `build_communities` — **P2 MEDIUM**

**File:** `src/core_classes.py:235–270`

Three separate Cypher queries execute sequentially (project, leiden, drop). Each is a separate network round-trip to Neo4j.

**Fix:** Wrap the project + leiden in a single transaction:

```python
def build_communities(self):
    with self._driver.session() as session:
        session.run("MATCH (n:__Entity__)-[r]->(m:__Entity__) ...")
        session.run("CALL gds.leiden.write(...) ...")
    # Drop must be separate (after commit)
    self._run_cypher(f"CALL gds.graph.drop('{self.graph_name}', false) YIELD graphName")
```

### 2.3 Community Entity Lookup Is O(n×m) Per Request — **P2 MEDIUM**

**File:** `src/app.py:426–443, 501, 518`

Every `/communities` request rebuilds the community→entity mapping from scratch. Every `/communities/{id}` request linear-scans all entities.

```python
# app.py:501 — linear scan on every request
entity_count = sum(1 for communities in entity_info.values() if id in communities)

# app.py:518 — another linear scan
entities = [name for name, communities in entity_info.items() if id in communities]
```

**Fix:** Pre-compute reverse indexes on startup and after ingestion, store in `app.state`:

```python
community_to_entities: Dict[int, List[str]] = defaultdict(list)
for entity_name, communities in entity_info.items():
    for comm_id in communities:
        community_to_entities[comm_id].append(entity_name)
app.state.community_to_entities = community_to_entities
```

### 2.4 Embedding Model Startup Blocking — **P2 MEDIUM**

**File:** `src/app.py:199–202`

Loading the HuggingFace embedding model is synchronous and can take 10–30 seconds. The entire `lifespan` runs sequentially, so Neo4j connection waits for model loading.

**Fix:** Run model loading and summary loading concurrently:

```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor() as executor:
    embed_future = executor.submit(HuggingFaceEmbedding, model_name=config.embedding.model)
    community_summaries, entity_info = load_summaries_and_entity_info()
    Settings.embed_model = embed_future.result()
```

### 2.5 No Connection Pooling Config for Neo4j — **P2 LOW**

**File:** `src/core_classes.py:158–182`

`GraphRAGStore` inherits from `Neo4jPropertyGraphStore` but doesn't configure connection pool size, max connections, or connection lifecycle. Under load (multiple concurrent ingestion + queries), the default driver settings may exhaust connections.

**Fix:** Pass connection pool parameters through config:

```python
# In config.py Neo4jConfig
self.max_connection_pool_size = config_dict.get("max_connection_pool_size", 100)
self.connection_acquisition_timeout = config_dict.get("connection_acquisition_timeout", 60)
```

### 2.6 Module-Level `load_dotenv()` Runs on Import — **P3 LOW**

**File:** `src/app.py:23`, `src/main.py:6`

Every import of these modules triggers `load_dotenv()`, which reads `.env` from disk. If the file is large or on a slow filesystem, this adds overhead to every test import.

**Fix:** Move to a single initialization point, or use `override=False` to avoid re-loading already-set env vars:

```python
load_dotenv(override=False)
```

---

## 3. Code Quality & Architecture

### 3.1 `app.py` Is 1268 Lines — Needs Modularization — **P1 HIGH**

`src/app.py` contains API routes, business logic, ingestion pipeline, file I/O, JSON parsing, and background task management all in one file. This makes it difficult to navigate, test, and maintain.

**Proposed structure:**

```
src/
├── app.py              # FastAPI app factory + lifespan (~200 lines)
├── routes/
│   ├── query.py         # /query endpoint
│   ├── entities.py      # /entities endpoints
│   ├── communities.py   # /communities endpoints
│   ├── ingest.py         # /ingest endpoints + background task
│   └── summaries.py     # /summaries endpoints
├── services/
│   ├── ingestion.py     # DocumentIngestion class + pipeline logic
│   ├── snapshot.py      # load_summaries_and_entity_info, find_most_recent_snapshot
│   └── kg_store.py     # GraphRAGStore, GraphRAGExtractor, GraphRAGQueryEngine
├── models/
│   ├── config.py        # Config classes (existing)
│   └── schemas.py       # Pydantic request/response models
└── utils/
    ├── parsing.py       # extract_json, parse_fn
    └── text.py          # _make_summary_preview
```

This doesn't change functionality — just splits the file for maintainability.

### 3.2 Duplicated Code — **P1 MEDIUM**

| Duplication | Locations | Fix |
|---|---|---|
| `parse_fn` / `extract_json` | `app.py:555–615` vs `main.py:55–86` | Extract to `utils/parsing.py` |
| `supported_extensions` set | `app.py:870` vs `app.py:1001` | Define once in `config.py` |
| `SUMMARIES_DIR` constant | `app.py:32` vs `app.py:1060` | Define once as module constant |
| Sync/async prompt duplication | `core_classes.py:413–450` vs `432–494` | Extract prompt, share logic via async wrapper |
| `datetime` import inside functions | `app.py:48,741,1094,1228` | Top-level import |
| Entity search logic | `app.py:374–391` | Single function in a service module |

### 3.3 `print()` Instead of `logger` in Core Classes — **P1 MEDIUM**

**File:** `src/core_classes.py:106, 210, 264, 309, 317`

Five `print()` statements mixed with proper `logger` calls in `app.py`. When running as a service (uvicorn), these bypass the logging system and may not be captured in log output.

**Fix:** Replace all `print()` with `logger.info()` or `logger.debug()`, with appropriate log levels. During ingestion especially, these should be `logger.debug()` calls for the verbose extraction output.

### 3.4 Inconsistent Error Handling — **P2 MEDIUM**

| Issue | Location | Fix |
|---|---|---|
| Bare `except Exception` in `build_communities` | `core_classes.py:263` | Log + return/raise (see §1.3) |
| Missing API key returns 500 | `app.py:939` | Return 503 "Service not configured" instead |
| Engine reload failure leaves partial state | `app.py:811–813` | Rollback to old state if reload fails |
| No error handling for corrupt JSON in `/summaries/{version}` | `app.py:1153` | Wrap in try/except, return 500 with detail |
| Only `ValueError` caught in `_aextract` | `core_classes.py:105` | Broaden exception handling (see §1.5) |

### 3.5 Typo: `enitites` → `entities` — **P3 LOW**

**File:** `src/core_classes.py:379`

```python
enitites = set()
```

Used consistently within the method but is a readability issue. Rename all occurrences.

### 3.6 Two Singleton Patterns for Config — **P3 LOW**

**Files:** `src/config.py:90` (class-level `_instance`) and `config.py:210` (module-level `_config`)

Both `Config._instance`/`Config.get()` and `get_config()`/`_config` implement separate singleton patterns. This is confusing and could lead to divergent config state.

**Fix:** Remove `Config._instance`, `Config.get()`, and `Config.reset()`. Use only `get_config()` / `_config` global.

### 3.7 Cypher String Interpolation — **P2 MEDIUM**

**File:** `src/core_classes.py:235–268`

While not a security risk in a local app (graph_name comes from config), it's still a code quality issue. Using f-strings for Cypher queries is fragile — any future change that makes `graph_name` user-controllable would create a real injection risk.

**Fix:** Add validation in `GraphRAGStore.__init__`:

```python
import re
if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', self.graph_name):
    raise ValueError(f"Invalid graph name: {self.graph_name}")
```

---

## 4. Reliability & Resilience

### 4.1 Ingestion Status Dict Never Evicted — **P2 MEDIUM**

**File:** `src/app.py:551`

```python
ingestion_status: Dict[str, dict] = {}
```

Completed ingestion task entries persist for the lifetime of the process. On a long-running local server, this grows without bound.

**Fix:** Add TTL-based cleanup after reading status:

```python
import time

CLEANUP_THRESHOLD = 3600  # 1 hour

@app.get("/ingest/status/{task_id}")
async def get_ingestion_status(task_id: str):
    # Periodic cleanup
    now = time.time()
    stale = [tid for tid, status in ingestion_status.items()
             if status.get("completed_at", 0) < now - CLEANUP_THRESHOLD]
    for tid in stale:
        del ingestion_status[tid]

    if task_id not in ingestion_status:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return ingestion_status[task_id]
```

### 4.2 No Atomic Write for `current.json` — **P2 MEDIUM**

**File:** `src/app.py:772–774`

```python
with open(current_path, "w", encoding="utf-8") as f:
    json.dump(current_info, f, indent=4)
```

If the process crashes mid-write, `current.json` is corrupted, and the next startup will fail to load the current version pointer.

**Fix:** Use write-then-rename (atomic on POSIX):

```python
import tempfile
fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".json")
with os.fdopen(fd, "w") as f:
    json.dump(current_info, f, indent=4)
os.replace(tmp_path, current_path)  # Atomic
```

### 4.3 No Retry Logic for LLM Calls — **P2 MEDIUM**

**File:** `src/core_classes.py:96–108`

LLM API calls can fail transiently (rate limits, network errors, timeouts). Currently only `ValueError` is caught, and failures silently drop entities.

**Fix:** Add `tenacity` retry with exponential backoff:

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=30),
    retry=retry_if_exception_type((TimeoutError, ConnectionError)),
)
async def _aextract(self, node: BaseNode) -> BaseNode:
    # ... existing extraction logic ...
```

### 4.4 Startup Failure Crashes the App — **P2 MEDIUM**

**File:** `src/app.py:265–267`

```python
except Exception as e:
    logger.error(f"Failed to initialize engine: {str(e)}")
    raise e
```

If Neo4j is unreachable at startup, the entire app fails to start. For a local tool, this is painful — the user has to manually start Neo4j first, then restart the app.

**Fix:** Add degraded startup mode:

```python
except Exception as e:
    logger.warning(f"Engine init failed: {e}. Starting in degraded mode.")
    app.state.summaries_loaded = False
    app.state.engine = None
    app.state.community_summaries = {}
    app.state.entity_info = {}
```

The existing 503 checks at `/query` (`app.py:298`) will handle this. Add a helpful startup message telling the user to start Neo4j and restart the app.

### 4.5 `_sanitize_metadata` Modifies Dict In-Place — **P3 LOW**

**File:** `src/ingestion.py:71`

The function modifies the input dict in-place AND returns it — a side effect that's easy to miss.

**Fix:** Copy first:

```python
def _sanitize_metadata(self, metadata: dict) -> dict:
    sanitized = metadata.copy()
    # ... modify sanitized instead of metadata ...
    return sanitized
```

### 4.6 No Graceful Shutdown — **P3 LOW**

**File:** `src/app.py:269–270`

The lifespan context manager has `yield` with no cleanup after it. Active ingestion tasks are orphaned, and the Neo4j driver connection is not explicitly closed.

**Fix:**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ... startup ...
    yield
    logger.info("Shutting down...")
    if hasattr(app.state, 'engine') and app.state.engine:
        app.state.engine.graph_store._driver.close()
```

---

## 5. Missing Features & Gaps

### 5.1 No Health Check for Neo4j Dependency — **P1 HIGH**

The `/` endpoint only shows that FastAPI is running. There's no way to verify Neo4j connectivity without attempting a query.

**Fix:** Add a `/health` endpoint:

```python
@app.get("/health")
async def health_check():
    checks = {"api": "ok"}
    if hasattr(app.state, 'engine') and app.state.engine:
        try:
            app.state.engine.graph_store._driver.verify_connectivity()
            checks["neo4j"] = "ok"
        except Exception as e:
            checks["neo4j"] = f"error: {e}"
    else:
        checks["neo4j"] = "not_initialized"
    return checks
```

### 5.2 No Test Coverage — **P1 HIGH**

Zero Python unit tests exist for the backend. The only test-like file is `cli/mock_api.py` — a mock server for CLI integration testing, not unit tests.

**Minimum test suite needed:**

| Test File | What It Tests |
|---|---|
| `tests/test_config.py` | Config loading, env var overrides, YAML parsing, missing file fallback |
| `tests/test_parsing.py` | `extract_json()` and `parse_fn()` edge cases (malformed JSON, empty responses, nested objects) |
| `tests/test_app_routes.py` | FastAPI TestClient tests for all endpoints (with mocked Neo4j) |
| `tests/test_ingestion.py` | `DocumentIngestion` routing, `_sanitize_metadata` edge cases |
| `tests/test_snapshot.py` | Snapshot loading, version detection, cleanup logic |

### 5.3 No Structured Logging — **P2 MEDIUM**

Logging uses `logging.basicConfig(level=...)` with plain text. For a local tool that outputs to terminal, this is acceptable, but makes it hard to grep logs when debugging issues.

**Fix:** Minimal improvement — add timestamps and module names:

```python
logging.basicConfig(
    level=config.server.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
```

### 5.4 `main.py` Is Legacy Code — **P3 LOW**

**File:** `src/main.py` is described as a "legacy artifact" in the README. It duplicates `app.py`'s `parse_fn` with an older version and has interactive `input()` calls that don't belong in a service.

**Fix:** Either remove it or mark it clearly as deprecated:

```python
"""
DEPRECATED: Use the FastAPI backend (app.py) or CLI instead.
This module is retained for historical reference only.
"""
import warnings
warnings.warn("main.py is deprecated. Use the API backend or CLI.", DeprecationWarning, stacklevel=2)
```

### 5.5 TUI Mode Not Implemented — **P3 LOW**

**File:** `cli/src/main.rs:222–225`

```rust
Commands::Tui => {
    print_error("TUI mode not yet implemented. Use 'query' command for now.");
    std::process::exit(1);
}
```

The TUI command exists in the CLI but immediately exits with an error. This should either be implemented or removed from the command list to avoid confusion.

### 5.6 No Config Validation Beyond API Key — **P3 LOW**

**File:** `src/config.py:174–181`

Only the API key is validated. Neo4j URL format, server port range, model name, and timeout bounds are accepted without validation.

**Fix:** Add basic validation:

```python
class Neo4jConfig:
    def __init__(self, config_dict: dict):
        self.url = config_dict.get("url", "bolt://localhost:7687")
        if not self.url.startswith(("bolt://", "neo4j://")):
            raise ConfigError(f"Invalid Neo4j URL scheme: {self.url}")
        self.timeout = config_dict.get("timeout", 120.0)
        if self.timeout <= 0:
            raise ConfigError("Neo4j timeout must be positive")
```

### 5.7 LLM Prompt Injection — Local Context Only — **P3 LOW**

**File:** `src/core_classes.py:415–419`

User queries are directly interpolated into LLM prompts. Since this is a local tool, there's no attacker to inject prompts. However, poorly formatted or maliciously pasted input could degrade LLM output quality.

**Fix (optional, light touch):** Add a max length to queries:

```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="...")
    similarity_top_k: int = Field(default=20, ge=1, le=50)
```

---

## 6. Prioritized Recommendations

### P0 — Critical Code Issues (Fix First)

| # | Issue | Impact | Effort |
|---|---|---|---|
| 1 | Fix `asyncio.run()` in `GraphRAGExtractor.__call__` | **Runtime crash** during background ingestion — complete failure of the ingestion pipeline | Medium |
| 2 | Atomic state swap during engine reload | Stale/partial state during concurrent queries after ingestion | Small |
| 3 | Fix bare `except Exception` in `build_communities` | Silent failures lead to corrupted/empty community data downstream | Small |
| 4 | Broaden exception handling in `_aextract` beyond `ValueError` | Transient LLM errors crash extraction loop, losing all entities from that chunk | Small |

### P1 — High Priority (Fix Soon)

| # | Issue | Impact | Effort |
|---|---|---|---|
| 5 | Add Neo4j health check endpoint (`/health`) | No way to verify Neo4j connectivity without attempting a query | Small |
| 6 | Replace `print()` with `logger` in core_classes.py | Debug output lost when running as uvicorn service | Tiny |
| 7 | Add basic test suite (config, parsing, routes) | Zero test coverage means regressions go undetected | Medium |
| 8 | Refactor `app.py` into modular structure | 1268-line god module is hard to navigate and maintain | Large |
| 9 | Deduplicate `parse_fn`, `supported_extensions`, `SUMMARIES_DIR`, etc. | Bug surface area, maintenance burden | Small |
| 10 | Move Neo4j password out of committed config file | Hygiene concern — credential in git history | Small |

### P2 — Medium Priority (Plan for Next Iteration)

| # | Issue | Impact | Effort |
|---|---|---|---|
| 11 | Pre-compute community→entity reverse index | O(n×m) per request for community endpoints | Small |
| 12 | Fix inconsistent error codes (500 vs 503 for config issues) | Client gets wrong error information | Tiny |
| 13 | Add TTL-based cleanup for `ingestion_status` dict | Memory leak on long-running servers | Small |
| 14 | Use atomic write-then-rename for `current.json` | Data corruption on crash | Small |
| 15 | Add degraded startup mode (no Neo4j) | App crash on startup if Neo4j is down — painful UX | Medium |
| 16 | Add retry logic for LLM calls | Transient network/API failures silently drop entities | Medium |
| 17 | Cypher string interpolation validation | Code hygiene, future-proofing | Tiny |
| 18 | Parallelize startup (model loading + summary loading) | 10–30s faster startup | Small |
| 19 | Convert sync LLM calls to async in `custom_query` path | Blocking event loop during queries | Medium |

### P3 — Low Priority (Nice to Have)

| # | Issue | Impact | Effort |
|---|---|---|---|
| 20 | Fix `enitites` typo to `entities` | Readability | Tiny |
| 21 | Remove or deprecate `main.py` legacy code | Developer confusion | Tiny |
| 22 | Consolidate config singleton pattern | Minor confusion | Tiny |
| 23 | Add Pydantic config validation | Runtime misconfiguration caught late | Small |
| 24 | Add graceful shutdown in lifespan context | Orphaned Neo4j connections on exit | Small |
| 25 | Improve logging format (timestamps, module names) | Hard to grep logs when debugging | Tiny |
| 26 | Add `QueryRequest` max_length validation | Poorly formatted queries degrade LLM output | Tiny |
| 27 | Migrate `_run_cypher` to async Neo4j driver | Threadpool blocking during ingestion | Large |
| 28 | Implement TUI mode in CLI (or remove command) | Incomplete feature confuses users | Large |
| 29 | Cache entity/community endpoint responses | Minor CPU savings on repeated reads | Small |
| 30 | Fix `_sanitize_metadata` to copy before modifying | Side-effect bug risk | Tiny |