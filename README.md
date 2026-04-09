# Study Buddy (GraphRAG)

Study Buddy is a Retrieval-Augmented Generation (RAG) system that leverages **Knowledge Graphs** to provide context-aware answers to complex queries. By combining LlamaIndex and Neo4j, it extracts entities and relationships from unstructured text to build a structured graph, then uses community detection to enable global reasoning across your entire document corpus.

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Document Ingestion                         │
│         PDF, DOCX, PPTX, HTML, XLSX, MD, CSV, TXT, JSON             │
│                    (via CLI or API endpoint)                         │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Text Chunking & Processing                      │
│                    SentenceSplitter (1024 token chunks)              │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Knowledge Graph Extraction                        │
│         LLM extracts entities, relationships, descriptions          │
│              (GraphRAGExtractor with custom prompts)                 │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Neo4j Graph Database                          │
│           Stores entities as nodes, relationships as edges          │
│                 Leiden algorithm detects communities                 │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Query Processing                              │
│   1. Vector similarity search finds relevant entities                │
│   2. Entity-to-community lookup retrieves context                    │
│   3. Community summaries provide global reasoning context            │
│   4. LLM synthesizes answer from all retrieved context              │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

- **Knowledge Graph Extraction**: Automatically identifies entities and relationships from text using LLMs
- **Community Detection**: Uses the Leiden algorithm to group related entities into communities for global context
- **Hybrid Retrieval**: Combines vector similarity search with graph traversal for multi-hop reasoning
- **Versioned Snapshots**: Community summaries and entity mappings are saved with timestamps, allowing rollback to previous states

### Query Flow

1. **Query** is embedded using KaLM-Embedding model
2. **Vector search** finds similar nodes in the knowledge graph
3. **Entity lookup** retrieves communities containing those entities
4. **Community summaries** provide high-level context about topics
5. **LLM synthesis** generates answer from retrieved context

---

## Tech Stack

| Component       | Technology                                             |
| --------------- | ------------------------------------------------------ |
| Orchestration   | LlamaIndex                                             |
| Graph Database  | Neo4j with APOC & GDS plugins                          |
| LLM             | Any OpenAI-compatible API (Groq, OpenAI, Ollama, etc.) |
| Embeddings      | HuggingFace (KaLM-Embedding-Multilingual)              |
| Backend API     | FastAPI (Python)                                       |
| CLI             | Rust (TUI planned)                                     |
| Development Env | Nix Flakes                                             |

---

## Prerequisites

- **Docker**: Required for Neo4j database
- **Python 3.12+**: For the FastAPI backend
- **Rust (optional)**: For building the CLI from source
- **LLM API Key**: Any OpenAI-compatible API

### Supported LLM Providers

Study Buddy uses LlamaIndex's `OpenAILike` module, which supports any OpenAI-compatible API. Switch providers by editing `study_buddy.yaml`:

| Provider      | Config `api_base`                    | Model Example                         |
| ------------- | ------------------------------------ | ------------------------------------- |
| **Groq** (default) | `https://api.groq.com/openai/v1` | `meta-llama/llama-4-scout-17b-16e-instruct` |
| OpenAI        | `https://api.openai.com/v1`          | `gpt-4o`, `gpt-4-turbo`              |
| Ollama (local)| `http://localhost:11434/v1`          | `llama3.2`, `mistral`                |
| Together AI   | `https://api.together.xyz/v1`        | `meta-llama/Llama-3-70b-chat-hf`      |
| Any OpenAI-compatible | varies                   | varies                                |

**Note:** The environment variable `OPENAI_API_KEY` is used for all providers (the name is kept for compatibility).

### On NixOS

The project includes a `flake.nix` for reproducible development environments. Enter the shell:

```bash
nix develop
```

This provides Python, Rust, and all system dependencies (OpenGL, CUDA libs, etc.).

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/study-buddy.git
cd study-buddy
```

### 2. Configure Environment

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```bash
# Required: Any OpenAI-compatible API key
OPENAI_API_KEY=your_api_key_here
```

**Configuration File:** All other settings are configured via `study_buddy.yaml`. See the [Configuration](#configuration) section for details. You can:

- Edit `study_buddy.yaml` directly to change LLM model, Neo4j credentials, server ports, etc.
- Or set environment variables to override specific settings (e.g., `NEO4J_PASSWORD` for secure deployments)

### 3. Start the Backend

The easiest way to start everything (Neo4j + FastAPI):

```bash
./study-buddy-server.sh
```

This script:

1. Checks Docker availability and starts Neo4j container
2. Creates Python virtual environment if needed
3. Installs dependencies from `requirements.txt`
4. Starts the FastAPI server on `http://localhost:8000`

On first run, this will download:

- Neo4j Docker image (~500MB)
- Python dependencies (~2GB)
- Embedding model files (~500MB)

### 4. Build the CLI (Optional)

The Rust CLI provides a nicer interface. To build:

```bash
cd cli
cargo build --release
cd ..
```

Or use the wrapper script which auto-builds:

```bash
./sb status
```

---

## Usage

### Starting the Backend

```bash
# Start Neo4j + FastAPI backend
./study-buddy-server.sh

# The server runs until you press Ctrl+C
# Backend API: http://localhost:8000
# Neo4j Browser: http://localhost:7474
# Neo4j Bolt: bolt://localhost:7687
```

### Using the CLI (`sb`)

```bash
# Check connection status
./sb status

# Ingest documents into the knowledge graph
./sb ingest input/
./sb ingest input/ --files paper.pdf,notes.md

# Query the knowledge graph
./sb query "What is machine learning?"

# Search for entities
./sb search "neural"
./sb search --entity "machine learning"

# List all communities
./sb community list

# Show a specific community
./sb community show 42

# Manage summary versions
./sb summaries list
./sb summaries current
./sb summaries cleanup --keep 5

# Show configuration
./sb config
```

### Using the API Directly

```bash
# Health check
curl http://localhost:8000/

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/documents"}'

# Query the graph
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics?", "similarity_top_k": 20}'

# Search entities
curl "http://localhost:8000/entities?q=neural&limit=10"

# Get specific entity
curl "http://localhost:8000/entities/Machine%20Learning"

# List communities
curl http://localhost:8000/communities

# Get community details
curl http://localhost:8000/communities/42
```

### API Documentation

Interactive Swagger UI: http://localhost:8000/docs

### Workflow Summary

```
1. Start backend:     ./study-buddy-server.sh
2. Ingest documents:  ./sb ingest input/
3. Query knowledge:   ./sb query "your question"
```

---

## Project Structure

```
study-buddy/
├── sb                      # CLI wrapper script (auto-builds Rust binary)
├── study-buddy-server.sh   # Entry point script (Neo4j + FastAPI)
├── study_buddy.yaml        # Main configuration file
├── flake.nix               # Nix development environment
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── .env.example            # Template for .env file
│
├── src/
│   ├── app.py              # FastAPI application (main backend)
│   ├── config.py           # Configuration loader module
│   ├── main.py             # Legacy standalone ingestion (artifact)
│   ├── core_classes.py     # GraphRAGExtractor, GraphRAGStore, QueryEngine
│   └── ingestion.py        # Document parsing and chunking
│
├── cli/                    # Rust CLI
│   ├── src/
│   │   ├── main.rs         # CLI entry point
│   │   ├── api/            # API client
│   │   ├── commands/       # Command implementations
│   │   └── config/         # Configuration handling
│   └── Cargo.toml
│
├── input/                  # Documents to ingest (gitignored)
├── summaries/              # Community summaries & entity info (gitignored)
├── data/                   # Neo4j data (gitignored)
└── plugins/                # Neo4j plugins (APOC, GDS)
```

**Note:** `src/main.py` is a legacy artifact from early development. The primary ingestion workflow is through the API (`/ingest` endpoint) or CLI (`./sb ingest`).

---

## Configuration

Study Buddy uses a YAML configuration file (`study_buddy.yaml`) for all user-tunable settings, with environment variable overrides for sensitive data.

### Configuration File: `study_buddy.yaml`

The main configuration file contains all settings with helpful comments:

```yaml
# LLM Settings
llm:
  model: "meta-llama/llama-4-scout-17b-16e-instruct"
  api_base: "https://api.groq.com/openai/v1"

# Embedding Model (runs locally)
embedding:
  model: "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"

# Neo4j Database
neo4j:
  url: "bolt://localhost:7687"
  username: "neo4j"
  password: "neo4j2026"
  timeout: 30

# Backend Server
server:
  port: 8000
  host: "0.0.0.0"
  log_level: "INFO"

# Docker (for study-buddy-server.sh)
docker:
  container_name: "neo4j-apoc-gds"
  image: "neo4j:latest"
```

### Switching LLM Providers

Edit `study_buddy.yaml` to use a different provider:

**OpenAI:**
```yaml
llm:
  model: "gpt-4o"
  api_base: "https://api.openai.com/v1"
```

**Ollama (local):**
```yaml
llm:
  model: "llama3.2"
  api_base: "http://localhost:11434/v1"
```

**Together AI:**
```yaml
llm:
  model: "meta-llama/Llama-3-70b-chat-hf"
  api_base: "https://api.together.xyz/v1"
```

### Environment Variables

Environment variables take precedence over config file values:

| Variable              | Description                              | Overrides Config     |
| --------------------- | ---------------------------------------- | -------------------- |
| `OPENAI_API_KEY`      | **Required.** API key for LLM provider   | `llm.api_key`        |
| `NEO4J_PASSWORD`      | Neo4j password (recommended for prod)   | `neo4j.password`     |
| `SERVER_PORT`        | Backend server port                      | `server.port`        |
| `STUDY_BUDDY_CONFIG` | Path to custom config file               | (file location)    |

**Example:**
```bash
# Use a secure Neo4j password
export NEO4J_PASSWORD=my-secure-password

# Use a custom config file
export STUDY_BUDDY_CONFIG=/path/to/custom_config.yaml
```

### Neo4j Settings

Default settings (in `study_buddy.yaml`):

| Setting | Default Value |
|---------|---------------|
| URL     | `bolt://localhost:7687` |
| Username| `neo4j` |
| Password| `neo4j2026` |
| Timeout | `30` seconds |

**Security Tip:** For production deployments, set the password via environment variable instead of storing it in the config file:

```bash
export NEO4J_PASSWORD=your-secure-password
```

### CLI Configuration

The CLI stores user preferences in `~/.config/study-buddy/config.toml`:

```toml
[api]
base_url = "http://localhost:8000"
timeout_seconds = 300

[display]
default_format = "table"
```

### Configuration Priority

Settings are loaded in this order (later overrides earlier):

1. **Built-in defaults** in `src/config.py`
2. **Config file** (`study_buddy.yaml` or `STUDY_BUDDY_CONFIG`)
3. **Environment variables** (highest priority)

---

## Development

### Running Tests

```bash
# Rust CLI tests
cd cli && cargo test

# Python backend (if tests exist)
cd src && python -m pytest
```

### Code Style

- Python: Ruff for linting (`ruff check src/`)
- Rust: Standard `cargo fmt` and `cargo clippy`

### Adding New Document Types

Edit `src/ingestion.py` to add new file format handlers. The `DocumentIngestion` class routes files by extension to appropriate parsers.

---

## Troubleshooting

### Common Issues

**"OPENAI_API_KEY environment variable is required"**

Make sure you've created a `.env` file with your API key:
```bash
cp .env.example .env
# Edit .env and add your key
```

**"Config file not found" Warning**

The application will use default values if `study_buddy.yaml` is missing. To customize settings:
```bash
# The default config file is created automatically on first run
# Or specify a custom config:
export STUDY_BUDDY_CONFIG=/path/to/custom_config.yaml
```

**Port Already in Use**

```bash
# Find what's using the port
ss -tln | grep :8000
# Kill the process or change the port in study_buddy.yaml:
# server:
#   port: 8001
```

**Neo4j Won't Start**

```bash
# Check Docker logs
docker logs neo4j-apoc-gds

# Remove old container
docker rm -f neo4j-apoc-gds
./study-buddy-server.sh
```

**Import Errors on NixOS**

```bash
# Make sure you're in nix-shell
nix develop
./study-buddy-server.sh
```

**"No summary files found" Warning**

This means the knowledge graph is empty. You need to ingest documents first:

1. Place documents in `input/` directory
2. Run `./sb ingest input/` (with backend running)
3. The API will automatically reload with new data

### Logs

- Backend logs print to stdout
- Neo4j logs: `docker logs neo4j-apoc-gds`

---

## API Reference

### Endpoints

| Method   | Path                         | Description                           |
| -------- | ---------------------------- | ------------------------------------- |
| `GET`    | `/`                          | Health check                          |
| `POST`   | `/ingest`                    | Ingest documents into knowledge graph |
| `GET`    | `/ingest/status/{task_id}`   | Check ingestion progress              |
| `POST`   | `/query`                     | Query the knowledge graph             |
| `GET`    | `/entities`                  | Search entities                       |
| `GET`    | `/entities/{name}`           | Get entity details                    |
| `GET`    | `/communities`               | List all communities                  |
| `GET`    | `/communities/{id}`          | Get community details                 |
| `GET`    | `/communities/{id}/entities` | Get entities in community             |
| `GET`    | `/summaries`                 | List summary versions                 |
| `GET`    | `/summaries/current`         | Show active version                   |
| `DELETE` | `/summaries/cleanup`         | Delete old versions                   |

For detailed request/response schemas, see the Swagger UI at `/docs`.

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) for GraphRAG orchestration
- [Neo4j](https://neo4j.com/) for graph storage
- [Groq](https://groq.com/) for fast LLM inference
- [KaLM-Embedding](https://huggingface.co/KaLM-Embedding) for multilingual embeddings

