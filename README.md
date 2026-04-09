# Study Buddy (GraphRAG)

Study Buddy is a Retrieval-Augmented Generation (RAG) system that leverages **Knowledge Graphs** to provide context-aware answers to complex queries. By combining LlamaIndex and Neo4j, it extracts entities and relationships from unstructured text to build a structured graph, then uses community detection to enable global reasoning across your entire document corpus.

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Document Ingestion                         │
│  PDF, DOCX, MD, CSV, TXT, HTML, XLSX                                 │
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

- **Knowledge Graph Extraction**: Automatically identifies entities and relationships from text using LLMs (Groq/Llama-4)
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

| Component       | Technology                                |
| --------------- | ----------------------------------------- |
| Orchestration   | LlamaIndex                                |
| Graph Database  | Neo4j with APOC & GDS plugins             |
| LLM             | Groq API (Llama-4-Scout)                  |
| Embeddings      | HuggingFace (KaLM-Embedding-Multilingual) |
| Backend API     | FastAPI (Python)                          |
| CLI             | Rust (TUI planned)                        |
| Development Env | Nix Flakes                                |

---

## Prerequisites

- **Docker**: Required for Neo4j database
- **Python 3.12+**: For the FastAPI backend
- **Rust (optional)**: For building the CLI from source
- **Groq API Key**: Get one at [console.groq.com](https://console.groq.com)

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

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=gsk_your_groq_api_key_here
```

Note: Despite the name, this is your Groq API key. The variable name is for compatibility with OpenAI-like API clients.

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

### Document Ingestion

To ingest documents into the knowledge graph:

```bash
# Using main.py directly (builds new graph)
python src/main.py

# Documents go in the input/ directory
# Supported formats: PDF, DOCX, PPTX, HTML, XLSX, MD, CSV, TXT
```

Place your documents in the `input/` directory before running ingestion.

---

## Project Structure

```
study-buddy/
├── sb                      # CLI wrapper script
├── study-buddy-server.sh   # Entry point script (Neo4j + FastAPI)
├── flake.nix               # Nix development environment
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
│
├── src/
│   ├── app.py              # FastAPI application (main backend)
│   ├── main.py             # Document ingestion script
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

---

## Configuration

### Environment Variables

| Variable         | Required | Description                       |
| ---------------- | -------- | --------------------------------- |
| `OPENAI_API_KEY` | Yes      | Groq API key (key name is legacy) |

### Neo4j Settings

Default credentials (configured in `study-buddy-server.sh`):

| Setting   | Value       |
| --------- | ----------- |
| Username  | `neo4j`     |
| Password  | `neo4j2026` |
| Bolt Port | `7687`      |
| HTTP Port | `7474`      |

### CLI Configuration

The CLI stores config in `~/.config/study-buddy/config.toml`:

```toml
[api]
base_url = "http://localhost:8000"
timeout_seconds = 300

[display]
default_format = "table"
```

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

**Port Already in Use**

```bash
# Find what's using the port
ss -tln | grep :8000
# Kill the process or change the port in study-buddy-server.sh
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
2. Run `python src/main.py` to build the graph
3. Restart the backend

### Logs

- Backend logs print to stdout
- Neo4j logs: `docker logs neo4j-apoc-gds`

---

## API Reference

### Endpoints

| Method   | Path                         | Description                |
| -------- | ---------------------------- | -------------------------- |
| `GET`    | `/`                          | Health check               |
| `POST`   | `/query`                     | Query the knowledge graph  |
| `GET`    | `/entities`                  | Search entities            |
| `GET`    | `/entities/{name}`           | Get entity details         |
| `GET`    | `/communities`               | List all communities       |
| `GET`    | `/communities/{id}`          | Get community details      |
| `GET`    | `/communities/{id}/entities` | Get entities in community  |
| `POST`   | `/ingest`                    | Ingest documents (planned) |
| `GET`    | `/summaries`                 | List summary versions      |
| `GET`    | `/summaries/current`         | Show active version        |
| `DELETE` | `/summaries/cleanup`         | Delete old versions        |

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

