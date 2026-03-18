# Study Buddy (GraphRAG)

Study Buddy is a Retrieval-Augmented Generation (RAG) system that leverages Knowledge Graphs to provide context-aware answers. It combines **LlamaIndex** and **Neo4j** to extract entities and relationships from unstructured text, building a structured graph that is then partitioned into communities using the **Leiden algorithm**.

## Project Overview

The project implements a **GraphRAG** pipeline with the following key phases:
1.  **Extraction**: Processes unstructured text (CSV) to identify entities and relationships using LLMs (Groq/Llama-4).
2.  **Indexing**: Populates a Neo4j property graph store.
3.  **Community Detection**: Uses the Leiden algorithm (via Neo4j Graph Data Science - GDS) to cluster related entities.
4.  **Summarization**: Generates high-level summaries for each community.
5.  **Querying**: A custom query engine retrieves relevant entities, finds their communities, and uses community summaries to answer complex global queries.

### Tech Stack
-   **Orchestration**: LlamaIndex
-   **Database**: Neo4j (requires Graph Data Science plugin)
-   **LLM**: Groq (meta-llama/llama-4-scout-17b-16e-instruct)
-   **Embeddings**: HuggingFace (KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5)
-   **Web Framework**: FastAPI, Uvicorn
-   **Environment**: Nix (flake.nix) or Python 3.12 (requirements.txt)

## Key Files & Directories

-   `src/main.py`: The indexing entry point. Reads `input/csv/news_articles.csv`, builds the graph, runs community detection, and saves summaries.
-   `src/app.py`: The FastAPI server entry point. Loads persisted summaries and provides a `/query` endpoint.
-   `src/core_classes.py`: Contains core logic for extraction (`GraphRAGExtractor`), storage (`GraphRAGStore`), and querying (`GraphRAGQueryEngine`).
-   `src/prompts/kg_extract_template.txt`: The system prompt used for entity and relationship extraction.
-   `summaries/community_summaries.json`: Persisted summaries generated during the indexing phase.
-   `input/csv/news_articles.csv`: Default input dataset.
-   `flake.nix`: Defines the Nix development environment (Python 3.12, Ruff, Basedpyright).

## Building and Running

### 1. Setup Environment
Using Nix:
```bash
nix develop
```
Or using pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file with your credentials:
```env
GROQ_API_KEY=your_api_key_here
```
*Note: `NEO4JPASSWORD` is currently hardcoded in `src/main.py` and `src/app.py` as `Drewert237?`. This should be moved to `.env` for production.*

### 3. Indexing
Ensure Neo4j is running locally at `bolt://localhost:7687` with GDS installed.
```bash
python src/main.py
```

### 4. Running the API
```bash
python src/app.py
```
The API will be available at `http://localhost:8000`. You can access the Swagger UI at `http://localhost:8000/docs`.

## Development Conventions

-   **Linting**: Use `ruff check .`
-   **Formatting**: Use `ruff format .`
-   **Type Checking**: Use `basedpyright src/`
-   **Data Consistency**: The system relies on `summaries/community_summaries.json`. If you re-index, ensure this file is updated and loaded by the API.
-   **LLM Choice**: Groq is used for both extraction and querying due to its high throughput.
