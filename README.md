# Study Buddy (GraphRAG)

Study Buddy is a Retrieval-Augmented Generation (RAG) system that leverages Knowledge Graphs to provide context-aware answers. By combining **LlamaIndex** and **Neo4j**, it extracts entities and relationships from unstructured text to build a structured graph.

## Key Features

- **Knowledge Graph Extraction**: Automatically identifies entities and relationships using LLMs.
- **Community Detection**: Utilizes the Leiden algorithm to group related entities into communities for better global context.
- **Advanced Querying**: Uses community summaries to answer complex queries that require a high-level understanding of the dataset.

## Tech Stack
- **Orchestration**: LlamaIndex
- **Database**: Neo4j
- **LLM**: Groq (Llama-3)
- **Embeddings**: HuggingFace (KaLM-Embedding)
