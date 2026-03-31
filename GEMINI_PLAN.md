# Data Ingestion Pipeline Plan

This plan outlines the architecture for a modular, local-first data ingestion pipeline for the `study-buddy` project. It combines the high-fidelity parsing of **Docling** with specialized methods for core text and data formats to ensure optimal Knowledge Graph extraction.

## 1. Class Architecture

### `IngestionPipeline` Class

A centralized class that detects file types and routes them to the most effective local parser while maintaining consistent metadata and token-ready output.

#### Core Methods:

1.  **`process_via_docling(file_path: str) -> List[BaseNode]` (LOCAL DEFAULT)**
    - **Applicable Types**: `.pdf`, `.docx`, `.pptx`, `.html`, `.xlsx`.
    - **Reader**: `DoclingReader`.
    - **Parser**: `MarkdownNodeParser` (as Docling exports to Markdown by default).
    - **Logic**: Uses the local Docling engine to preserve tables and layout structures, outputting structured Markdown that is highly effective for GraphRAG.

2.  **`process_txt(file_path: str) -> List[BaseNode]`**
    - **Reader**: `SimpleDirectoryReader`.
    - **Parser**: `SentenceSplitter`.
    - **Logic**: Standard sentence-aware chunking for raw text files.

3.  **`process_markdown(file_path: str) -> List[BaseNode]`**
    - **Reader**: `SimpleDirectoryReader`.
    - **Parser**: `MarkdownNodeParser`.
    - **Logic**: Splits by headers (`#`, `##`), preserving document hierarchy.

4.  **`process_csv(file_path: str) -> List[BaseNode]`**
    - **Reader**: `CSVReader`.
    - **Parser**: `SentenceSplitter`.
    - **Logic**: Treats rows as individual documents.

5.  **`process_json(file_path: str) -> List[BaseNode]`**
    - **Reader**: `JSONReader`.
    - **Parser**: `JSONNodeParser`.
    - **Logic**: Flattens nested JSON into path-value pairs (e.g., `user profile status active`), which is significantly better for entity extraction than raw JSON strings.

6.  **`process_code(file_path: str) -> List[BaseNode]`**
    - **Reader**: `SimpleDirectoryReader`.
    - **Parser**: `CodeSplitter`.
    - **Logic**: Uses `tree-sitter` for AST-aware splitting, keeping functions and classes functionally complete.

7.  **`ingest_file(file_path: str) -> List[BaseNode]` (The Dispatcher)**
    - **Logic**:
      1. Checks if the extension is `.txt`, `.md`, `.csv`, `.json`, or a code file.
      2. Routes to the specialized method if matched.
      3. If the extension is supported by Docling (PDF, DOCX, etc.), routes to `process_via_docling`.
      4. Defaults to `process_txt` for unknown text formats.

---

## 2. Technical Implementation Details

### Node Parser Mapping

| File Type         | LlamaIndex Component                   | Reason                                            |
| :---------------- | :------------------------------------- | :------------------------------------------------ |
| **PDF/DOCX/HTML** | `DoclingReader` + `MarkdownNodeParser` | High-fidelity local structure/table preservation. |
| **.txt**          | `SentenceSplitter`                     | Efficient, lightweight chunking.                  |
| **.md**           | `MarkdownNodeParser`                   | Preserves header-based context.                   |
| **.csv**          | `CSVReader` + `SentenceSplitter`       | Reliable row-level processing.                    |
| **.json**         | `JSONNodeParser`                       | Flattens hierarchy for triple extraction.         |
| **Code**          | `CodeSplitter`                         | AST-aware splitting (Python, JS, etc.).           |

### Preserving "Token Input" Structure

To maintain compatibility with the existing `GraphRAGExtractor` prompt, all methods will ensure the final text content of the nodes follows the format:
`{Title/Filename}: {Content}`

---

## 3. Recommended Workflow for Integration

1.  **Dependency Update**:
    Update `requirements.txt` and run `pip install` (or let the Nix flake rebuild):
    ```text
    llama-index-readers-docling
    llama-index-node-parser-all
    llama-index-readers-file
    pymupdf
    tree-sitter
    tree-sitter-languages
    ```
2.  **Main Script Refactoring**:
    - Replace the `pd.read_csv` block in `src/main.py` with an instantiation of `IngestionPipeline`.
    - Call `pipeline.ingest_file(input_path)` to get `nodes` directly.
3.  **Validation**: Run the pipeline with one file of each type (txt, md, csv, json, pdf, py) to verify that the `PropertyGraphIndex` correctly extracts triples from structured local inputs.

## 4. Technical Notes & Troubleshooting

- **Package Names**: `CodeSplitter` is included in `llama-index-node-parser-all`. Ensure both `tree-sitter` and `tree-sitter-languages` are installed to support AST-aware code splitting.
- **Docling Resources**: `DoclingReader` is resource-intensive and may require additional system libraries (e.g., `libGL`, `glib`) already defined in `flake.nix`. If OCR issues occur, verify that `torch` and its dependencies are correctly linked in the Nix environment.
- **Environment Consistency**: Always ensure new dependencies are added to `requirements.txt` to maintain a reproducible environment within the Nix shell.
