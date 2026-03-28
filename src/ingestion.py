import json
from typing import List, Sequence

from llama_index.core import Document, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import (JSONNodeParser, MarkdownNodeParser,
                                          SentenceSplitter)
from llama_index.core.schema import BaseNode, TextNode
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.readers.file import CSVReader, FlatReader, MarkdownReader


class DocumentIngestion:
    def __init__(self):
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        self.md_parser = MarkdownNodeParser()
        self.json_parser = JSONNodeParser()
        self.docling_reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
        self.docling_parser = DoclingNodeParser()
        self.md_reader = MarkdownReader()
        self.csv_reader = CSVReader(concat_rows=False)
        self.txt_reader = FlatReader()

    DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".xlsx"}

    def ingestion(self, file_path: str) -> List[BaseNode]:
        """Routes files based on extension"""
        file_extractor = {
            ".pdf": self.docling_reader,
            ".docx": self.docling_reader,
            ".pptx": self.docling_reader,
            ".html": self.docling_reader,
            ".xlsx": self.docling_reader,
            ".md": self.md_reader,
            ".csv": self.csv_reader,
            ".txt": self.txt_reader,
        }

        dir_reader = SimpleDirectoryReader(
            input_dir=file_path, file_extractor=file_extractor
        )

        all_nodes: List[BaseNode] = []
        for docs in dir_reader.iter_data():
            parser = self.get_parser_for_doc(docs[0])
            nodes = parser.get_nodes_from_documents(docs)
            # Enforce chunk size
            if parser != self.text_splitter:
                split_nodes = []
                for node in nodes:
                    chunks = self.text_splitter.split_text(node.get_content())
                    for chunk in chunks:
                        split_nodes.append(
                            TextNode(
                                text=chunk,
                                metadata=node.metadata.copy(),
                                relationships=node.relationships,
                            )
                        )

            if nodes:
                print(f"DEBUG: First node metadata keys: {nodes[0].metadata.keys()}")
                for k, v in nodes[0].metadata.items():
                    print(f"DEBUG: metadata[{k}] type: {type(v)}")
            for node in nodes:
                if node.metadata:
                    self._sanitize_metadata(node.metadata)
            all_nodes.extend(nodes)
        return all_nodes

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """Sanitize metadata for Neo4j compatibility:
        - Convert large integers to strings.
        - Convert dictionaries to JSON strings.
        - Convert lists with non-primitive elements to JSON strings.
        """
        MAX_INT = 9223372036854775807
        MIN_INT = -9223372036854775808

        for k, v in metadata.items():
            if isinstance(v, int) and (v > MAX_INT or v < MIN_INT):
                metadata[k] = str(v)
            elif isinstance(v, dict):
                metadata[k] = json.dumps(v)
            elif isinstance(v, list):
                # Check if the list contains only Neo4j-compatible primitives
                is_primitive_list = all(
                    isinstance(item, (str, int, float, bool)) or item is None
                    for item in v
                )
                if is_primitive_list:
                    # Still need to check for large ints within the primitive list
                    for i, item in enumerate(v):
                        if isinstance(item, int) and (item > MAX_INT or item < MIN_INT):
                            v[i] = str(item)
                else:
                    # Contains dicts or other lists, so stringify the whole list
                    metadata[k] = json.dumps(v)
        return metadata

    def get_parser_for_doc(self, doc: Document):
        """Returns appropriate node parser based on file extension"""
        from pathlib import Path

        ext = Path(doc.metadata.get("file_path", "")).suffix.lower()
        if ext in self.DOCLING_EXTENSIONS:
            return self.docling_parser
        elif ext == ".md":
            return self.md_parser
        elif ext == ".json":
            return self.json_parser
        else:
            return self.text_splitter

    # def process_via_docling(self, docs: Sequence[Document]) -> List[BaseNode]:
    #     return self.docling_parser.get_nodes_from_documents(documents=docs)
    #
    # def process_md(self, docs: Sequence[Document]) -> List[BaseNode]:
    #     return self.md_parser.get_nodes_from_documents(documents=docs)
    #
    # def process_json(self, docs: Sequence[Document]) -> List[BaseNode]:
    #     return self.json_parser.get_nodes_from_documents(documents=docs)
    #
    # def process_csv(self, docs: Sequence[Document]) -> List[BaseNode]:
    #     # CSVReader produces one Document per row; just return as text nodes
    #     return self.text_splitter.get_nodes_from_documents(documents=docs)
    #
    # def process_txt(self, docs: Sequence[Document]) -> List[BaseNode]:
    #     return self.text_splitter.get_nodes_from_documents(documents=docs)
