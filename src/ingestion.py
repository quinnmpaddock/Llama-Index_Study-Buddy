from typing import List

from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import (JSONNodeParser, MarkdownNodeParser,
                                          SentenceSplitter)
from llama_index.core.schema import BaseNode
from llama_index.readers.docling import DoclingReader


class DocumentIngestion:
    def __init__(self):
        self.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
        self.md_parser = MarkdownNodeParser()
        self.json_parser = JSONNodeParser()
        self.docling_reader = DoclingReader()

    def ingestion(self, file_path: str) -> Document:
        """Routes files based on extension"""
        ext = file_path.lower().split(".")[-1]
        if ext in ["pdf", "docx", "pptx", "html", "xlsx"]:
            return self.process_via_docling(file_path)
        elif ext == "md":
            return self.process_md(file_path)
        elif ext == "json":
            return self.process_json(file_path)
        elif ext == "csv":
            return self.process_csv(file_path)
        else:
            return self.process_txt(file_path)

    def process_via_docling(self, file_path: str) -> Document:
        pass

    def process_md(self, file_path: str) -> Document:
        pass

    def process_json(self, file_path: str) -> Document:
        pass

    def process_csv(self, file_path: str) -> Document:
        pass

    def process_txt(self, file_path: str) -> Document:
        pass
