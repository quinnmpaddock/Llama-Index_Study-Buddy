from typing import List, Sequence

from llama_index.core import Document, SimpleDirectoryReader, StorageContext
from llama_index.core.node_parser import (JSONNodeParser, MarkdownNodeParser,
                                          SentenceSplitter)
from llama_index.core.schema import BaseNode
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
            all_nodes.extend(parser.get_nodes_from_documents(docs))
        return all_nodes

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
