import json
import logging
import os
import re

import pandas as pd
from IPython.display import Markdown, display
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.groq import Groq

from core_classes import GraphRAGExtractor, GraphRAGQueryEngine, GraphRAGStore

NEO4JPASSWORD = "Drewert237?"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Setup local embedding model ...")
# setup local embedding model (called in classes)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5"
)
Settings.llm = Groq(model="meta-llama/llama-4-scout-17b-16e-instruct")

print("Setup logging ...")
# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TEMP - specify the test data location
# news = pd.read_csv("input/csv/news_articles.csv")
# news.head()
input_file = "news_articles.csv"
input_path = os.path.join(BASE_DIR, "..", "input", "csv", input_file)
data_csv = pd.read_csv(input_path, nrows=516)
data_csv.head()

# input csv => array
documents = [
    Document(text=f"{row['title']}: {row['text']}", metadata={"title": row["title"]})
    for i, row in data_csv.iterrows()
]

splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=20,
)

print("Extract nodes from documents ...")
nodes = splitter.get_nodes_from_documents(documents)
# source llm

llm = Groq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    # CHANGE THIS!! UNSAFE!!
    # api_key="",
)
print("LLM called!")

# test llm connection
# response = llm.complete("Explain the importance of low latency LLMs")
# print(response)

template_loc = "prompts"
template_fileName = "kg_extract_template.txt"
template_prompt = os.path.join(BASE_DIR, template_loc, template_fileName)


def parse_fn(response_str: str):
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, response_str, re.DOTALL)
    entities = []
    relationships = []
    if not match:
        return entities, relationships
    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        entities = [
            (
                entity["entity_name"],
                entity["entity_type"],
                entity["entity_description"],
            )
            for entity in data.get("entities", [])
        ]
        relationships = [
            (
                relation["source_entity"],
                relation["target_entity"],
                relation["relation"],
                relation["relationship_description"],
            )
            for relation in data.get("relationships", [])
        ]
        return entities, relationships
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return entities, relationships


def main():
    print("Extracting triplets ...")
    with open(template_prompt, "r", encoding="utf-8") as f:
        KG_TRIPLET_EXTRACT_TMPL = f.read()

        kg_extractor = GraphRAGExtractor(
            llm=llm,
            extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
            max_paths_per_chunk=2,
            parse_fn=parse_fn,
        )

    print("Connecting to Neo4j ...")
    graph_store = GraphRAGStore(
        username="neo4j", password=NEO4JPASSWORD, url="bolt://localhost:7687"
    )

    print("Indexing data ...")

    index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )
    print("Setting up query engine")
    index.property_graph_store.build_communities()

    query_engine = GraphRAGQueryEngine(
        graph_store=index.property_graph_store,
        llm=llm,
        index=index,
        similarity_top_k=10,
    )

    response = query_engine.query("What are the main news discussed in the document?")

    print("Displaying response ...")
    print(response.response)


if __name__ == "__main__":
    main()
