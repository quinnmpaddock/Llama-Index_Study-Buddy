import json
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Document, PropertyGraphIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.openai_like import OpenAILike

from config import get_config
from core_classes import GraphRAGExtractor, GraphRAGStore
from ingestion import DocumentIngestion
from utils.parsing import parse_fn

# Load environment variables
load_dotenv()

# Load configuration
config = get_config()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup logging
logging.basicConfig(level=config.server.log_level)
logger = logging.getLogger(__name__)

# Setup models using config
print(f"Loading embedding model: {config.embedding.model}")
Settings.embed_model = HuggingFaceEmbedding(
    model_name=config.embedding.model
)

if not config.llm.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

print(f"Using LLM: {config.llm.model} @ {config.llm.api_base}")
Settings.llm = OpenAILike(
    model=config.llm.model,
    api_base=config.llm.api_base,
    api_key=config.llm.api_key,
    is_chat_model=True,
)

llm = Settings.llm

input_path = os.path.join(BASE_DIR, "..", "input")

template_loc = "prompts"
template_fileName = "kg_extract_template.txt"
template_prompt = os.path.join(BASE_DIR, template_loc, template_fileName)


def main():
    ingestor = DocumentIngestion()
    print(f"Extracting nodes from {input_path}")
    nodes = ingestor.ingestion(input_path)
    print(f"Extracted {len(nodes)} nodes.")

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
    print(f"Connecting to {config.neo4j.url} as {config.neo4j.username}")
    graph_store = GraphRAGStore(
        username=config.neo4j.username,
        password=config.neo4j.password,
        url=config.neo4j.url,
    )
    documents = [
        Document(
            text=node.get_content(),
            metadata={
                k: v
                for k, v in node.metadata.items()
                if isinstance(v, (str, int, float, bool))
            },
        )
        for node in nodes
    ]
    print(f"Indexing {len(documents)} document chunks ...")

    index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        show_progress=True,
    )

    # save community summaries to json
    index.property_graph_store.get_community_summaries()
    output_dir = os.path.join(BASE_DIR, "..", "summaries")
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "community_summaries.json")
    entity_info_path = os.path.join(output_dir, "entity_info.json")

    # persisting community summaries
    if os.path.exists(summary_path):
        if (
            input(f"'{summary_path}' already exists. Overwrite? (y/n):").strip().lower()
            != "y"
        ):
            print("Operation cancelled.")
            return
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(index.property_graph_store.community_summary, f, indent=4)
    print(f"Community summaries saved to {summary_path}")

    # persisting entity info paths
    if os.path.exists(entity_info_path):
        if (
            input(f"'{entity_info_path}' already exists. Overwrite? (y/n):")
            .strip()
            .lower()
            != "y"
        ):
            print("Operation cancelled.")
            return
    with open(entity_info_path, "w", encoding="utf-8") as f:
        json.dump(index.property_graph_store.entity_info, f, indent=4)
    print(f"Entity info mapping saved to {entity_info_path}")


if __name__ == "__main__":
    main()
