import asyncio
import re

import nest_asyncio

# import networkx as nx

nest_asyncio.apply()

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Union

# from graspologic.partition import hierarchical_leiden
from IPython.display import Markdown, display
from llama_index.core import PropertyGraphIndex
from llama_index.core.async_utils import run_jobs
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.graph_stores.types import (KG_NODES_KEY,
                                                 KG_RELATIONS_KEY, EntityNode,
                                                 Relation)
from llama_index.core.indices.property_graph.utils import \
    default_parse_triplets_fn
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import \
    DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.groq import Groq


class GraphRAGExtractor(TransformComponent):
    """Extract triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.

    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int

    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 10,
        num_workers: int = 4,
    ) -> None:
        """Init params."""
        from llama_index.core import Settings

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"

    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes."""
        return asyncio.run(self.acall(nodes, show_progress=show_progress, **kwargs))

    async def _aextract(self, node: BaseNode) -> BaseNode:
        """Extract triples from a node."""
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )

            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """Extract triples from nodes async."""
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))

        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )


class GraphRAGStore(Neo4jPropertyGraphStore):
    community_summary = {}
    entity_info = None
    max_cluster_size = 5
    graph_name = "neo4j"
    print("Start of GraphRAGStore")

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        # print("constructing chat message to gnerate community summaries ...")
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis of the relationship descriptions."
                    "You must cite the source (provided in brackets) for every key fact or group of facts mentioned"
                    "The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        llm = Groq(model="meta-llama/llama-4-scout-17b-16e-instruct")
        response = llm.chat(messages)

        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        print("TEST: constructed")
        return clean_response

    def _run_cypher(self, query: str, params: Dict[str, Any] | None = None):
        """Sends cypher commands to the neo4j database"""
        if params is None:
            params = {}
        records, _, _ = self._driver.execute_query(
            query, parameters_=params, database_=self.graph_name
        )
        return [record.data() for record in records]

    def build_communities(self):
        """Builds communities from the graph and persists them to the neo4j database"""

        # check for existing graph projection
        # try:
        #     self._run_cypher(
        #         f"CALL gds.graph.drop('{self.graph_name}') YIELD graphName"
        #     )
        #     print(f"{self.graph_name} was found open and dropped from memory.")
        # except Exception:
        #     pass
        try:
            # project the graph to memory
            self._run_cypher(
                f"""
                MATCH (source:__Node__)
                OPTIONAL MATCH (source)-[r]->(target:__Node__)
                RETURN gds.graph.project(
                    '{self.graph_name}',
                    source,
                    target,
                    {{}},
                    {{ undirectedRelationshipTypes: ['*']}}
                )
            """
            )

            # run leiden community detection and write to neo4j
            self._run_cypher(
                f"""
                CALL gds.leiden.write('{self.graph_name}', {{
                    writeProperty: 'community_ids',
                    randomSeed: 19,
                    includeIntermediateCommunities: true,
                    concurrency: 1
                }})
                YIELD communityCount
            """
            )
        finally:
            # drop graph projection
            self._run_cypher(
                f"CALL gds.graph.drop('{self.graph_name}', false) YIELD graphName"
            )
        self._collect_community_info()

    def _collect_community_info(self):
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.
        """

        query = """
            MATCH (n)
            WHERE n.community_ids IS NOT NULL
            UNWIND n.community_ids AS community_id
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN
                community_id,
                n.name AS node,
                type(r) as rel_type,
                r.relationship_description AS description,
                coalesce(r.title, 'Unknown Source') AS source,
                m.name as neighbor 
        """
        results = self._run_cypher(query)
        entity_info = defaultdict(list)
        community_info = defaultdict(list)

        for row in results:
            cluster_id = row["community_id"]
            node = row["node"]
            entity_info[node].append(cluster_id)
            if row["neighbor"] is not None and row["rel_type"] is not None:
                detail = f"{node} -> {row['neighbor']} -> {row['rel_type']} -> {row['description']} [Source: {row['source']}]"
                community_info[cluster_id].append(detail)

        # converts entity_info sets into lists for easier serialization (CURRENTLY UNNECESSARY)
        self.entity_info = {k: list(v) for k, v in entity_info.items()}
        self._summarize_communities(community_info)

    def _summarize_communities(self, community_info):
        print("summarize communities")
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = "\n".join(details) + "."  # Ensure it ends with a period
            self.community_summary[community_id] = self.generate_community_summary(
                details_text
            )

    def get_community_summaries(self):
        print("getting community summaries")
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary


class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 20

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""

        entities = self.get_entities(query_str, self.similarity_top_k)

        community_summaries = self.graph_store.get_community_summaries()
        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for id, community_summary in community_summaries.items()
            if id in community_ids
        ]

        final_answer = self.aggregate_answers(community_answers)
        return final_answer

    def get_entities(self, query_str, similarity_top_k):
        nodes_retrieved = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)

        enitites = set()
        pattern = r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"

        for node in nodes_retrieved:
            matches = re.findall(pattern, node.text, re.MULTILINE | re.IGNORECASE)

            for match in matches:
                subject = match[0]
                obj = match[2]
                enitites.add(subject)
                enitites.add(obj)

        return list(enitites)

    def retrieve_entity_communities(self, entity_info, entities):
        """
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
        entity_info (dict): Dictionary mapping entities to their cluster IDs (list).
        entities (list): List of entity names to retrieve information for.

        Returns:
        List of community or cluster IDs to which an entity belongs.
        """
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",
            ),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        # intermediate_text = " ".join(community_answers)
        prompt = (
            "Combine the following intermediate answers into a final, concise response."
            "Ensure that all source citations provided in the intermediate answers are"
            "preserved in the final output"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response
