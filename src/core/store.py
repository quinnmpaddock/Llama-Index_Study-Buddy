"""GraphRAGStore and GraphRAGQueryEngine — workspace-aware graph storage and querying.

Extracted from core_classes.py to support multi-workspace Neo4j databases
and per-workspace community summary persistence.
"""

import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, LiteralString, Optional, Union, cast

from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore

logger = logging.getLogger(__name__)


class GraphRAGStore(Neo4jPropertyGraphStore):
    """Neo4j-backed property graph store with workspace-aware community detection.

    When *workspace_id* is supplied, the store automatically uses a
    workspace-scoped Neo4j database name (via ``workspace.neo4j_db_name``) and
    scopes GDS projection names to avoid collisions between workspaces.
    """

    def __init__(
        self,
        username: Optional[str] = "neo4j",
        password: Optional[str] = None,
        url: Optional[str] = "bolt://localhost:7867",
        database: Optional[str] = "neo4j",
        llm: Optional[LLM] = None,
        entity_info: Optional[Dict[str, Any]] = None,
        community_summary: Optional[Dict[str, Any]] = None,
        refresh_schema: bool = True,
        create_indexes: bool = True,
        timeout: Optional[float] = None,
        workspace_id: Optional[str] = None,
        data_dir: Optional[str] = None,
        **kwargs,
    ) -> None:
        # If workspace_id provided, use it for the database name
        if workspace_id:
            from workspace import neo4j_db_name

            database = neo4j_db_name(workspace_id)

        super().__init__(
            username=username,
            password=password,
            url=url,
            database=database,
            refresh_schema=refresh_schema,
            create_indexes=create_indexes,
            timeout=timeout,
            **kwargs,
        )

        self.workspace_id = workspace_id
        self.data_dir = data_dir
        self.llm = llm or Settings.llm
        self.graph_name = database  # This is the Neo4j database name
        self.entity_info = entity_info or {}
        self.community_summary = community_summary or {}

    # ------------------------------------------------------------------
    # Community summary generation
    # ------------------------------------------------------------------

    def generate_community_summary(self, text):
        """Generate summary for a given text using an LLM."""
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis of the relationship descriptions."
                    "You must cite the source (provided in brackets) for every key fact or group of facts mentioned."
                    "The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = self.llm.chat(messages)

        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        logger.debug("Community summary response constructed")
        return clean_response

    # ------------------------------------------------------------------
    # Cypher helpers
    # ------------------------------------------------------------------

    def _run_cypher(self, query: str, params: Dict[str, Any] | None = None):
        """Sends cypher commands to the neo4j database"""
        if params is None:
            params = {}
        records, _, _ = self._driver.execute_query(
            cast(LiteralString, query), parameters_=params, database_=self.graph_name
        )
        return [record.data() for record in records]

    # ------------------------------------------------------------------
    # Community building
    # ------------------------------------------------------------------

    def build_communities(self):
        """Builds communities from the graph and persists them to the neo4j database."""
        # Use workspace-scoped projection name to avoid collisions
        if self.workspace_id:
            gds_projection = f"{self.workspace_id}_graph"
        else:
            gds_projection = self.graph_name  # backward compat

        try:
            # project the graph to memory
            self._run_cypher(
                f"""
                MATCH (n:__Entity__)-[r]->(m:__Entity__)
                Return gds.graph.project(
                    '{gds_projection}',
                    n,
                    m,
                    {{}},
                    {{ undirectedRelationshipTypes: ['*']}}

                )
            """
            )

            # run leiden community detection and write to neo4j
            self._run_cypher(
                f"""
                CALL gds.leiden.write('{gds_projection}', {{
                    writeProperty: 'community_id',
                    randomSeed: 19,
                    includeIntermediateCommunities: false,
                    concurrency: 1
                }})
                YIELD communityCount
            """
            )
            self._collect_community_info()
        except Exception as e:
            logger.error("build_communities failed: %s: %s", type(e).__name__, e)
            raise
        finally:
            # drop graph projection
            try:
                self._run_cypher(
                    f"CALL gds.graph.drop('{gds_projection}', false) YIELD graphName"
                )
            except Exception as e:
                logger.debug(
                    "Could not drop GDS graph projection '%s': %s: %s",
                    gds_projection, type(e).__name__, e,
                )

    def _collect_community_info(self):
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.
        """

        query = """
            MATCH (n)
            WHERE n.community_id IS NOT NULL
            UNWIND n.community_id AS community_id
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN
                community_id,
                n.name AS node,
                type(r) as rel_type,
                r.relationship_description AS description,
                coalesce(r.file_name, 'Unknown') AS source,
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
        """Generate and store summaries for each community."""
        logger.debug("summarize communities")
        for community_id, details in community_info.items():
            details_text = "\n".join(details) + "."  # Ensure it ends with a period
            self.community_summary[community_id] = self.generate_community_summary(
                details_text
            )

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        logger.debug("getting community summaries")
        if not self.community_summary:
            self.build_communities()
        return self.community_summary

    # ------------------------------------------------------------------
    # Workspace-scoped summary persistence
    # ------------------------------------------------------------------

    def get_summaries_dir(self) -> Path:
        """Return the directory for this workspace's summaries."""
        if self.data_dir:
            d = Path(self.data_dir) / (self.workspace_id or "default") / "summaries"
        else:
            d = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / ".." / "summaries"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_summaries(self, version: Optional[str] = None) -> str:
        """Save community summaries and entity info to disk.
        Returns the version timestamp string."""
        if version is None:
            version = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        summaries_dir = self.get_summaries_dir()

        summary_path = summaries_dir / f"community_summaries_{version}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.community_summary, f, indent=4)

        entity_path = summaries_dir / f"entity_info_{version}.json"
        with open(entity_path, "w", encoding="utf-8") as f:
            json.dump(self.entity_info, f, indent=4)

        # Update current.json pointer
        current_path = summaries_dir / "current.json"
        current_info = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "files": {
                "community_summaries": f"community_summaries_{version}.json",
                "entity_info": f"entity_info_{version}.json",
            },
            "stats": {
                "total_entities": len(self.entity_info),
                "total_communities": len(self.community_summary),
            },
        }
        with open(current_path, "w", encoding="utf-8") as f:
            json.dump(current_info, f, indent=4)

        return version

    def load_summaries(self) -> tuple:
        """Load community summaries and entity info from disk.
        Returns (community_summaries, entity_info).

        Reads from the version pointed to by current.json.
        If current.json is missing, falls back to the most recent
        community_summaries_*.json / entity_info_*.json files.
        """
        summaries_dir = self.get_summaries_dir()
        current_path = summaries_dir / "current.json"

        if current_path.exists():
            with open(current_path, "r", encoding="utf-8") as f:
                current_info = json.load(f)
            version = current_info["version"]
        else:
            # Find most recent version files
            summary_files = sorted(summaries_dir.glob("community_summaries_*.json"))
            if not summary_files:
                logger.warning("No saved summaries found in %s", summaries_dir)
                return {}, {}
            # Extract version from filename: community_summaries_{version}.json
            latest = summary_files[-1]
            version = latest.stem.replace("community_summaries_", "")

        summary_path = summaries_dir / f"community_summaries_{version}.json"
        entity_path = summaries_dir / f"entity_info_{version}.json"

        community_summaries = {}
        entity_info = {}

        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                community_summaries = json.load(f)

        if entity_path.exists():
            with open(entity_path, "r", encoding="utf-8") as f:
                entity_info = json.load(f)

        if community_summaries:
            self.community_summary = community_summaries
        if entity_info:
            self.entity_info = entity_info

        return community_summaries, entity_info


class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 20  # possible validation error here, come back

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

    async def acustom_query(self, query_str: str) -> Response:
        """Process all community summaries to generate answers to a specific query."""

        entities = self.get_entities(query_str, self.similarity_top_k)

        community_summaries = self.graph_store.get_community_summaries()
        community_ids = self.retrieve_entity_communities(
            self.graph_store.entity_info, entities
        )
        tasks = [
            self.agenerate_answer_from_summary(community_summary, query_str)
            for id, community_summary in community_summaries.items()
            if id in community_ids
        ]

        community_answers = await asyncio.gather(*tasks)
        final_answer = await self.aaggregate_answers(community_answers)
        return Response(
            response=final_answer,
            metadata={
                "communities_consulted": community_ids,
                "entities_found": entities,
            },
        )

    def get_entities(self, query_str, similarity_top_k):
        nodes_retrieved = self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).retrieve(query_str)

        enitites = set()
        pattern = r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$"

        for node_with_score in nodes_retrieved:
            text = node_with_score.node.get_content()
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)

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
            f"how would you answer the following query? Query: {query}\n\n"
            f"IMPORTANT: Preserve all source citations [Source: ...] from the summary in your answer. "
            f"Do not remove or modify any citation markers."
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information. Keep all source citations intact.",
            ),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    async def agenerate_answer_from_summary(self, community_summary, query):
        """async version of generate_answer_from_summary"""

        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}\n\n"
            f"IMPORTANT: Preserve all source citations [Source: ...] from the summary in your answer. "
            f"Do not remove or modify any citation markers."
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information. Keep all source citations intact.",
            ),
        ]
        response = await self.llm.achat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        prompt = (
            "Combine the following intermediate answers into a final, concise response. "
            "IMPORTANT: You MUST preserve every bracketed citation token exactly as it appears "
            + "(for example `[Source: ...]` and `[test_data0.pdf]`). "
            + "Every citation token that appears in the intermediate answers must appear in the final output unchanged. "
            + "Do not remove, rewrite, normalize, or summarize citation markers."
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

    async def aaggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        prompt = (
            "Combine the following intermediate answers into a final, concise response. "
            "IMPORTANT: You MUST preserve every bracketed citation token exactly as it appears "
            + "(for example `[Source: ...]` and `[test_data0.pdf]`). "
            + "Every citation token that appears in the intermediate answers must appear in the final output unchanged. "
            + "Do not remove, rewrite, normalize, or summarize citation markers."
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = await self.llm.achat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response