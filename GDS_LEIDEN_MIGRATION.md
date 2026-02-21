# Guide: Migrating from Graspologic to Neo4j GDS Leiden

This document outlines the steps to replace the local `graspologic` Leiden implementation with Neo4j Graph Data Science (GDS) within the Study Buddy project.

## Overview
- **Source:** Client-side processing (Python/NetworkX/Graspologic).
- **Target:** Database-side processing (Neo4j GDS).
- **Benefits:** Improved performance for large graphs, reduced client-side memory usage, and persistence of community assignments in the database.

---

## Implementation Details

### 1. Requirements
- Neo4j GDS Plugin installed and enabled in the Neo4j instance.
- Verify with: `RETURN gds.version()`

### 2. Code Changes in `src/core_classes.py`

#### Remove Legacy Dependencies
Delete the following imports:
```python
import networkx as nx
from graspologic.partition import hierarchical_leiden
```

#### Update `GraphRAGStore`
Replace the local graph processing logic with GDS Cypher calls.

##### Add Cypher Execution Helper
```python
def _run_cypher(self, query: str, params: dict = None):
    if params is None:
        params = {}
    with self._driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]
```

##### Update `build_communities`
```python
def build_communities(self):
    graph_name = "kg_projection"
    
    # Cleanup previous projection
    try:
        self._run_cypher(f"CALL gds.graph.drop('{graph_name}', false)")
    except Exception:
        pass

    # Project the graph
    self._run_cypher(f"""
        CALL gds.graph.project(
            '{graph_name}',
            '*',
            '*'
        )
    """)

    # Run Leiden in write mode
    self._run_cypher(f"""
        CALL gds.leiden.write(
            '{graph_name}',
            {{
                writeProperty: 'community_ids',
                includeIntermediateCommunities: true, 
                gamma: 1.0,
                tolerance: 0.0001
            }}
        )
    """)

    # Drop projection and collect info
    self._run_cypher(f"CALL gds.graph.drop('{graph_name}', false)")
    self._collect_community_info_from_db()
```

##### Implement `_collect_community_info_from_db`
```python
def _collect_community_info_from_db(self):
    query = """
        MATCH (n)
        WHERE n.community_ids IS NOT NULL
        UNWIND n.community_ids AS community_id
        MATCH (n)-[r]->(m)
        RETURN 
            community_id,
            n.name AS node,
            type(r) AS rel_type,
            r.relationship_description AS description,
            coalesce(r.title, 'Unknown Source') AS source,
            m.name AS neighbor
    """
    
    results = self._run_cypher(query)
    entity_info = defaultdict(set)
    community_info = defaultdict(list)

    for row in results:
        cluster_id = row['community_id']
        node = row['node']
        entity_info[node].add(cluster_id)
        
        detail = f"{node} -> {row['neighbor']} -> {row['rel_type']} -> {row['description']} [Source: {row['source']}]"
        community_info[cluster_id].append(detail)

    self.entity_info = {k: list(v) for k, v in entity_info.items()}
    unique_community_info = {k: list(set(v)) for k, v in community_info.items()}
    self._summarize_communities(unique_community_info)
```

### 3. Key GDS Parameters
- **`includeIntermediateCommunities: true`**: Required to replicate the hierarchical clustering behavior of `graspologic`.
- **`gamma`**: Controls community granularity. 
    - `1.0` (Default).
    - `> 1.0` for smaller communities.
    - `< 1.0` for larger communities.
- **`writeProperty`**: The node property where community IDs are stored (`community_ids`).
