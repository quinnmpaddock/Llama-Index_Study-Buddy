"""
Database check utility for Study Buddy.
Verifies Neo4j connection and displays basic statistics.
"""
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import get_config
from neo4j import GraphDatabase

def check_db():
    """Check Neo4j database connection and display statistics."""
    config = get_config()    
    print(f"Connecting to Neo4j at {config.neo4j.url}...")
    
    driver = GraphDatabase.driver(
        config.neo4j.url,
        auth=(config.neo4j.username, config.neo4j.password)
    )
    
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        labels = session.run("CALL db.labels() YIELD label RETURN label").value()
        rel_types = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType").value()
        
        print(f"\nNeo4j Database Statistics:")
        print(f"  Nodes: {nodes}")
        print(f"  Relationships: {rels}")
        print(f"  Labels: {labels}")
        print(f"  Relationship Types: {rel_types}")
        
        if rels > 0:
            sample = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 1").single()
            print(f"\n  Sample relationship: {sample}")

    driver.close()
    print("\nDatabase connection successful!")

if __name__ == "__main__":
    check_db()