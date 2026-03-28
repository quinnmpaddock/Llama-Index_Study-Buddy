from neo4j import GraphDatabase
import os

NEO4JPASSWORD = "neo4j2026"
url = "bolt://localhost:7687"

def check_db():
    driver = GraphDatabase.driver(url, auth=("neo4j", NEO4JPASSWORD))
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        labels = session.run("CALL db.labels() YIELD label RETURN label").value()
        rel_types = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType").value()
        
        print(f"Nodes: {nodes}")
        print(f"Relationships: {rels}")
        print(f"Labels: {labels}")
        print(f"Relationship Types: {rel_types}")
        
        if rels > 0:
            sample = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 1").single()
            print(f"Sample relationship: {sample}")

    driver.close()

if __name__ == "__main__":
    check_db()
