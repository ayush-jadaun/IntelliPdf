from neo4j import GraphDatabase
from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Create or update an entity node
def create_entity_node(tx, node):
    tx.run(
        """
        MERGE (n:Entity {id: $id})
        SET n.label = $label,
            n.type = $type,
            n.score = $score
        """,
        id=node.get("id"),
        label=node.get("label"),
        type=node.get("type"),
        score=node.get("score"),
    )

# Create or update a relationship (edge)
def create_edge(tx, edge):
    tx.run(
        """
        MATCH (a:Entity {id: $source})
        MATCH (b:Entity {id: $target})
        MERGE (a)-[r:RELATION {type: $type}]->(b)
        SET r.weight = $weight
        """,
        source=edge.get("source"),
        target=edge.get("target"),
        type=edge.get("type", "CO_OCCURRENCE"),
        weight=edge.get("weight", 1.0),
    )

# Store the full knowledge graph (nodes + edges)
def store_knowledge_graph(graph):
    with driver.session() as session:
        for node in graph.nodes or []:
            session.write_transaction(create_entity_node, node)
        for edge in graph.edges or []:
            # Skip invalid edges
            if not edge.get("source") or not edge.get("target"):
                continue
            session.write_transaction(create_edge, edge)
