from neo4j import GraphDatabase
from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def create_node(tx, name):
    tx.run("CREATE (n:Entity {name: $name})", name=name)

def add_entity(name: str):
    with driver.session() as session:
        session.write_transaction(create_node, name)
