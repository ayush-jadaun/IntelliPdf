from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from app.schemas.graph import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from neo4j import GraphDatabase
from app.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

router = APIRouter()

# Neo4j driver setup
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def fetch_knowledge_graph_from_neo4j(doc_id: Optional[str] = None) -> KnowledgeGraph:
    """
    Query Neo4j for the whole graph or a subgraph filtered by doc_id.
    """
    with driver.session() as session:
        if doc_id:
            cypher = """
            MATCH (d:Document {id: $doc_id})-[r]->(n)
            RETURN d, r, n
            UNION
            MATCH (n)-[r]->(d:Document {id: $doc_id})
            RETURN n, r, d
            """
            results = session.run(cypher, doc_id=doc_id)
        else:
            cypher = """
            MATCH (a)-[r]->(b)
            RETURN a, r, b
            """
            results = session.run(cypher)

        # Collect nodes and edges
        nodes_dict = {}
        edges: List[KnowledgeGraphEdge] = []
        for record in results:
            n1 = record[0]
            rel = record[1]
            n2 = record[2]
            for n in [n1, n2]:
                if n.id not in nodes_dict:
                    # You may want to extract a label from n.labels or n.get("label", ...).
                    nodes_dict[n.id] = KnowledgeGraphNode(
                        id=n.id,
                        label=next(iter(n.labels), "Node"),
                        type=n.get("type", "Unknown"),
                        score=n.get("score")
                    )
            edges.append(KnowledgeGraphEdge(
                source=n1.id,
                target=n2.id,
                type=rel.type,
                weight=rel.get("weight")
            ))
        return KnowledgeGraph(
            nodes=list(nodes_dict.values()),
            edges=edges
        )

@router.get("/graph/", response_model=KnowledgeGraph)
async def get_knowledge_graph(doc_id: Optional[str] = Query(None, description="Filter by document ID")):
    """
    Return the knowledge graph or a filtered subgraph.
    """
    try:
        graph = fetch_knowledge_graph_from_neo4j(doc_id)
        return graph
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch knowledge graph: {str(e)}")