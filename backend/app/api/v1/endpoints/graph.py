from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from app.schemas.graph import KnowledgeGraph

router = APIRouter()

# Dummy in-memory graph for demonstration, replace with real data source
demo_graph = KnowledgeGraph(
    nodes=[
        {"id": "doc1", "label": "Document", "properties": {"title": "AI Paper"}},
        {"id": "kw1", "label": "Keyword", "properties": {"name": "Machine Learning"}},
    ],
    edges=[
        {"source": "doc1", "target": "kw1", "relation": "HAS_KEYWORD"}
    ]
)

@router.get("/graph/", response_model=KnowledgeGraph)
async def get_knowledge_graph(doc_id: Optional[str] = Query(None, description="Filter by document ID")):
    """
    Return the knowledge graph or a filtered subgraph.
    """
    # For real use, fetch and filter your actual graph here
    if doc_id:
        nodes = [n for n in demo_graph.nodes if n.id == doc_id]
        node_ids = {n.id for n in nodes}
        edges = [e for e in demo_graph.edges if e.source in node_ids or e.target in node_ids]
        return KnowledgeGraph(nodes=nodes, edges=edges)
    return demo_graph