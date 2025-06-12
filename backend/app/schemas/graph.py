from typing import Any, List, Dict, Optional
from pydantic import BaseModel

class KnowledgeGraphNode(BaseModel):
    id: str
    label: str
    type: str
    score: Optional[float] = None

class KnowledgeGraphEdge(BaseModel):
    source: str
    target: str
    type: str
    weight: Optional[float] = None

class KnowledgeGraph(BaseModel):
    nodes: List[KnowledgeGraphNode]
    edges: List[KnowledgeGraphEdge]

class ProcessedDocumentResponse(BaseModel):
    file_path: str
    metadata: Dict[str, Any]
    full_text: str
    text_chunks: List[Dict[str, Any]]
    tables: List[Any]
    images: List[Any]
    structure: Dict[str, Any]
    analytics: Dict[str, Any]
    semantic_chunks: List[Dict[str, Any]]
    knowledge_graph: KnowledgeGraph      # <--- Add this