from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class SearchRequest(BaseModel):
    query: str
    document_id: Optional[int] = None
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.0
    # Remove chunk_list

class SearchResultChunk(BaseModel):
    text: str
    score: float
    chunk_metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResultChunk]