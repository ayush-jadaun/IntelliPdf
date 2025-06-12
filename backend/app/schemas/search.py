from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(..., description="The user's search query.")
    top_k: int = Field(5, description="Number of top results to return.")
    alpha: float = Field(0.7, ge=0.0, le=1.0, description="Weight for embedding vs keyword (0=only keyword, 1=only embedding).")
    # Each chunk must have 'text', 'embedding', and optionally any metadata fields
    chunk_list: List[Dict[str, Any]] = Field(..., description="List of text chunks with pre-computed embeddings.")

class SearchResultChunk(BaseModel):
    text: str
    score: float
    chunk_metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[SearchResultChunk]