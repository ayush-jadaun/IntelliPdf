from fastapi import APIRouter, HTTPException, Body, Query, Depends
from app.schemas.search import SearchRequest, SearchResponse, SearchResultChunk
from app.core.ai.embeddings import embed_texts_with_gemini
from app.core.database.vector_store import query_similar_chunks  # <-- use vector store!
from app.core.database.session import get_db
from sqlalchemy.orm import Session
from typing import Optional

router = APIRouter()

@router.post("/search/", response_model=SearchResponse)
async def search_chunks(
    request: SearchRequest = Body(...),
    db: Session = Depends(get_db)
):
    """
    Search for relevant semantic chunks using embeddings in the database.
    Optionally filter by document ID.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required.")

    # 1. Embed the query
    try:
        query_embedding = embed_texts_with_gemini(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # 2. Query the vector store for similar chunks
    try:
        top_chunks = query_similar_chunks(
            db,
            embedding=query_embedding,
            top_k=request.top_k or 5,
            min_score=request.min_score or 0.0,
            document_id=request.document_id if hasattr(request, "document_id") else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

    results = [
        SearchResultChunk(
            text=chunk["text"],
            score=chunk["score"],
            chunk_metadata={k: v for k, v in chunk.items() if k not in ("embedding", "text", "score")},
        )
        for chunk in top_chunks
    ]

    return SearchResponse(results=results)