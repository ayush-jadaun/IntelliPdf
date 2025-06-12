from fastapi import APIRouter, HTTPException, Body
from app.schemas.search import SearchRequest, SearchResponse, SearchResultChunk  # <-- import your schema!
from app.core.ai.embeddings import embed_text_with_gemini
from app.core.ai.similarity import hybrid_similar_chunks

router = APIRouter()

@router.post("/search/", response_model=SearchResponse)
async def search_chunks(request: SearchRequest = Body(...)):
    """
    Hybrid search: uses both Gemini semantic embedding and TF-IDF keyword similarity.
    Requires chunk_list to be passed with pre-computed embeddings.
    """
    if not request.query or not request.chunk_list:
        raise HTTPException(status_code=400, detail="Query and chunk_list are required.")

    # Get embedding for user query
    try:
        query_embedding = embed_text_with_gemini(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # Hybrid search (embedding + keyword)
    try:
        top_chunks = hybrid_similar_chunks(
            query=request.query,
            query_embedding=query_embedding,
            chunk_list=request.chunk_list,
            top_k=request.top_k,
            alpha=request.alpha,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")

    results = [
        SearchResultChunk(
            text=chunk["text"],
            score=score,
            chunk_metadata={k: v for k, v in chunk.items() if k not in ("embedding", "text")},
        )
        for chunk, score in top_chunks
    ]

    return SearchResponse(results=results)