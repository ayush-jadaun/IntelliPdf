from fastapi import APIRouter, HTTPException, Body, Depends
from datetime import datetime
from app.schemas.chat import ChatMessage, ChatResponse
from app.core.ai.chat_engine import gemini_chat
from app.core.ai.embeddings import SentenceTransformerEmbedder
from app.core.database.vector_store import query_similar_chunks
from app.core.database.session import get_db
from sqlalchemy.orm import Session
from app.api.v1.endpoints.graph import fetch_knowledge_graph_from_neo4j

router = APIRouter()

@router.post("/chat/", response_model=ChatResponse)
async def chat(message: ChatMessage = Body(...), db: Session = Depends(get_db)):
    # Add timestamp if not present
    if not message.timestamp:
        message.timestamp = datetime.utcnow().isoformat()
    
    # 1. Embed the query
    try:
        embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        query_embedding = embedder.embed_texts([message.text])[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

    # 2. Retrieve relevant document chunks (optionally filtered by document_id)
    search_document_id = getattr(message, "document_id", None)
    try:
        top_chunks = query_similar_chunks(
            db,
            embedding=query_embedding,
            top_k=5,
            min_score=0.1,
            document_id=search_document_id
        )
        context_snippets = [chunk["text"] for chunk in top_chunks]
    except Exception as e:
        context_snippets = []
    
    # 3. Retrieve relevant graph info (optionally filter for document_id)
    try:
        graph = fetch_knowledge_graph_from_neo4j(doc_id=search_document_id)
        query_words = set(message.text.lower().split())
        relevant_nodes = [
            node for node in graph.nodes
            if any(word in node.label.lower() for word in query_words)
        ]
        relevant_edges = [
            edge for edge in graph.edges
            if edge.source in {n.id for n in relevant_nodes} or edge.target in {n.id for n in relevant_nodes}
        ]
        if relevant_nodes:
            graph_context = "Knowledge Graph Nodes:\n" + "\n".join(
                f"- {node.label} ({node.type})" for node in relevant_nodes
            )
            if relevant_edges:
                graph_context += "\nRelations:\n" + "\n".join(
                    f"- {edge.source} --[{edge.type}]--> {edge.target}" for edge in relevant_edges
                )
        else:
            graph_context = "No relevant graph entities found."
    except Exception as e:
        graph_context = "Graph lookup failed."
    
    # 4. Compose full context for Gemini
    system_prompt = (
        "You are an expert research assistant. Use the provided document context and knowledge graph information "
        "to answer the user's question accurately and concisely. If you cite facts, mention which document or entity you found them in."
    )
    context_text = "\n---\n".join([
        "DOCUMENT CONTEXT:",
        "\n".join(context_snippets) if context_snippets else "No relevant document chunks found.",
        "KNOWLEDGE GRAPH CONTEXT:",
        graph_context
    ])
    user_prompt = f"{context_text}\n\nQUESTION:\n{message.text}"

    gemini_messages = [
        {"role": "system", "parts": [system_prompt]},
        {"role": "user", "parts": [user_prompt]}
    ]
    
    # 5. Call Gemini
    try:
        assistant_reply = gemini_chat(gemini_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API failed: {str(e)}")

    return ChatResponse(messages=[
        message,
        ChatMessage(
            sender="assistant",
            text=assistant_reply,
            timestamp=datetime.utcnow().isoformat(),
        )
    ])