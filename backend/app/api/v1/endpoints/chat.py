from fastapi import APIRouter, HTTPException, Body
from datetime import datetime
from app.schemas.chat import ChatMessage, ChatResponse
from app.core.ai.chat_engine import generate_gemini_response  # <-- your wrapper for Gemini API

router = APIRouter()

@router.post("/chat/", response_model=ChatResponse)
async def chat(message: ChatMessage = Body(...)):
    # Add timestamp if not present
    if not message.timestamp:
        message.timestamp = datetime.utcnow().isoformat()
    
    # Call Gemini API to generate a response
    try:
        assistant_reply = generate_gemini_response(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API failed: {str(e)}")
    
    # Return both user message and assistant response
    return ChatResponse(messages=[
        message,
        ChatMessage(
            sender="assistant",
            text=assistant_reply,
            timestamp=datetime.utcnow().isoformat(),
        )
    ])