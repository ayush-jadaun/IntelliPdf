from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class FileAttachment(BaseModel):
    filename: str = Field(..., description="Name of the uploaded file")
    filetype: str = Field(..., description="Type of the file, e.g., 'pdf'")
    url: Optional[str] = Field(None, description="URL or path to retrieve the file")
    doc_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Extracted metadata such as page count, title, etc."
    )

class ChatMessage(BaseModel):
    sender: str = Field(..., description="Who sent the message (user, assistant, system, etc.)")
    text: Optional[str] = Field(None, description="Message content")
    timestamp: Optional[str] = Field(None, description="ISO8601 timestamp of the message")
    attachments: Optional[List[FileAttachment]] = Field(
        None, description="List of attached files"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Optional metadata (e.g., message id, source, etc.)"
    )

class ChatSession(BaseModel):
    session_id: str = Field(..., description="Unique identifier for the chat session")
    messages: List[ChatMessage] = Field(..., description="Ordered list of messages in the chat")
    created_at: Optional[str] = Field(None, description="When the chat session was created")
    updated_at: Optional[str] = Field(None, description="When the chat session was last updated")

class ChatResponse(BaseModel):
    messages: List[ChatMessage]