from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DocumentMetadata(BaseModel):
    title: Optional[str]
    author: Optional[str]
    page_count: int
    file_size: int

class TextChunk(BaseModel):
    text: str
    page_number: int
    chunk_type: str
    font_info: Optional[Dict[str, Any]]
    bbox: Optional[List[float]]

class TableData(BaseModel):
    page_number: int
    data: List[List[str]]

class ImageData(BaseModel):
    page_number: int
    ext: str
    width: int
    height: int

class DocumentStructure(BaseModel):
    total_chunks: int
    chunk_types: Dict[str, int]
    headers: List[Dict[str, Any]]
    pages: Dict[int, List[Dict[str, Any]]]

class TextAnalytics(BaseModel):
    word_count: int
    sentence_count: int
    keywords: List[str]
    summary: Optional[str]
    entities: List[str]

class ProcessedDocumentResponse(BaseModel):
    file_path: str
    metadata: DocumentMetadata
    full_text: str
    text_chunks: List[TextChunk]
    tables: List[TableData]
    images: List[ImageData]
    structure: DocumentStructure
    analytics: TextAnalytics