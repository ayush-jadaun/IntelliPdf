from pydantic import BaseModel, Field, ValidationError, validator
from typing import List, Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    page_count: Optional[int] = 0
    file_size: Optional[int] = 0

class TextChunk(BaseModel):
    text: str
    page_number: int
    chunk_type: str
    font_info: Optional[Dict[str, Any]] = None
    bbox: Optional[List[float]] = None

class SemanticChunk(BaseModel):
    text: str
    embedding: Optional[List[float]] = None
    chunk_id: Optional[str] = None
    semantic_type: Optional[str] = None

class TableData(BaseModel):
    page_number: Optional[int] = Field(None, description="Page number where the table appears")
    data: Optional[List[List[str]]] = Field(default_factory=list, description="Table data as a list of rows")
    
    @validator('data', pre=True)
    def validate_table_data(cls, v):
        if v is None:
            return []
        if hasattr(v, 'to_dict'):
            try:
                df_dict = v.to_dict('records')
                rows = []
                for record in df_dict:
                    row = [str(value) for value in record.values()]
                    if any(cell.strip() for cell in row):
                        rows.append(row)
                return rows
            except Exception as e:
                logger.warning(f"Could not convert DataFrame to list: {e}")
                return []
        if isinstance(v, dict):
            logger.warning(f"Received dict table data, converting: {v}")
            rows = []
            for key, value in v.items():
                if isinstance(value, dict):
                    row_data = list(value.values())
                    if row_data and any(str(cell).strip() for cell in row_data):
                        rows.append([str(cell) for cell in row_data])
            return rows if rows else []
        if isinstance(v, list):
            return [[str(cell) for cell in row] if isinstance(row, list) else [str(row)] for row in v]
        return []

class ImageData(BaseModel):
    page_number: Optional[int] = Field(None, description="Page number where the image appears")
    ext: Optional[str] = Field(None, description="Image file extension")
    width: Optional[int] = Field(None, description="Image width in pixels")
    height: Optional[int] = Field(None, description="Image height in pixels")

class DocumentStructure(BaseModel):
    total_chunks: Optional[int] = 0
    chunk_types: Optional[Dict[str, int]] = Field(default_factory=dict)
    headers: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    pages: Optional[Dict[int, List[Dict[str, Any]]]] = Field(default_factory=dict)

class TextAnalytics(BaseModel):
    word_count: Optional[int] = 0
    sentence_count: Optional[int] = 0
    keywords: Optional[List[str]] = Field(default_factory=list)
    summary: Optional[str] = None
    entities: Optional[List[str]] = Field(default_factory=list)

class KnowledgeGraph(BaseModel):
    nodes: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    edges: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ProcessedDocumentResponse(BaseModel):
    file_path: str
    doc_metadata: DocumentMetadata
    full_text: Optional[str] = ""
    text_chunks: Optional[List[TextChunk]] = Field(default_factory=list)
    semantic_chunks: Optional[List[SemanticChunk]] = Field(default_factory=list)
    tables: Optional[List[TableData]] = Field(default_factory=list)
    images: Optional[List[ImageData]] = Field(default_factory=list)
    structure: Optional[DocumentStructure] = None
    analytics: Optional[TextAnalytics] = None
    knowledge_graph: Optional[KnowledgeGraph] = None
    document_id: Optional[str] = None
    
    class Config:
        extra = "allow"
        validate_assignment = True

def safe_table_data(raw_table: Any) -> Optional[TableData]:
    try:
        if hasattr(raw_table, 'to_dict'):
            logger.info("Processing DataFrame table")
            if raw_table.empty:
                logger.info("Skipping empty DataFrame")
                return None
            table_data = TableData(
                page_number=None,
                data=raw_table
            )
            if table_data.data and any(any(str(cell).strip() for cell in row) for row in table_data.data):
                return table_data
            else:
                logger.info("Skipping empty table after conversion")
                return None
        elif isinstance(raw_table, dict):
            logger.info("Processing dict table")
            table_data = TableData(**raw_table)
            if table_data.data and any(any(str(cell).strip() for cell in row) for row in table_data.data):
                return table_data
            else:
                logger.info("Skipping empty table")
                return None
        else:
            logger.warning(f"Expected dict or DataFrame for table data, got {type(raw_table)}")
            return None
    except ValidationError as e:
        logger.error(f"Table validation error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in table processing: {e}")
        return None

def safe_image_data(raw_image: Dict[str, Any]) -> Optional[ImageData]:
    try:
        return ImageData(**raw_image)
    except ValidationError as e:
        logger.error(f"Image validation error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in image processing: {e}")
        return None

def build_tables_safe(raw_tables: List[Any]) -> List[TableData]:
    if not raw_tables:
        return []
    valid_tables = []
    for i, raw_table in enumerate(raw_tables):
        try:
            logger.info(f"Processing table {i}, type: {type(raw_table)}")
            table = safe_table_data(raw_table)
            if table:
                valid_tables.append(table)
                logger.info(f"Successfully processed table {i}")
            else:
                logger.info(f"Skipped table {i} (empty or invalid)")
        except Exception as e:
            logger.error(f"Error processing table {i}: {e}")
    logger.info(f"Successfully processed {len(valid_tables)} out of {len(raw_tables)} tables")
    return valid_tables

def build_images_safe(raw_images: List[Dict[str, Any]]) -> List[ImageData]:
    if not raw_images:
        return []
    valid_images = []
    for i, raw_image in enumerate(raw_images):
        try:
            image = safe_image_data(raw_image)
            if image:
                valid_images.append(image)
        except Exception as e:
            logger.error(f"Error processing image {i}: {e}")
    return valid_images