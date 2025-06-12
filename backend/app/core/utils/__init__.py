# IntelliPDF utilities package
from .text_processing import *
# app/core/utils/__init__.py

"""
Utilities package for IntelliPDF
Contains various utility modules for document processing, PDF handling, etc.
"""

# Import specific classes and functions to avoid circular imports
# DO NOT use wildcard imports (from .module import *) as they can cause circular imports

from .pdf_processor import (
    PDFProcessor, 
    SemanticTextChunker, 
    DocumentMetadata, 
    TextChunk, 
    ProcessedDocument,
    validate_pdf,
    create_pdf_processor,
    create_semantic_chunker
)

# Define what gets imported when someone does "from app.core.utils import *"
__all__ = [
    'PDFProcessor',
    'SemanticTextChunker', 
    'DocumentMetadata',
    'TextChunk',
    'ProcessedDocument',
    'validate_pdf',
    'create_pdf_processor',
    'create_semantic_chunker'
]
from .file_handlers import *