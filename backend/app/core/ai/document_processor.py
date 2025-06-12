"""
Document Processor for IntelliPDF
Pipeline for PDF processing: text extraction, layout, OCR, tables, chunking, and structure analysis.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.core.utils.pdf_processor import PDFProcessor, SemanticTextChunker

logger = logging.getLogger(__name__)

class DocumentProcessingPipeline:
    """
    Orchestrates the document processing pipeline using PDFProcessor and SemanticTextChunker.
    """

    def __init__(self, use_ocr: bool = True, ocr_language: str = "eng", max_chunk_size: int = 1000, overlap_size: int = 200):
        self.processor = PDFProcessor(use_ocr=use_ocr, ocr_language=ocr_language)
        self.chunker = SemanticTextChunker(max_chunk_size=max_chunk_size, overlap_size=overlap_size)

    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Main pipeline entry point.
        Args:
            file_path: Path to the PDF file
        Returns:
            Dict with processed document data and semantic chunks
        """
        logger.info(f"Starting pipeline for: {file_path}")

        try:
            processed_doc = self.processor.process_pdf(file_path)
            semantic_chunks = self.chunker.chunk_by_semantics(processed_doc.text_chunks)

            # Prepare a summary dict for API or downstream processing
            result = {
                "file_path": processed_doc.file_path,
                "metadata": self._serialize_metadata(processed_doc.metadata),
                "pages": processed_doc.metadata.page_count,
                "text_chunks": [self._serialize_text_chunk(chunk) for chunk in processed_doc.text_chunks],
                "full_text": processed_doc.full_text,
                "tables": [table.to_dict() if hasattr(table, 'to_dict') else table.to_json() for table in processed_doc.tables],
                "images": processed_doc.images,
                "structure": processed_doc.structure,
                "semantic_chunks": semantic_chunks,
            }

            logger.info(f"Completed pipeline for {file_path}: {len(result['semantic_chunks'])} semantic chunks")
            return result

        except Exception as e:
            logger.error(f"Error in document processing pipeline: {str(e)}")
            raise

    def _serialize_metadata(self, metadata) -> Dict[str, Any]:
        """Convert metadata dataclass to dict safely"""
        if hasattr(metadata, '__dict__'):
            return {k: v for k, v in metadata.__dict__.items()}
        else:
            return vars(metadata)

    def _serialize_text_chunk(self, chunk) -> Dict[str, Any]:
        """Convert text chunk dataclass to dict safely"""
        chunk_dict = {
            "text": chunk.text,
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type,
            "font_info": chunk.font_info,
            "bbox": chunk.bbox
        }
        return chunk_dict


# Example usage for testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} does not exist")
        sys.exit(1)
    
    try:
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process(pdf_path)
        
        print(f"Document: {result['metadata'].get('title', 'Unknown')}")
        print(f"Pages: {result['pages']}")
        print(f"Text Chunks: {len(result['text_chunks'])}")
        print(f"Tables: {len(result['tables'])}")
        print(f"Images: {len(result['images'])}")
        print(f"Semantic Chunks: {len(result['semantic_chunks'])}")
        
        # Print first few chunks as sample
        print("\nFirst 3 semantic chunks:")
        for i, chunk in enumerate(result['semantic_chunks'][:3]):
            print(f"Chunk {i+1}: {chunk['text'][:100]}...")
            
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        sys.exit(1)