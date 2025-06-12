"""
Document Processor for IntelliPDF
Pipeline for PDF processing: text extraction, layout, OCR, tables, chunking, and structure analysis.
"""

from ..utils.pdf_processor import PDFProcessor, SemanticTextChunker
from typing import Dict, Any, List
import logging

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

        processed_doc = self.processor.process_pdf(file_path)

        semantic_chunks = self.chunker.chunk_by_semantics(processed_doc.text_chunks)

        # Prepare a summary dict for API or downstream processing
        result = {
            "file_path": processed_doc.file_path,
            "metadata": vars(processed_doc.metadata),
            "pages": processed_doc.metadata.page_count,
            "text_chunks": [vars(chunk) for chunk in processed_doc.text_chunks],
            "full_text": processed_doc.full_text,
            "tables": [table.to_dict() for table in processed_doc.tables],
            "images": processed_doc.images,
            "structure": processed_doc.structure,
            "semantic_chunks": semantic_chunks,
        }

        logger.info(f"Completed pipeline for {file_path}: {len(result['semantic_chunks'])} semantic chunks")
        return result


# Example usage for testing (can be removed in production)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python document_processor.py <pdf_path>")
    else:
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process(sys.argv[1])
        print(f"Document: {result['metadata'].get('title')}")
        print(f"Pages: {result['pages']}")
        print(f"Text Chunks: {len(result['text_chunks'])}")
        print(f"Tables: {len(result['tables'])}")
        print(f"Images: {len(result['images'])}")
        print(f"Semantic Chunks: {len(result['semantic_chunks'])}")