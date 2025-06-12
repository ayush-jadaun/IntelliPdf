# app/core/utils/pdf_processor.py

"""
PDF Processing utilities for IntelliPDF
Handles PDF text extraction, OCR, semantic chunking, and metadata extraction.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# Third-party imports
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    title: str = ""
    author: str = ""
    subject: str = ""
    creator: str = ""
    producer: str = ""
    creation_date: str = ""
    modification_date: str = ""
    page_count: int = 0
    file_size: int = 0


@dataclass
class TextChunk:
    """Text chunk with metadata"""
    text: str
    page_number: int
    chunk_id: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    font_info: Optional[Dict[str, Any]] = None


@dataclass
class ProcessedDocument:
    """Complete processed document structure"""
    full_text: str
    text_chunks: List[TextChunk]
    doc_metadata: DocumentMetadata
    structure: Dict[str, Any]
    tables: List[pd.DataFrame]
    images: List[Dict[str, Any]]


class PDFProcessor:
    """Advanced PDF processing with OCR support"""
    
    def __init__(self, use_ocr: bool = True, ocr_language: str = 'eng'):
        """
        Initialize PDF processor
        
        Args:
            use_ocr: Whether to use OCR for scanned documents
            ocr_language: Tesseract OCR language code
        """
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        
        # Test OCR availability
        if self.use_ocr:
            try:
                pytesseract.get_tesseract_version()
                logger.info("OCR (Tesseract) is available")
            except Exception as e:
                logger.warning(f"OCR not available: {e}")
                self.use_ocr = False
    
    def process_pdf(self, file_path: str) -> ProcessedDocument:
        """
        Process PDF file and extract all content
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            ProcessedDocument with all extracted content
        """
        try:
            # Open PDF document
            doc = fitz.open(file_path)
            
            # Extract metadata
            metadata = self._extract_metadata(doc, file_path)
            
            # Extract text chunks
            text_chunks = self._extract_text_chunks(doc)
            
            # Extract full text
            full_text = "\n".join([chunk.text for chunk in text_chunks])
            
            # Extract structure information
            structure = self._extract_structure(doc)
            
            # Extract tables
            tables = self._extract_tables(doc)
            
            # Extract images
            images = self._extract_images(doc)
            
            doc.close()
            
            return ProcessedDocument(
                full_text=full_text,
                text_chunks=text_chunks,
                doc_metadata=metadata,
                structure=structure,
                tables=tables,
                images=images
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    def _extract_metadata(self, doc, file_path: str) -> DocumentMetadata:
        """Extract document metadata"""
        try:
            meta = doc.metadata
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            return DocumentMetadata(
                title=meta.get('title', '') or '',
                author=meta.get('author', '') or '',
                subject=meta.get('subject', '') or '',
                creator=meta.get('creator', '') or '',
                producer=meta.get('producer', '') or '',
                creation_date=meta.get('creationDate', '') or '',
                modification_date=meta.get('modDate', '') or '',
                page_count=len(doc),
                file_size=file_size
            )
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return DocumentMetadata()
    
    def _extract_text_chunks(self, doc) -> List[TextChunk]:
        """Extract text chunks from PDF"""
        text_chunks = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Try to extract text directly
            text_dict = page.get_text("dict")
            page_text = ""
            
            # Extract text from blocks
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            page_text += span["text"]
                        page_text += "\n"
            
            # If no text found and OCR is enabled, try OCR
            if not page_text.strip() and self.use_ocr:
                page_text = self._ocr_page(page)
            
            if page_text.strip():
                chunk = TextChunk(
                    text=page_text.strip(),
                    page_number=page_num + 1,
                    chunk_id=str(uuid.uuid4()),
                    bbox=page.rect,
                    font_info=self._extract_font_info(text_dict)
                )
                text_chunks.append(chunk)
        
        return text_chunks
    
    def _ocr_page(self, page) -> str:
        """Perform OCR on a page"""
        try:
            # Convert page to image
            pix = page.get_pixmap()
            img_data = pix.tobytes("ppm")
            image = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=self.ocr_language)
            return text
        except Exception as e:
            logger.warning(f"OCR failed for page: {e}")
            return ""
    
    def _extract_font_info(self, text_dict) -> Dict[str, Any]:
        """Extract font information from text dictionary"""
        fonts = {}
        try:
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font = span.get("font", "")
                            size = span.get("size", 0)
                            if font:
                                fonts[font] = fonts.get(font, 0) + 1
        except Exception as e:
            logger.warning(f"Error extracting font info: {e}")
        
        return fonts
    
    def _extract_structure(self, doc) -> Dict[str, Any]:
        """Extract document structure information"""
        structure = {
            "page_count": len(doc),
            "has_toc": False,
            "toc": [],
            "page_sizes": []
        }
        
        try:
            # Extract table of contents
            toc = doc.get_toc()
            if toc:
                structure["has_toc"] = True
                structure["toc"] = toc
            
            # Extract page sizes
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                structure["page_sizes"].append({
                    "page": page_num + 1,
                    "width": page.rect.width,
                    "height": page.rect.height
                })
        
        except Exception as e:
            logger.warning(f"Error extracting structure: {e}")
        
        return structure
    
    def _extract_tables(self, doc) -> List[pd.DataFrame]:
        """Extract tables from PDF"""
        tables = []
        
        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Try to find tables using PyMuPDF
                tabs = page.find_tables()
                for tab in tabs:
                    try:
                        table_data = tab.extract()
                        if table_data:
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                            tables.append(df)
                    except Exception as e:
                        logger.warning(f"Error extracting table on page {page_num + 1}: {e}")
        
        except Exception as e:
            logger.warning(f"Error extracting tables: {e}")
        
        return tables
    
    def _extract_images(self, doc) -> List[Dict[str, Any]]:
        """Extract images from PDF"""
        images = []
        
        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = {
                                "page": page_num + 1,
                                "index": img_index,
                                "xref": xref,
                                "width": pix.width,
                                "height": pix.height,
                                "colorspace": pix.colorspace.name if pix.colorspace else "unknown",
                                "has_alpha": bool(pix.alpha)
                            }
                            images.append(img_data)
                        
                        pix = None
                    except Exception as e:
                        logger.warning(f"Error extracting image on page {page_num + 1}: {e}")
        
        except Exception as e:
            logger.warning(f"Error extracting images: {e}")
        
        return images


class SemanticTextChunker:
    """Advanced semantic text chunking using embeddings"""
    
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 200, 
                 similarity_threshold: float = 0.5):
        """
        Initialize semantic chunker
        
        Args:
            max_chunk_size: Maximum characters per chunk
            overlap_size: Overlap between chunks
            similarity_threshold: Similarity threshold for semantic boundaries
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.similarity_threshold = similarity_threshold
        
        # Initialize sentence transformer model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model")
        except Exception as e:
            logger.warning(f"Could not load sentence transformer: {e}")
            self.model = None
    
    def chunk_by_semantics(self, text_chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        """
        Perform semantic chunking on text chunks
        
        Args:
            text_chunks: List of TextChunk objects
            
        Returns:
            List of semantic chunks with metadata
        """
        semantic_chunks = []
        
        for text_chunk in text_chunks:
            # Split text into sentences
            sentences = self._split_into_sentences(text_chunk.text)
            
            if not sentences:
                continue
            
            if self.model is None:
                # Fallback to simple chunking
                chunks = self._simple_chunk(text_chunk.text)
            else:
                # Use semantic chunking
                chunks = self._semantic_chunk(sentences, text_chunk)
            
            semantic_chunks.extend(chunks)
        
        return semantic_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _semantic_chunk(self, sentences: List[str], original_chunk: TextChunk) -> List[Dict[str, Any]]:
        """Perform semantic chunking using embeddings"""
        if not sentences:
            return []
        
        try:
            # Generate embeddings for sentences
            embeddings = self.model.encode(sentences)
            
            # Find semantic boundaries
            boundaries = self._find_semantic_boundaries(embeddings)
            
            # Create chunks based on boundaries
            chunks = []
            current_chunk = ""
            current_sentences = []
            
            for i, sentence in enumerate(sentences):
                current_chunk += sentence + ". "
                current_sentences.append(sentence)
                
                # Check if we should create a new chunk
                if (i in boundaries or 
                    len(current_chunk) >= self.max_chunk_size or 
                    i == len(sentences) - 1):
                    
                    if current_chunk.strip():
                        chunk_data = {
                            "text": current_chunk.strip(),
                            "chunk_id": str(uuid.uuid4()),
                            "metadata": {
                                "original_page": original_chunk.page_number,
                                "original_chunk_id": original_chunk.chunk_id,
                                "sentence_count": len(current_sentences),
                                "char_count": len(current_chunk),
                                "chunk_type": "semantic"
                            }
                        }
                        chunks.append(chunk_data)
                    
                    # Start new chunk with overlap
                    if i < len(sentences) - 1:
                        overlap_sentences = current_sentences[-self._calculate_overlap_sentences(current_sentences):]
                        current_chunk = ". ".join(overlap_sentences) + ". " if overlap_sentences else ""
                        current_sentences = overlap_sentences
                    else:
                        current_chunk = ""
                        current_sentences = []
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Semantic chunking failed, falling back to simple chunking: {e}")
            return self._simple_chunk(original_chunk.text, original_chunk)
    
    def _find_semantic_boundaries(self, embeddings) -> List[int]:
        """Find semantic boundaries using cosine similarity"""
        boundaries = []
        
        if len(embeddings) < 2:
            return boundaries
        
        # Calculate similarities between consecutive sentences
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            
            if similarity < self.similarity_threshold:
                boundaries.append(i)
        
        return boundaries
    
    def _calculate_overlap_sentences(self, sentences: List[str]) -> int:
        """Calculate number of sentences for overlap"""
        total_chars = sum(len(s) for s in sentences)
        if total_chars == 0:
            return 0
        
        overlap_chars = min(self.overlap_size, total_chars // 2)
        
        # Find how many sentences make up the overlap
        char_count = 0
        for i in range(len(sentences) - 1, -1, -1):
            char_count += len(sentences[i])
            if char_count >= overlap_chars:
                return len(sentences) - i
        
        return len(sentences)
    
    def _simple_chunk(self, text: str, original_chunk: Optional[TextChunk] = None) -> List[Dict[str, Any]]:
        """Fallback simple chunking method"""
        chunks = []
        
        # Split text into chunks of max_chunk_size
        for i in range(0, len(text), self.max_chunk_size - self.overlap_size):
            chunk_text = text[i:i + self.max_chunk_size]
            
            if chunk_text.strip():
                chunk_data = {
                    "text": chunk_text.strip(),
                    "chunk_id": str(uuid.uuid4()),
                    "metadata": {
                        "original_page": original_chunk.page_number if original_chunk else 1,
                        "original_chunk_id": original_chunk.chunk_id if original_chunk else str(uuid.uuid4()),
                        "char_count": len(chunk_text),
                        "chunk_type": "simple",
                        "start_pos": i,
                        "end_pos": min(i + self.max_chunk_size, len(text))
                    }
                }
                chunks.append(chunk_data)
        
        return chunks


# Utility functions
def validate_pdf(file_path: str) -> bool:
    """Validate if file is a readable PDF"""
    try:
        if not os.path.exists(file_path):
            return False
        
        if not file_path.lower().endswith('.pdf'):
            return False
        
        # Try to open with PyMuPDF
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
        
        return page_count > 0
        
    except Exception as e:
        logger.error(f"PDF validation failed: {e}")
        return False


def create_pdf_processor(config: Optional[Dict[str, Any]] = None) -> PDFProcessor:
    """Factory function to create PDF processor with configuration"""
    if config is None:
        config = {}
    
    return PDFProcessor(
        use_ocr=config.get('use_ocr', True),
        ocr_language=config.get('ocr_language', 'eng')
    )


def create_semantic_chunker(config: Optional[Dict[str, Any]] = None) -> SemanticTextChunker:
    """Factory function to create semantic chunker with configuration"""
    if config is None:
        config = {}
    
    return SemanticTextChunker(
        max_chunk_size=config.get('max_chunk_size', 1000),
        overlap_size=config.get('overlap_size', 200),
        similarity_threshold=config.get('similarity_threshold', 0.5)
    )


# Test function
def main():
    """Test the PDF processor"""
    processor = PDFProcessor()
    
    # Test file
    test_file = "test.pdf"
    
    if validate_pdf(test_file):
        try:
            result = processor.process_pdf(test_file)
            print(f"Successfully processed: {result.doc_metadata.title}")
            print(f"Pages: {result.doc_metadata.page_count}")
            print(f"Text chunks: {len(result.text_chunks)}")
            print(f"Tables: {len(result.tables)}")
            print(f"Images: {len(result.images)}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Invalid PDF: {test_file}")


if __name__ == "__main__":
    main()