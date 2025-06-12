"""
PDF Processing Module for IntelliPDF
Handles PDF text extraction, metadata extraction, and document structure analysis
"""

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import camelot
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    file_size: int = 0

@dataclass
class TextChunk:
    """Text chunk with metadata"""
    text: str
    page_number: int
    chunk_type: str  # 'paragraph', 'header', 'table', 'list'
    font_info: Optional[Dict] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # x0, y0, x1, y1

@dataclass
class ProcessedDocument:
    """Complete processed document structure"""
    file_path: str
    doc_metadata: DocumentMetadata  # <--- CHANGED FROM 'metadata'
    text_chunks: List[TextChunk]
    full_text: str
    tables: List[pd.DataFrame]
    images: List[Dict]
    structure: Dict[str, Any]

class PDFProcessor:
    """Main PDF processing class"""
    
    def __init__(self, use_ocr: bool = True, ocr_language: str = 'eng'):
        """
        Initialize PDF processor
        
        Args:
            use_ocr: Whether to use OCR for scanned documents
            ocr_language: Language for OCR (default: English)
        """
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        
    def process_pdf(self, file_path: str) -> ProcessedDocument:
        """
        Main method to process a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ProcessedDocument: Complete processed document
        """
        try:
            logger.info(f"Processing PDF: {file_path}")
            
            # Open PDF document
            doc = fitz.open(file_path)
            
            # Extract metadata
            doc_metadata = self._extract_metadata(doc, file_path)  # <--- CHANGED
            
            # Extract text chunks with structure
            text_chunks = self._extract_text_chunks(doc)
            
            # Extract full text
            full_text = self._extract_full_text(text_chunks)
            
            # Extract tables
            tables = self._extract_tables(file_path)
            
            # Extract images
            images = self._extract_images(doc)
            
            # Analyze document structure
            structure = self._analyze_structure(text_chunks)
            
            doc.close()
            
            return ProcessedDocument(
                file_path=file_path,
                doc_metadata=doc_metadata,  # <--- CHANGED
                text_chunks=text_chunks,
                full_text=full_text,
                tables=tables,
                images=images,
                structure=structure
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
    
    def _extract_metadata(self, doc: fitz.Document, file_path: str) -> DocumentMetadata:
        """Extract PDF metadata"""
        try:
            meta = doc.metadata
            file_size = Path(file_path).stat().st_size
            
            return DocumentMetadata(
                title=meta.get('title', ''),
                author=meta.get('author', ''),
                subject=meta.get('subject', ''),
                creator=meta.get('creator', ''),
                producer=meta.get('producer', ''),
                creation_date=meta.get('creationDate', ''),
                modification_date=meta.get('modDate', ''),
                page_count=len(doc),
                file_size=file_size
            )
        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")
            return DocumentMetadata(page_count=len(doc))
    
    def _extract_text_chunks(self, doc: fitz.Document) -> List[TextChunk]:
        """Extract text chunks with formatting and structure information"""
        text_chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get text blocks with formatting
            blocks = page.get_text("dict")
            
            for block in blocks.get("blocks", []):
                if "lines" in block:  # Text block
                    chunk_text = ""
                    font_info = {}
                    bbox = block.get("bbox")
                    
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                            # Collect font information
                            if not font_info:
                                font_info = {
                                    "font": span.get("font", ""),
                                    "size": span.get("size", 0),
                                    "flags": span.get("flags", 0),
                                    "color": span.get("color", 0)
                                }
                        chunk_text += line_text + "\n"
                    
                    if chunk_text.strip():
                        # Determine chunk type based on font and content
                        chunk_type = self._classify_chunk_type(chunk_text, font_info)
                        
                        text_chunks.append(TextChunk(
                            text=chunk_text.strip(),
                            page_number=page_num + 1,
                            chunk_type=chunk_type,
                            font_info=font_info,
                            bbox=bbox
                        ))
            
            # Handle scanned pages with OCR
            if self.use_ocr and self._is_scanned_page(page):
                ocr_text = self._extract_ocr_text(page)
                if ocr_text:
                    text_chunks.append(TextChunk(
                        text=ocr_text,
                        page_number=page_num + 1,
                        chunk_type="ocr_text",
                        font_info=None,
                        bbox=None
                    ))
        
        return text_chunks
    
    def _classify_chunk_type(self, text: str, font_info: Dict) -> str:
        """Classify text chunk type based on content and formatting"""
        text_clean = text.strip()
        
        # Check for headers (larger font, short text, title case)
        if (font_info.get("size", 0) > 14 and 
            len(text_clean.split()) < 10 and 
            text_clean.istitle()):
            return "header"
        
        # Check for lists (starting with bullets or numbers)
        if (re.match(r'^\s*[•\-\*\d+\.]\s', text_clean) or 
            re.match(r'^\s*\([a-zA-Z0-9]+\)', text_clean)):
            return "list"
        
        # Check for table-like content
        if '\t' in text or len(re.findall(r'\s{3,}', text)) > 2:
            return "table"
        
        # Default to paragraph
        return "paragraph"
    
    def _is_scanned_page(self, page: fitz.Page) -> bool:
        """Check if page is scanned (image-based)"""
        text = page.get_text()
        images = page.get_images()
        
        # If very little text but has images, likely scanned
        return len(text.strip()) < 50 and len(images) > 0
    
    def _extract_ocr_text(self, page: fitz.Page) -> str:
        """Extract text using OCR for scanned pages"""
        try:
            # Convert page to image
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Apply OCR
            ocr_text = pytesseract.image_to_string(
                image, 
                lang=self.ocr_language,
                config='--psm 6'  # Assume uniform block of text
            )
            
            return ocr_text.strip()
            
        except Exception as e:
            logger.warning(f"OCR failed: {str(e)}")
            return ""
    
    def _extract_full_text(self, text_chunks: List[TextChunk]) -> str:
        """Combine all text chunks into full document text"""
        return "\n\n".join([chunk.text for chunk in text_chunks])
    
    def _extract_tables(self, file_path: str) -> List[pd.DataFrame]:
        """Extract tables using Camelot"""
        tables = []
        
        try:
            # Extract tables with Camelot
            camelot_tables = camelot.read_pdf(file_path, pages='all')
            
            for table in camelot_tables:
                if len(table.df) > 1:  # Skip single-row tables
                    tables.append(table.df)
                    
        except Exception as e:
            logger.warning(f"Table extraction failed: {str(e)}")
        
        return tables
    
    def _extract_images(self, doc: fitz.Document) -> List[Dict]:
        """Extract images and their metadata"""
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    images.append({
                        "page": page_num + 1,
                        "index": img_index,
                        "ext": base_image["ext"],
                        "width": base_image["width"],
                        "height": base_image["height"],
                        "colorspace": base_image["colorspace"],
                        "xref": xref
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to extract image: {str(e)}")
        
        return images
    
    def _analyze_structure(self, text_chunks: List[TextChunk]) -> Dict[str, Any]:
        """Analyze document structure"""
        structure = {
            "total_chunks": len(text_chunks),
            "chunk_types": {},
            "pages": {},
            "headers": [],
            "sections": []
        }
        
        # Count chunk types
        for chunk in text_chunks:
            chunk_type = chunk.chunk_type
            structure["chunk_types"][chunk_type] = structure["chunk_types"].get(chunk_type, 0) + 1
            
            # Group by pages
            page = chunk.page_number
            if page not in structure["pages"]:
                structure["pages"][page] = []
            structure["pages"][page].append({
                "type": chunk_type,
                "text_preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            })
            
            # Collect headers for structure
            if chunk_type == "header":
                structure["headers"].append({
                    "page": page,
                    "text": chunk.text,
                    "level": self._get_header_level(chunk)
                })
        
        return structure
    
    def _get_header_level(self, chunk: TextChunk) -> int:
        """Determine header level based on font size and style"""
        if chunk.font_info:
            font_size = chunk.font_info.get("size", 12)
            if font_size > 18:
                return 1
            elif font_size > 16:
                return 2
            elif font_size > 14:
                return 3
        return 4

# Utility functions for semantic chunking
class SemanticTextChunker:
    """Advanced text chunking for better semantic understanding"""
    
    def __init__(self, max_chunk_size: int = 1000, overlap_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    def chunk_by_semantics(self, text_chunks: List[TextChunk]) -> List[Dict]:
        """Create semantic chunks optimized for embeddings"""
        semantic_chunks = []
        current_chunk = ""
        current_metadata = []
        
        for chunk in text_chunks:
            # Add chunk to current semantic chunk
            if len(current_chunk) + len(chunk.text) < self.max_chunk_size:
                current_chunk += "\n" + chunk.text
                current_metadata.append({
                    "page": chunk.page_number,
                    "type": chunk.chunk_type,
                    "font_info": chunk.font_info
                })
            else:
                # Save current chunk and start new one
                if current_chunk:
                    semantic_chunks.append({
                        "text": current_chunk.strip(),
                        "metadata": current_metadata,
                        "chunk_id": len(semantic_chunks)
                    })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.overlap_size:] if len(current_chunk) > self.overlap_size else ""
                current_chunk = overlap_text + "\n" + chunk.text
                current_metadata = [{
                    "page": chunk.page_number,
                    "type": chunk.chunk_type,
                    "font_info": chunk.font_info
                }]
        
        # Add final chunk
        if current_chunk:
            semantic_chunks.append({
                "text": current_chunk.strip(),
                "metadata": current_metadata,
                "chunk_id": len(semantic_chunks)
            })
        
        return semantic_chunks

# Main function for testing
def main():
    """Test function"""
    processor = PDFProcessor()
    
    # Example usage
    try:
        result = processor.process_pdf("ok.pdf")
        print(f"Processed document: {result.doc_metadata.title}")
        print(f"Pages: {result.doc_metadata.page_count}")
        print(f"Text chunks: {len(result.text_chunks)}")
        print(f"Tables found: {len(result.tables)}")
        print(f"Images found: {len(result.images)}")
        
        # Create semantic chunks
        chunker = SemanticTextChunker()
        semantic_chunks = chunker.chunk_by_semantics(result.text_chunks)
        print(f"Semantic chunks: {len(semantic_chunks)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()