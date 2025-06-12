import os
import pytest
from backend.app.core.utils.pdf_processor import PDFProcessor, SemanticTextChunker

SAMPLE_PDF = "ok.pdf"  # Put a real small PDF here for testing

@pytest.fixture
def processor():
    return PDFProcessor(use_ocr=False)

def test_metadata_extraction(processor):
    doc = processor.process_pdf(SAMPLE_PDF)
    assert doc.metadata.page_count > 0
    assert isinstance(doc.metadata.title, str)

def test_text_extraction(processor):
    doc = processor.process_pdf(SAMPLE_PDF)
    assert len(doc.text_chunks) > 0
    assert any(chunk.text for chunk in doc.text_chunks)

def test_table_extraction(processor):
    doc = processor.process_pdf(SAMPLE_PDF)
    assert isinstance(doc.tables, list)

def test_semantic_chunking(processor):
    doc = processor.process_pdf(SAMPLE_PDF)
    chunker = SemanticTextChunker()
    chunks = chunker.chunk_by_semantics(doc.text_chunks)
    assert isinstance(chunks, list)
    assert chunks  # Not empty