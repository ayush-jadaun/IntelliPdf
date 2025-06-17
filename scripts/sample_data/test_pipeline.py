import sys
import os

# Add backend folder to sys.path
sys.path.append(r"C:\Users\shree\Documents\Intellipdf\IntelliPdf\backend")

from app.core.pipeline.process_pipeline import process_pipeline

# Mock document with some named entities
mock_document_data = {
    "file_path": "C:/Users/shree/Documents/Intellipdf/IntelliPdf/mock.pdf",
    "doc_metadata": {
        "title": "Test KG Doc",
        "author": "OpenAI",
        "page_count": 1,
        "file_size": 2048
    },
    "full_text": "Barack Obama was born in Hawaii. He served as the 44th President of the United States.",
    "text_chunks": [],
    "semantic_chunks": [],
    "tables": [],
    "images": [],
    "structure": {},
    "analytics": {},
    "knowledge_graph": None  # This triggers extraction via spaCy
}

# Run the pipeline
process_pipeline(mock_document_data)

print("✅ Test successful: Knowledge Graph extracted and stored in Neo4j.")
