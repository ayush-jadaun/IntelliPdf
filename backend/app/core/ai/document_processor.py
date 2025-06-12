"""
Document Processor for IntelliPDF
Integrated pipeline for PDF processing, text analytics, Gemini embeddings, knowledge graph construction,
and persistence of embeddings using pgvector vector store.

Author: IntelliPDF Team
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
from app.core.utils.text_processing import (
    preprocess_text_pipeline,
    extract_text_features,
    summarize_text_simple,
    extract_named_entities,
    extract_keywords_tfidf,
)
from app.core.ai.embeddings import embed_texts_with_gemini
from app.core.ai.knowledge_graph import build_knowledge_graph
from app.core.database.session import SessionLocal
from app.core.database.vector_store import (
    add_document_embedding,
    add_chunk_embedding,
    bulk_add_chunk_embeddings,
)

logger = logging.getLogger(__name__)

class DocumentProcessingPipeline:
    """
    Orchestrates the document processing pipeline using PDFProcessor,
    SemanticTextChunker, text analytics, Gemini embeddings, KG, and vector store.
    """

    def __init__(
        self,
        use_ocr: bool = True,
        ocr_language: str = "eng",
        max_chunk_size: int = 1000,
        overlap_size: int = 200,
    ):
        self.processor = PDFProcessor(use_ocr=use_ocr, ocr_language=ocr_language)
        self.chunker = SemanticTextChunker(
            max_chunk_size=max_chunk_size, overlap_size=overlap_size
        )

    def process(self, file_path: str) -> Dict[str, Any]:
        """
        Main pipeline entry point.
        Args:
            file_path: Path to the PDF file
        Returns:
            Dict with processed document data, analytics, semantic embeddings, and knowledge graph.
        """
        logger.info(f"Starting pipeline for: {file_path}")
        db = SessionLocal()

        try:
            # 1. PDF Extraction
            processed_doc = self.processor.process_pdf(file_path)

            # 2. Semantic Chunking
            semantic_chunks = self.chunker.chunk_by_semantics(processed_doc.text_chunks)

            # 3. Text Analytics (full document level)
            full_text = processed_doc.full_text
            processed_text, preprocess_meta = preprocess_text_pipeline(full_text)
            analytics = extract_text_features(processed_text)
            summary = summarize_text_simple(processed_text)
            entities = extract_named_entities(processed_text)
            keywords = extract_keywords_tfidf([processed_text])

            # 4. Embeddings (chunk level)
            semantic_chunks_with_embeddings = embed_texts_with_gemini(semantic_chunks)

            # 4b. Persist document and chunk embeddings to the database
            # Store document-level embedding if desired (e.g., average of chunk embeddings, or dedicated embedding)
            # Here we use the average of chunk embeddings as a simple example
            if semantic_chunks_with_embeddings and "embedding" in semantic_chunks_with_embeddings[0]:
                avg_embedding = [
                    float(sum(x) / len(semantic_chunks_with_embeddings))
                    for x in zip(*(ch["embedding"] for ch in semantic_chunks_with_embeddings))
                ]
            else:
                avg_embedding = None

            # Add document to DB and get its ID
            doc_metadata_dict = self._serialize_metadata(processed_doc.metadata)
            doc_id = add_document_embedding(
                db,
                title=doc_metadata_dict.get("title") or os.path.basename(file_path),
                embedding=avg_embedding,
                file_path=file_path,
                metadata=doc_metadata_dict,
            )

            # Prepare chunks for bulk insert
            chunk_db_objs = []
            for chunk, chunk_embed in zip(semantic_chunks, semantic_chunks_with_embeddings):
                chunk_db_objs.append({
                    "document_id": doc_id,
                    "text": chunk["text"],
                    "embedding": chunk_embed["embedding"],
                    "page_number": chunk["metadata"][0].get("page") if chunk.get("metadata") else None,
                    "chunk_type": chunk["metadata"][0].get("type") if chunk.get("metadata") else None,
                    "metadata": chunk["metadata"] if chunk.get("metadata") else {},
                })
            # Bulk insert all chunk embeddings
            if chunk_db_objs:
                bulk_add_chunk_embeddings(db, chunk_db_objs)

            # 5. Knowledge Graph Construction
            knowledge_graph = build_knowledge_graph(
                entities=entities,
                keywords=keywords,
                doc_metadata=processed_doc.metadata,
                doc_id=doc_id,
            )

            # 6. Prepare result for API or downstream processing
            result = {
                "file_path": processed_doc.file_path,
                "metadata": doc_metadata_dict,
                "pages": processed_doc.metadata.page_count,
                "text_chunks": [
                    self._serialize_text_chunk(chunk)
                    for chunk in processed_doc.text_chunks
                ],
                "full_text": processed_text,
                "tables": [
                    table.to_dict() if hasattr(table, "to_dict") else table.to_json()
                    for table in processed_doc.tables
                ],
                "images": processed_doc.images,
                "structure": processed_doc.structure,
                "analytics": {
                    "preprocess_meta": preprocess_meta,
                    "features": analytics,
                    "summary": summary,
                    "entities": entities,
                    "keywords": keywords,
                },
                "semantic_chunks": semantic_chunks_with_embeddings,
                "knowledge_graph": knowledge_graph,
                "document_id": doc_id,  # Useful for downstream queries
            }

            logger.info(
                f"Completed pipeline for {file_path}: {len(result['semantic_chunks'])} semantic chunks"
            )
            return result

        except Exception as e:
            logger.error(f"Error in document processing pipeline: {str(e)}")
            raise
        finally:
            db.close()

    def _serialize_metadata(self, metadata) -> Dict[str, Any]:
        """Convert metadata dataclass to dict safely."""
        if hasattr(metadata, "__dict__"):
            return {k: v for k, v in metadata.__dict__.items()}
        else:
            return vars(metadata)

    def _serialize_text_chunk(self, chunk) -> Dict[str, Any]:
        """Convert text chunk dataclass to dict safely."""
        chunk_dict = {
            "text": chunk.text,
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type,
            "font_info": chunk.font_info,
            "bbox": chunk.bbox,
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

        # Print analytics
        print("\nDocument Analytics:")
        print(f"- Summary: {result['analytics']['summary']}")
        print(f"- Entities: {result['analytics']['entities']}")
        print(f"- Top Keywords: {result['analytics']['keywords']}")
        print(f"- Stats: {result['analytics']['features']['stats']}")

        # Print first few semantic chunks as sample
        print("\nFirst 3 semantic chunks (with embeddings):")
        for i, chunk in enumerate(result["semantic_chunks"][:3]):
            print(f"Chunk {i+1}: {chunk['text'][:100]}...")
            print(f"Embedding (first 5 values): {chunk['embedding'][:5]}")

        # Print knowledge graph sample
        print("\nKnowledge Graph Nodes/Edges Sample:")
        print("Nodes:", result["knowledge_graph"]["nodes"][:2])
        print("Edges:", result["knowledge_graph"]["edges"][:2])

    except Exception as e:
        print(f"Error processing document: {str(e)}")
        sys.exit(1)