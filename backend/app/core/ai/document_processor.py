"""
Document Processor for IntelliPDF - FIXED VERSION
Integrated pipeline for PDF processing, text analytics, Gemini embeddings, knowledge graph construction,
and persistence of embeddings using pgvector vector store.

Author: IntelliPDF Team
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import traceback

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
from app.core.ai.embeddings import SentenceTransformerEmbedder
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
        db = None

        try:
            # 1. PDF Extraction
            logger.info("Step 1: PDF extraction")
            processed_doc = self.processor.process_pdf(file_path)
            logger.info(f"Extracted {len(processed_doc.text_chunks)} text chunks")

            # 2. Semantic Chunking
            logger.info("Step 2: Semantic chunking")
            semantic_chunks = self.chunker.chunk_by_semantics(processed_doc.text_chunks)
            logger.info(f"Created {len(semantic_chunks)} semantic chunks")

            # 3. Text Analytics (full document level)
            logger.info("Step 3: Text analytics")
            full_text = processed_doc.full_text

            # Safe text processing with fallbacks
            try:
                processed_text, preprocess_meta = preprocess_text_pipeline(full_text)
                analytics = extract_text_features(processed_text)
                summary = summarize_text_simple(processed_text)
                entities = extract_named_entities(processed_text)
                keywords = extract_keywords_tfidf([processed_text])
            except Exception as e:
                logger.warning(f"Text analytics failed: {e}")
                processed_text = full_text
                preprocess_meta = {}
                analytics = {}
                summary = "Summary generation failed"
                entities = []
                keywords = []

            # 4. Embeddings (chunk level)
            logger.info("Step 4: Creating embeddings")
            try:
                embedder = SentenceTransformerEmbedder()
                for chunk in semantic_chunks:
                    chunk["embedding"] = embedder.embed_texts([chunk["text"]])[0]
                logger.info(f"Generated embeddings for {len(semantic_chunks)} chunks")
                semantic_chunks_with_embeddings = semantic_chunks
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")
                semantic_chunks_with_embeddings = []
                for chunk in semantic_chunks:
                    chunk_copy = chunk.copy()
                    chunk_copy["embedding"] = None
                    semantic_chunks_with_embeddings.append(chunk_copy)

            # 5. Database operations (with safe handling)
            logger.info("Step 5: Database operations")
            doc_id = None
            avg_embedding = None

            try:
                db = SessionLocal()

                # Calculate average embedding if embeddings exist
                valid_embeddings = [ch.get("embedding") for ch in semantic_chunks_with_embeddings
                                  if ch.get("embedding") is not None]

                if valid_embeddings:
                    avg_embedding = [
                        float(sum(x) / len(valid_embeddings))
                        for x in zip(*valid_embeddings)
                    ]

                # Add document to DB and get its ID
                doc_metadata_dict = self._serialize_metadata(processed_doc.doc_metadata)
                doc_id = add_document_embedding(
                    db,
                    title=doc_metadata_dict.get("title") or os.path.basename(file_path),
                    embedding=avg_embedding,
                    file_path=file_path,
                    doc_metadata=doc_metadata_dict,
                )
                logger.info(f"Added document to DB with ID: {doc_id}")

                # Prepare chunks for bulk insert
                chunk_db_objs = []
                for i, chunk in enumerate(semantic_chunks_with_embeddings):
                    try:
                        chunk_metadata = chunk.get("metadata", {})
                        chunk_db_objs.append({
                            "document_id": doc_id,
                            "text": chunk["text"],
                            "embedding": chunk.get("embedding"),
                            "page_number": chunk_metadata.get("original_page", 1),
                            "chunk_type": chunk_metadata.get("chunk_type", "semantic"),
                            "doc_metadata": chunk_metadata,
                        })
                    except Exception as e:
                        logger.warning(f"Error preparing chunk {i}: {e}")
                        continue

                # Bulk insert all chunk embeddings
                if chunk_db_objs:
                    bulk_add_chunk_embeddings(db, chunk_db_objs)
                    logger.info(f"Added {len(chunk_db_objs)} chunks to DB")

            except Exception as e:
                logger.error(f"Database operations failed: {e}")
                logger.error(traceback.format_exc())
                # Continue without database operations

            finally:
                if db:
                    try:
                        db.close()
                    except:
                        pass

            # 6. Knowledge Graph Construction (with safe handling)
            logger.info("Step 6: Knowledge graph construction")
            try:
                # Always serialize metadata to dict before passing to KG builder
                doc_metadata_dict = self._serialize_metadata(processed_doc.doc_metadata)
                knowledge_graph = build_knowledge_graph(
                    entities=entities,
                    keywords=keywords,
                    doc_metadata=doc_metadata_dict,
                    doc_id=doc_id,
                )
            except Exception as e:
                logger.warning(f"Knowledge graph construction failed: {e}")
                knowledge_graph = {"nodes": [], "edges": [], "error": str(e)}

            # 7. Prepare result for API or downstream processing
            logger.info("Step 7: Preparing result")

            # Safe table serialization
            def ensure_str_keys(d):
                if not isinstance(d, dict): return d
                return {str(k): v for k, v in d.items()}

            serialized_tables = []
            for table in processed_doc.tables:
                try:
                    if hasattr(table, "page_number") and hasattr(table, "data"):
                        serialized_tables.append({
                            "page_number": table.page_number,
                            "data": table.data
                        })
                    elif hasattr(table, "to_dict"):
                        tdict = table.to_dict()
                        serialized_tables.append(ensure_str_keys(tdict))
                    elif hasattr(table, "to_json"):
                        serialized_tables.append(table.to_json())
                    elif isinstance(table, dict):
                        serialized_tables.append(ensure_str_keys(table))
                    else:
                        serialized_tables.append(str(table))
                except Exception as e:
                    logger.warning(f"Error serializing table: {e}")
                    serialized_tables.append({"error": "Serialization failed"})

            # Safe image serialization
            serialized_images = []
            for img in processed_doc.images or []:
                try:
                    if hasattr(img, "page_number") and hasattr(img, "ext") and hasattr(img, "width") and hasattr(img, "height"):
                        serialized_images.append({
                            "page_number": img.page_number,
                            "ext": img.ext,
                            "width": img.width,
                            "height": img.height,
                        })
                    elif isinstance(img, dict):
                        img_obj = {
                            "page_number": img.get("page_number"),
                            "ext": img.get("ext"),
                            "width": img.get("width"),
                            "height": img.get("height"),
                        }
                        serialized_images.append(img_obj)
                    else:
                        serialized_images.append(str(img))
                except Exception as e:
                    logger.warning(f"Error serializing image: {e}")
                    serialized_images.append({"error": "Serialization failed"})

            # Safe structure serialization
            structure = {}
            try:
                s = processed_doc.structure
                if s and all(hasattr(s, attr) for attr in ("total_chunks", "chunk_types", "headers", "pages")):
                    structure = {
                        "total_chunks": s.total_chunks,
                        "chunk_types": s.chunk_types,
                        "headers": s.headers,
                        "pages": s.pages,
                    }
                elif isinstance(s, dict):
                    structure = s
                else:
                    structure = {}
            except Exception as e:
                logger.warning(f"Error serializing structure: {e}")
                structure = {}

            # Safe analytics serialization
            analytics_obj = {}
            try:
                if isinstance(analytics, dict):
                    analytics_obj = {
                        "word_count": analytics.get("word_count", 0),
                        "sentence_count": analytics.get("sentence_count", 0),
                        "keywords": keywords if isinstance(keywords, list) else [],
                        "summary": summary,
                        "entities": entities if isinstance(entities, list) else [],
                    }
                else:
                    analytics_obj = {
                        "word_count": 0,
                        "sentence_count": 0,
                        "keywords": [],
                        "summary": summary,
                        "entities": [],
                    }
            except Exception as e:
                logger.warning(f"Error serializing analytics: {e}")
                analytics_obj = {
                    "word_count": 0,
                    "sentence_count": 0,
                    "keywords": [],
                    "summary": summary,
                    "entities": [],
                }

            result = {
                "file_path": file_path,
                "doc_metadata": self._serialize_metadata(processed_doc.doc_metadata),
                "full_text": processed_text,
                "text_chunks": [
                    self._serialize_text_chunk(chunk)
                    for chunk in processed_doc.text_chunks
                ],
                "tables": serialized_tables,
                "images": serialized_images,
                "structure": structure,
                "analytics": analytics_obj,
                "semantic_chunks": semantic_chunks_with_embeddings,
                "knowledge_graph": knowledge_graph,
                "document_id": str(doc_id) if doc_id is not None else None,
            }

            logger.info(
                f"Completed pipeline for {file_path}: {len(result['semantic_chunks'])} semantic chunks"
            )
            return result

        except Exception as e:
            logger.error(f"Error in document processing pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        finally:
            if db:
                try:
                    db.close()
                except:
                    pass

    def _serialize_metadata(self, metadata) -> Dict[str, Any]:
        """Convert metadata dataclass to dict safely."""
        try:
            if hasattr(metadata, "__dict__"):
                return {str(k): v for k, v in metadata.__dict__.items()}
            elif hasattr(metadata, "_asdict"):
                return {str(k): v for k, v in metadata._asdict().items()}
            elif isinstance(metadata, dict):
                return {str(k): v for k, v in metadata.items()}
            else:
                return {str(k): v for k, v in vars(metadata).items()}
        except Exception as e:
            logger.warning(f"Error serializing metadata: {e}")
            return {}

    def _serialize_text_chunk(self, chunk) -> Dict[str, Any]:
        """Convert text chunk dataclass to dict safely."""
        try:
            chunk_dict = {
                "text": getattr(chunk, "text", ""),
                "page_number": getattr(chunk, "page_number", 1),
                "chunk_type": getattr(chunk, "chunk_type", ""),
                "font_info": getattr(chunk, "font_info", {}),
                "bbox": getattr(chunk, "bbox", None),
            }
            # Convert tuple bbox to list if needed
            if chunk_dict["bbox"] and isinstance(chunk_dict["bbox"], tuple):
                chunk_dict["bbox"] = list(chunk_dict["bbox"])
            return chunk_dict
        except Exception as e:
            logger.warning(f"Error serializing text chunk: {e}")
            return {
                "text": str(chunk),
                "page_number": 1,
                "chunk_type": "",
                "font_info": {},
                "bbox": None,
            }

# Example usage for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

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

        meta = result.get("doc_metadata", {})
        print(f"Document: {meta.get('title', 'Unknown')}")
        print(f"Text Chunks: {len(result['text_chunks'])}")
        print(f"Tables: {len(result['tables'])}")
        print(f"Images: {len(result['images'])}")
        print(f"Semantic Chunks: {len(result['semantic_chunks'])}")

        # Print analytics
        analytics = result.get("analytics", {})
        print("\nDocument Analytics:")
        print(f"- Summary: {analytics.get('summary', '')}")
        print(f"- Entities: {analytics.get('entities', [])}")
        print(f"- Top Keywords: {analytics.get('keywords', [])}")
        print(f"- Word Count: {analytics.get('word_count', 0)}")
        print(f"- Sentence Count: {analytics.get('sentence_count', 0)}")

    except Exception as e:
        print(f"Error processing document: {str(e)}")
        sys.exit(1)