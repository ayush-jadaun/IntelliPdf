from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from app.core.ai.document_processor import DocumentProcessingPipeline
from app.schemas.document import ProcessedDocumentResponse, DocumentMetadata, DocumentStructure, TextAnalytics, KnowledgeGraph
from app.schemas.document import build_tables_safe, build_images_safe
from app.core.database.vector_store import add_document_embedding, add_chunk_embedding
from app.core.database.session import get_db
from sqlalchemy.orm import Session
import os
import tempfile
import asyncio
import logging
import traceback

router = APIRouter()
logger = logging.getLogger(__name__)

def create_safe_document_response(result: dict, file_path: str, filename: str) -> ProcessedDocumentResponse:
    """Create ProcessedDocumentResponse with safe key mapping and error handling."""
    try:
        # Handle metadata mapping - your pipeline returns 'metadata' but schema expects 'doc_metadata'
        metadata_raw = result.get("metadata", {})
        if isinstance(metadata_raw, dict):
            doc_metadata = DocumentMetadata(**metadata_raw)
        else:
            doc_metadata = DocumentMetadata()
        
        # Handle structure mapping
        structure_raw = result.get("structure", {})
        if isinstance(structure_raw, dict):
            structure = DocumentStructure(**structure_raw)
        else:
            structure = DocumentStructure()
        
        # Handle analytics mapping
        analytics_raw = result.get("analytics")
        analytics = None
        if analytics_raw and isinstance(analytics_raw, dict):
            try:
                analytics = TextAnalytics(**analytics_raw)
            except Exception as e:
                logger.warning(f"Could not create analytics object: {e}")
                analytics = TextAnalytics()
        
        # Handle knowledge graph mapping
        kg_raw = result.get("knowledge_graph")
        knowledge_graph = None
        if kg_raw and isinstance(kg_raw, dict):
            try:
                knowledge_graph = KnowledgeGraph(**kg_raw)
            except Exception as e:
                logger.warning(f"Could not create knowledge graph object: {e}")
                knowledge_graph = KnowledgeGraph()
        
        # Process tables and images safely
        logger.info(f"Processing {len(result.get('tables', []))} tables")
        tables = build_tables_safe(result.get("tables", []))
        logger.info(f"Successfully processed {len(tables)} tables")
        
        images = build_images_safe(result.get("images", []))
        
        # Ensure document_id is always a string
        doc_id = result.get("document_id")
        document_id_str = str(doc_id) if doc_id is not None else None

        # Create the response
        response = ProcessedDocumentResponse(
            file_path=file_path,
            doc_metadata=doc_metadata,
            full_text=result.get("full_text", ""),
            text_chunks=result.get("text_chunks", []),
            semantic_chunks=result.get("semantic_chunks", []),
            tables=tables,
            images=images,
            structure=structure,
            analytics=analytics,
            knowledge_graph=knowledge_graph,
            document_id=document_id_str,
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating safe response: {e}")
        logger.error(f"Result keys: {list(result.keys()) if result else 'None'}")
        # Return minimal valid response
        return ProcessedDocumentResponse(
            file_path=file_path,
            doc_metadata=DocumentMetadata(),
            structure=DocumentStructure(),
            analytics=TextAnalytics()
        )

@router.post("/process/", response_model=ProcessedDocumentResponse)
async def process_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = os.path.splitext(file.filename)[-1]
    if suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    tmp_path = None
    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"Processing file: {file.filename} (temp: {tmp_path})")

        # Process document
        pipeline = DocumentProcessingPipeline()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, pipeline.process, tmp_path)
        logger.info("Pipeline processing completed successfully")

        # --- SAVE TO DATABASE ---
        # Save document first, get its id
        doc_metadata = result.get("metadata", {})
        doc_embedding = result.get("embedding")  # Or however your pipeline outputs document-level embedding
        document_id = add_document_embedding(
            db,
            title=file.filename,
            embedding=doc_embedding,
            file_path=tmp_path,
            doc_metadata=doc_metadata
        )

        # Save chunks
        for chunk_data in result.get("semantic_chunks", []):
            add_chunk_embedding(
                db,
                document_id=document_id,
                text=chunk_data.get("text"),
                embedding=chunk_data.get("embedding"),
                page_number=chunk_data.get("page_number"),
                chunk_type=chunk_data.get("chunk_type"),
                doc_metadata=chunk_data.get("metadata")
            )

        # Prepare API response
        # Patch result["document_id"] so response uses correct value as string
        result["document_id"] = str(document_id) if document_id is not None else None
        response = create_safe_document_response(result, tmp_path, file.filename)
        # No need for: response.document_id = str(document_id) -- it's handled above

        logger.info("Response object created successfully")
        return response

    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.info(f"Cleaned up temp file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {tmp_path}: {e}")

# Keep your debugging endpoint as is - it's very helpful!
@router.post("/process-simple/")
async def process_document_simple(file: UploadFile = File(...)):
    """Simplified version for debugging - returns basic dict instead of Pydantic model"""
    suffix = os.path.splitext(file.filename)[-1]
    if suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    tmp_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"Processing file: {file.filename} (temp: {tmp_path})")

        # Create pipeline
        pipeline = DocumentProcessingPipeline()
        
        # Run sync pipeline code in threadpool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, pipeline.process, tmp_path)
        
        logger.info("Pipeline processing completed successfully")
        
        # Return detailed result structure for debugging
        doc_id = result.get("document_id")
        simplified_result = {
            "status": "success",
            "filename": file.filename,
            "result_keys": list(result.keys()) if result else [],
            "metadata_keys": list(result.get("metadata", {}).keys()) if result.get("metadata") else [],
            "structure_keys": list(result.get("structure", {}).keys()) if result.get("structure") else [],
            "analytics_keys": list(result.get("analytics", {}).keys()) if result.get("analytics") else [],
            "pages": result.get("metadata", {}).get("page_count", 0),
            "text_chunks_count": len(result.get("text_chunks", [])),
            "semantic_chunks_count": len(result.get("semantic_chunks", [])),
            "tables_count": len(result.get("tables", [])),
            "images_count": len(result.get("images", [])),
            "document_id": str(doc_id) if doc_id is not None else None,
            "has_analytics": "analytics" in result,
            "has_knowledge_graph": "knowledge_graph" in result,
            # Include first 500 chars of text for verification
            "text_preview": result.get("full_text", "")[:500],
            # Show table structure for debugging
            "table_structures": [type(table).__name__ for table in result.get("tables", [])],
            # Show sample table data
            "sample_table_data": result.get("tables", [])[0] if result.get("tables") else None
        }
        
        return simplified_result
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            "status": "error",
            "error": str(e),
            "filename": file.filename
        }
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {tmp_path}: {e}")