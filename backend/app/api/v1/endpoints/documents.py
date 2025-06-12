from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.ai.document_processor import DocumentProcessingPipeline
from app.schemas.document import ProcessedDocumentResponse
import os
import tempfile
import asyncio
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/process/", response_model=ProcessedDocumentResponse)
async def process_document(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1]
    if suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        pipeline = DocumentProcessingPipeline()
        # Run sync pipeline code in threadpool to avoid blocking event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, pipeline.process, tmp_path)

        return ProcessedDocumentResponse(
            file_path=result["file_path"],
            doc_metadata=result["doc_metadata"],  # Make sure this matches your schema!
            full_text=result["full_text"],
            text_chunks=result["text_chunks"],
            tables=result["tables"],
            images=result["images"],
            structure=result["structure"],
            analytics=result.get("analytics"),
            semantic_chunks=result.get("semantic_chunks"),
            knowledge_graph=result.get("knowledge_graph"),
            document_id=result.get("document_id"),  # Optional: expose doc_id if you want
        )
    except Exception as e:
        logger.exception("Document processing failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)