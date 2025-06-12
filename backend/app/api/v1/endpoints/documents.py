from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.ai.document_processor import DocumentProcessingPipeline
from app.schemas.document import ProcessedDocumentResponse
import os
import tempfile

router = APIRouter()

@router.post("/process/", response_model=ProcessedDocumentResponse)
async def process_document(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1]
    if suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process(tmp_path)

        # Return all fields, including analytics and semantic chunks with embeddings
        return ProcessedDocumentResponse(
            file_path=result["file_path"],
            metadata=result["metadata"],
            full_text=result["full_text"],
            text_chunks=result["text_chunks"],
            tables=result["tables"],
            images=result["images"],
            structure=result["structure"],
            analytics=result["analytics"],
            semantic_chunks=result["semantic_chunks"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        os.remove(tmp_path)