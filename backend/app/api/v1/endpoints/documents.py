from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.utils.file_handlers import save_uploaded_file, delete_file
from app.core.ai.document_processor import DocumentProcessingPipeline
from app.schemas.document import ProcessedDocumentResponse
import os
import tempfile

router = APIRouter()

@router.post("/process/", response_model=ProcessedDocumentResponse)
async def process_document(file: UploadFile = File(...)):
    # Save uploaded file
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
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        os.remove(tmp_path)