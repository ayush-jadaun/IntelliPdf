from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.ai.document_processor import DocumentProcessingPipeline
import os
import tempfile

router = APIRouter()

@router.post("/process_document/")
async def process_document(file: UploadFile = File(...)):
    # Save file to temp location
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