from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.ai.document_processor import DocumentProcessingPipeline
from app.schemas.document import ProcessedDocumentResponse, TextAnalytics
from app.core.utils.text_processing import (
    preprocess_text_pipeline,
    extract_text_features,
    summarize_text_simple,
    extract_named_entities,
    extract_keywords
)
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

        # --- Integrate Text Processing ---
        full_text = result["full_text"]
        processed_text, meta = preprocess_text_pipeline(full_text)
        features = extract_text_features(processed_text)
        summary = summarize_text_simple(processed_text)
        entities = extract_named_entities(processed_text)
        keywords = extract_keywords(processed_text)

        analytics = TextAnalytics(
            word_count=features["word_count"],
            sentence_count=features["sentence_count"],
            keywords=keywords,
            summary=summary,
            entities=entities,
        )

        # Return all fields + analytics
        return ProcessedDocumentResponse(
            file_path=result["file_path"],
            metadata=result["metadata"],
            full_text=processed_text,
            text_chunks=result["text_chunks"],
            tables=result["tables"],
            images=result["images"],
            structure=result["structure"],
            analytics=analytics,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        os.remove(tmp_path)