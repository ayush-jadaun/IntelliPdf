"""
pgvector Vector Store for IntelliPDF

- Store document and chunk embeddings
- Query for semantic similarity (cosine)
- Bulk operations
- Designed for integration with Document and Chunk SQLAlchemy models

Author: IntelliPDF Team
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, func, cast, literal_column
from app.core.database.models import Document, Chunk
from pgvector.sqlalchemy import Vector

# =======================
# Document-level operations
# =======================

def add_document_embedding(
    db: Session,
    title: str,
    embedding: List[float],
    file_path: Optional[str] = None,
    doc_metadata: Optional[dict] = None,
) -> int:
    """
    Insert a new document with embedding.
    Returns: document id
    """
    doc = Document(
        title=title,
        file_path=file_path,
        embedding=embedding,
        doc_metadata=doc_metadata,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc.id

def update_document_embedding(
    db: Session,
    doc_id: int,
    embedding: List[float]
) -> bool:
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if doc:
        doc.embedding = embedding
        db.commit()
        return True
    return False

def delete_document_embedding(
    db: Session,
    doc_id: int
) -> bool:
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if doc:
        db.delete(doc)
        db.commit()
        return True
    return False

def query_similar_documents(
    db: Session,
    embedding: List[float],
    top_k: int = 5,
    min_score: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Query for the top_k most similar documents using cosine similarity.
    Returns: [{id, title, score}]
    """
    # Convert embedding to pgvector literal and use literal_column to inline it in SQL
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
    # ---- FIX: Ensure correct vector dimension! ----
    VECTOR_DIM = len(embedding)
    stmt = (
        select(
            Document.id,
            Document.title,
            func.cosine_distance(
                Document.embedding,
                cast(literal_column(f"'{embedding_str}'"), Vector(VECTOR_DIM))
            ).label("distance"),
        )
        .order_by("distance")
        .limit(top_k)
    )
    results = db.execute(stmt).fetchall()
    output = []
    for row in results:
        score = 1.0 - row.distance  # 1.0 = identical, 0.0 = orthogonal
        if min_score is None or score >= min_score:
            output.append({
                "id": row.id,
                "title": row.title,
                "score": score
            })
    return output

def bulk_add_document_embeddings(
    db: Session,
    docs: List[Dict[str, Any]]
) -> List[int]:
    """
    Insert multiple documents at once.
    Each dict: {title, embedding, file_path, doc_metadata}
    Returns: list of document ids
    """
    objects = [Document(**doc) for doc in docs]
    db.add_all(objects)
    db.commit()
    return [obj.id for obj in objects]

# =======================
# Chunk-level operations
# =======================

def add_chunk_embedding(
    db: Session,
    document_id: int,
    text: str,
    embedding: List[float],
    page_number: Optional[int] = None,
    chunk_type: Optional[str] = None,
    doc_metadata: Optional[dict] = None,
) -> int:
    """
    Insert a new chunk with embedding.
    Returns: chunk id
    """
    chunk = Chunk(
        document_id=document_id,
        text=text,
        embedding=embedding,
        page_number=page_number,
        chunk_type=chunk_type,
        doc_metadata=doc_metadata,
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk.id

def update_chunk_embedding(
    db: Session,
    chunk_id: int,
    embedding: List[float]
) -> bool:
    chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
    if chunk:
        chunk.embedding = embedding
        db.commit()
        return True
    return False

def delete_chunk_embedding(
    db: Session,
    chunk_id: int
) -> bool:
    chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
    if chunk:
        db.delete(chunk)
        db.commit()
        return True
    return False

def query_similar_chunks(
    db: Session,
    embedding: List[float],
    top_k: int = 5,
    min_score: Optional[float] = None,
    document_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Query for the top_k most similar chunks using cosine similarity.
    Optionally filter by document_id.
    Returns: [{id, document_id, text, score, page_number, chunk_type}]
    """
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
    VECTOR_DIM = len(embedding)
    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.text,
            Chunk.page_number,
            Chunk.chunk_type,
            func.cosine_distance(
                Chunk.embedding,
                cast(literal_column(f"'{embedding_str}'"), Vector(VECTOR_DIM))
            ).label("distance")
        )
    )
    if document_id:
        stmt = stmt.filter(Chunk.document_id == document_id)
    stmt = stmt.order_by("distance").limit(top_k)
    results = db.execute(stmt).fetchall()
    output = []
    for row in results:
        score = 1.0 - row.distance
        if min_score is None or score >= min_score:
            output.append({
                "id": row.id,
                "document_id": row.document_id,
                "text": row.text,
                "score": score,
                "page_number": row.page_number,
                "chunk_type": row.chunk_type,
            })
    return output

def bulk_add_chunk_embeddings(
    db: Session,
    chunks: List[Dict[str, Any]]
) -> List[int]:
    """
    Insert multiple chunks at once.
    Each dict: {document_id, text, embedding, ...}
    Returns: list of chunk ids
    """
    objects = [Chunk(**chunk) for chunk in chunks]
    db.add_all(objects)
    db.commit()
    return [obj.id for obj in objects]

# =======================
# Simple fetchers
# =======================

def get_document_by_id(db: Session, doc_id: int) -> Optional[Document]:
    return db.query(Document).filter(Document.id == doc_id).first()

def get_chunk_by_id(db: Session, chunk_id: int) -> Optional[Chunk]:
    return db.query(Chunk).filter(Chunk.id == chunk_id).first()

def get_all_documents(db: Session, limit: int = 100) -> List[Document]:
    return db.query(Document).limit(limit).all()

def get_chunks_for_document(db: Session, document_id: int, limit: int = 100) -> List[Chunk]:
    return db.query(Chunk).filter(Chunk.document_id == document_id).limit(limit).all()