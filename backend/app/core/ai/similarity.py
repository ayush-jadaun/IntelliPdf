"""
Advanced Similarity & Hybrid Ranking Utilities for IntelliPDF

Supports:
- Cosine similarity on embeddings
- TF-IDF-based keyword similarity
- Hybrid scoring (weighted sum)
- Batch and top-k search

Author: IntelliPDF Team
"""

from typing import List, Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns 0.0 if either vector is zero.
    """
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm_v1 * norm_v2))

def most_similar_chunks(
    query_embedding: List[float],
    chunk_list: List[Dict[str, Any]],
    top_k: int = 20
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Return top_k most similar chunks to the query_embedding using cosine similarity.
    """
    similarities = []
    for chunk in chunk_list:
        emb = chunk.get("embedding")
        if emb is not None and isinstance(emb, (list, np.ndarray)):
            score = cosine_similarity(query_embedding, emb)
            similarities.append((chunk, score))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def tfidf_keyword_scores(query: str, texts: List[str]) -> np.ndarray:
    """
    Returns array of TF-IDF-based similarity scores for the query against each text.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([query] + texts)
    query_vec = tfidf[0]
    doc_vecs = tfidf[1:]
    # Compute cosine similarity
    scores = (doc_vecs * query_vec.T).toarray().flatten()
    return scores

def hybrid_similar_chunks(
    query: str,
    query_embedding: List[float],
    chunk_list: List[Dict[str, Any]],
    top_k: int = 5,
    alpha: float = 0.5
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Hybrid ranking: alpha * embedding_sim + (1-alpha) * tfidf_sim
    `alpha` controls the mix: 1.0 = only embeddings, 0.0 = only tfidf.
    Returns top_k most relevant chunks.
    """
    texts = [chunk.get("text", "") for chunk in chunk_list]
    tfidf_scores = tfidf_keyword_scores(query, texts)
    emb_scores = np.array([
        cosine_similarity(query_embedding, chunk.get("embedding"))
        if chunk.get("embedding") is not None else 0.0
        for chunk in chunk_list
    ])
    # Normalize scores to 0-1
    if emb_scores.max() > emb_scores.min():
        emb_norm = (emb_scores - emb_scores.min()) / (emb_scores.max() - emb_scores.min())
    else:
        emb_norm = emb_scores
    if tfidf_scores.max() > tfidf_scores.min():
        tfidf_norm = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min())
    else:
        tfidf_norm = tfidf_scores

    hybrid_scores = alpha * emb_norm + (1 - alpha) * tfidf_norm

    scored_chunks = list(zip(chunk_list, hybrid_scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return scored_chunks[:top_k]

def batch_query_embeddings(
    queries: List[List[float]],
    chunk_list: List[Dict[str, Any]],
    top_k: int = 3
) -> List[List[Tuple[Dict[str, Any], float]]]:
    """
    For each query embedding, return top_k most similar chunks.
    Returns: List of lists [(chunk, score)] per query.
    """
    results = []
    for query_emb in queries:
        sims = []
        for chunk in chunk_list:
            emb = chunk.get("embedding")
            if emb is not None and isinstance(emb, (list, np.ndarray)):
                score = cosine_similarity(query_emb, emb)
                sims.append((chunk, score))
        sims.sort(key=lambda x: x[1], reverse=True)
        results.append(sims[:top_k])
    return results