import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


def embed_texts_with_gemini(texts: list[str], model: str = "models/embedding-001"):
    """
    Generate embeddings for a list of text chunks.
    Returns a list of embedding lists.
    """
    model = genai.EmbeddingModel(model)
    response = model.embed(texts)
    return response["embeddings"]