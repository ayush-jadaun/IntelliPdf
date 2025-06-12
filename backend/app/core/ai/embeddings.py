import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


def embed_texts_with_gemini(texts, model: str = "models/embedding-001") -> list[list[float]]:
    """
    Generate embeddings for a list of text chunks using Gemini/Google Generative AI Python SDK.
    texts can be a list of strings or a list of dictionaries with 'text' key.
    Returns a list of embedding lists.
    """
    embeddings = []
    
    for item in texts:
        # Extract text content - handle both string and dict formats
        if isinstance(item, dict):
            text_content = item.get('text', '')
        else:
            text_content = str(item)
        
        # Skip empty texts
        if not text_content.strip():
            embeddings.append([0.0] * 768)  # Default embedding size
            continue
            
        # Use embed_content method for each text
        result = genai.embed_content(
            model=model,
            content=text_content,
            task_type="retrieval_document"  # Optional: specify task type
        )
        embeddings.append(result['embedding'])
    
    return embeddings


# Alternative batch processing approach (if you want to optimize for fewer API calls)
def embed_texts_with_gemini_batch(texts, model: str = "models/embedding-001") -> list[list[float]]:
    """
    Alternative approach with better error handling.
    texts can be a list of strings or a list of dictionaries with 'text' key.
    """
    embeddings = []
    
    for item in texts:
        try:
            # Extract text content - handle both string and dict formats
            if isinstance(item, dict):
                text_content = item.get('text', '')
            else:
                text_content = str(item)
            
            # Skip empty texts
            if not text_content.strip():
                embeddings.append([0.0] * 768)  # Default embedding size
                continue
                
            result = genai.embed_content(
                model=model,
                content=text_content,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        except Exception as e:
            print(f"Error embedding text: {e}")
            # You might want to handle this differently based on your needs
            embeddings.append([0.0] * 768)  # Default embedding size, adjust as needed
    
    return embeddings