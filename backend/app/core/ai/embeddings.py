import os
import logging
from typing import List, Union, Dict, Any, Optional
import numpy as np
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model."""
        pass


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Local sentence transformers embeddings - no API limits, runs locally."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence transformers."""
        # Handle empty texts
        processed_texts = [text if text.strip() else "empty" for text in texts]
        # Generate embeddings
        embeddings = self.model.encode(processed_texts, convert_to_tensor=False)
        return embeddings.tolist()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        sample_embedding = self.model.encode(["test"])
        return {
            "provider": "SentenceTransformers",
            "model": self.model_name,
            "dimension": len(sample_embedding[0]),
            "max_input_tokens": 512,
            "status": "available",
            "local": True
        }


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI embeddings - reliable API with good quotas."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize OpenAI embedder.
        """
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.model = model
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API."""
        processed_texts = [text if text.strip() else "empty" for text in texts]
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=processed_texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            dimension = 1536 if "small" in self.model or "ada" in self.model else 3072
            return [[0.0] * dimension for _ in texts]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return {
            "provider": "OpenAI",
            "model": self.model,
            "dimension": dimension_map.get(self.model, 1536),
            "max_input_tokens": 8191,
            "status": "available",
            "local": False
        }


class HuggingFaceEmbedder(EmbeddingProvider):
    """Hugging Face embeddings - free API with reasonable limits."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", api_key: Optional[str] = None):
        """
        Initialize Hugging Face embedder.
        """
        try:
            import requests
            self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
            self.model = model
            self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
            self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self.session = requests.Session()
        except ImportError:
            raise ImportError("Install requests: pip install requests")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Hugging Face API."""
        import requests
        processed_texts = [text if text.strip() else "empty" for text in texts]
        try:
            response = self.session.post(
                self.api_url,
                headers=self.headers,
                json={
                    "inputs": processed_texts,
                    "options": {"wait_for_model": True}
                }
            )
            if response.status_code == 200:
                embeddings = response.json()
                return embeddings
            else:
                logger.error(f"HuggingFace API error: {response.status_code} - {response.text}")
                return [[0.0] * 384 for _ in texts]
        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            return [[0.0] * 384 for _ in texts]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            sample_embeddings = self.embed_texts(["test"])
            dimension = len(sample_embeddings[0]) if sample_embeddings else 384
        except:
            dimension = 384
        return {
            "provider": "HuggingFace",
            "model": self.model,
            "dimension": dimension,
            "max_input_tokens": 512,
            "status": "available",
            "local": False
        }


class CohereEmbedder(EmbeddingProvider):
    """Cohere embeddings - good API with generous free tier."""

    def __init__(self, model: str = "embed-english-v3.0", api_key: Optional[str] = None):
        """
        Initialize Cohere embedder.
        """
        try:
            import cohere
            self.client = cohere.Client(api_key or os.getenv("COHERE_API_KEY"))
            self.model = model
        except ImportError:
            raise ImportError("Install cohere: pip install cohere")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Cohere API."""
        processed_texts = [text if text.strip() else "empty" for text in texts]
        try:
            response = self.client.embed(
                texts=processed_texts,
                model=self.model,
                input_type="search_document"
            )
            return response.embeddings
        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            dimension = 384 if "light" in self.model else 1024
            return [[0.0] * dimension for _ in texts]

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        dimension = 384 if "light" in self.model else 1024
        return {
            "provider": "Cohere",
            "model": self.model,
            "dimension": dimension,
            "max_input_tokens": 512,
            "status": "available",
            "local": False
        }


class UniversalEmbedder:
    """Universal embedder that can use any provider."""

    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def embed_texts(
        self,
        texts: List[Union[str, Dict[str, Any]]],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with batching.
        """
        text_strings = []
        for item in texts:
            if isinstance(item, dict):
                text_strings.append(item.get('text', ''))
            else:
                text_strings.append(str(item))
        all_embeddings = []
        total_batches = (len(text_strings) + batch_size - 1) // batch_size
        for i in range(0, len(text_strings), batch_size):
            batch = text_strings[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            try:
                batch_embeddings = self.provider.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                dimension = self.get_embedding_dimension()
                all_embeddings.extend([[0.0] * dimension for _ in batch])
        return all_embeddings

    def get_embedding_dimension(self) -> int:
        info = self.provider.get_model_info()
        return info.get('dimension', 384)

    def get_model_info(self) -> Dict[str, Any]:
        return self.provider.get_model_info()


def create_embedder(provider: str, **kwargs) -> UniversalEmbedder:
    """
    Create an embedder for the specified provider.
    """
    providers = {
        'sentence_transformers': SentenceTransformerEmbedder,
        'openai': OpenAIEmbedder,
        'huggingface': HuggingFaceEmbedder,
        'cohere': CohereEmbedder
    }
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    provider_class = providers[provider]
    provider_instance = provider_class(**kwargs)
    return UniversalEmbedder(provider_instance)


if __name__ == "__main__":
    # Test data
    sample_texts = [
        "This is a test document about machine learning.",
        {"text": "This document discusses natural language processing."},
        "Vector embeddings are useful for similarity search.",
        {"text": "Rate limiting is important for API calls."},
        ""
    ]
    providers_to_test = [
        ('sentence_transformers', {'model_name': 'all-MiniLM-L6-v2'}),
        # ('openai', {'model': 'text-embedding-3-small'}),
        # ('huggingface', {'model': 'sentence-transformers/all-MiniLM-L6-v2'}),
        # ('cohere', {'model': 'embed-english-light-v3.0'}),
    ]
    for provider_name, kwargs in providers_to_test:
        print(f"\n=== Testing {provider_name.upper()} ===")
        try:
            embedder = create_embedder(provider_name, **kwargs)
            model_info = embedder.get_model_info()
            print(f"Model Info: {model_info}")
            embeddings = embedder.embed_texts(sample_texts)
            print(f"Generated {len(embeddings)} embeddings")
            if embeddings and embeddings[0]:
                print(f"Embedding dimension: {len(embeddings[0])}")
                print(f"Sample values: {embeddings[0][:5]}...")
        except Exception as e:
            print(f"Error testing {provider_name}: {e}")
            print("Make sure you have the required dependencies and API keys")