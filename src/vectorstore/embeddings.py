"""
Embedding Module for AI Research Assistant

This module handles generating embeddings from text chunks using various providers
including OpenAI, HuggingFace, and Cohere.
"""

import os
import time
from typing import List, Dict, Union, Optional
from abc import ABC, abstractmethod
import logging
from dotenv import load_dotenv

# Import embedding providers
import openai
from sentence_transformers import SentenceTransformer
import cohere

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        Initialize OpenAI embeddings.
        
        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (or from environment)
        """
        self.model = model
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        # Model dimensions
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        if model not in self._dimensions:
            logger.warning(f"Unknown model {model}, assuming 1536 dimensions")
            self._dimensions[model] = 1536
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimensions[self.model]


class HuggingFaceEmbeddings(EmbeddingProvider):
    """HuggingFace sentence-transformers embedding provider."""
    
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        """
        Initialize HuggingFace embeddings.
        
        Args:
            model: HuggingFace model name
        """
        self.model_name = model
        logger.info(f"Loading HuggingFace model: {model}")
        self.model = SentenceTransformer(model)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self.model.get_sentence_embedding_dimension()


class CohereEmbeddings(EmbeddingProvider):
    """Cohere embedding provider."""
    
    def __init__(self, model: str = "embed-english-light-v3.0", api_key: Optional[str] = None):
        """
        Initialize Cohere embeddings.
        
        Args:
            model: Cohere embedding model name
            api_key: Cohere API key (or from environment)
        """
        self.model = model
        self.client = cohere.Client(api_key or os.getenv("COHERE_API_KEY"))
        
        # Model dimensions (approximate)
        self._dimensions = {
            "embed-english-light-v3.0": 384,
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024
        }
        
        if model not in self._dimensions:
            logger.warning(f"Unknown Cohere model {model}, assuming 1024 dimensions")
            self._dimensions[model] = 1024
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type="search_document"
            )
            return response.embeddings
        except Exception as e:
            logger.error(f"Cohere embedding error: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimensions[self.model]


class EmbeddingManager:
    """Manager class for handling different embedding providers."""
    
    def __init__(self, provider: str = "huggingface", model: str = None, **kwargs):
        """
        Initialize embedding manager.
        
        Args:
            provider: Provider name ('openai', 'huggingface', 'cohere')
            model: Model name (provider-specific)
            **kwargs: Additional arguments for the provider
        """
        self.provider_name = provider.lower()
        
        # Default models for each provider
        default_models = {
            "openai": "text-embedding-3-small",
            "huggingface": "all-MiniLM-L6-v2",
            "cohere": "embed-english-light-v3.0"
        }
        
        if model is None:
            model = default_models.get(self.provider_name)
        
        # Initialize provider
        if self.provider_name == "openai":
            self.provider = OpenAIEmbeddings(model=model, **kwargs)
        elif self.provider_name == "huggingface":
            self.provider = HuggingFaceEmbeddings(model=model)
        elif self.provider_name == "cohere":
            self.provider = CohereEmbeddings(model=model, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        logger.info(f"Initialized {self.provider_name} embeddings with model {model}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 100, 
                   delay: float = 0.1) -> List[List[float]]:
        """
        Embed texts in batches with rate limiting.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch
            delay: Delay between batches in seconds
            
        Returns:
            List of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                batch_embeddings = self.provider.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                # Rate limiting
                if delay > 0 and i + batch_size < len(texts):
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Add empty embeddings for failed batch
                all_embeddings.extend([[0.0] * self.dimension] * len(batch))
        
        return all_embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """Embed a single text."""
        return self.provider.embed_single(text)
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.provider.dimension


def create_embedding_manager(config: Dict) -> EmbeddingManager:
    """
    Create embedding manager from configuration.
    
    Args:
        config: Configuration dictionary with provider settings
        
    Returns:
        EmbeddingManager instance
    """
    provider = config.get("provider", "huggingface")
    model = config.get("model")
    
    # Extract provider-specific config
    provider_config = config.get(f"{provider}_config", {})
    
    return EmbeddingManager(
        provider=provider,
        model=model,
        **provider_config
    )


# Example usage and testing
if __name__ == "__main__":
    # Test with HuggingFace (no API key required)
    print("Testing HuggingFace embeddings...")
    hf_manager = EmbeddingManager(provider="huggingface")
    
    test_texts = [
        "This is a test document about machine learning.",
        "Linear algebra is fundamental to understanding neural networks.",
        "Python is a popular programming language for data science."
    ]
    
    embeddings = hf_manager.embed_batch(test_texts)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {hf_manager.dimension}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")