"""
Vector Store Module

This module provides embedding generation and vector storage capabilities
for the RAG-based AI Research Assistant.
"""

from .embeddings import EmbeddingManager, create_embedding_manager
from .vector_store import (
    VectorStoreManager, 
    FAISSVectorStore, 
    ChromaDBVectorStore,
    create_vector_store_manager
)

__all__ = [
    'EmbeddingManager',
    'VectorStoreManager', 
    'FAISSVectorStore',
    'ChromaDBVectorStore',
    'create_embedding_manager',
    'create_vector_store_manager'
]