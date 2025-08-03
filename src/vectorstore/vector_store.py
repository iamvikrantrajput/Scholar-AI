"""
Vector Store Module for ScholarAI

This module handles vector storage and retrieval using FAISS and ChromaDB
for the RAG-based research assistant system.
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from abc import ABC, abstractmethod
import uuid

import numpy as np
import faiss
import chromadb
from chromadb.config import Settings

from .embeddings import EmbeddingManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict], ids: Optional[List[str]] = None):
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def save(self, path: Path):
        """Save the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path):
        """Load the vector store from disk."""
        pass


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, dimension: int, index_type: str = "IndexFlatIP"):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension
            index_type: FAISS index type
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        elif index_type == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Store metadata and texts separately
        self.metadata = []
        self.texts = []
        self.ids = []
        
        logger.info(f"Initialized FAISS {index_type} with dimension {dimension}")
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict], ids: Optional[List[str]] = None):
        """Add documents to FAISS index."""
        if len(texts) != len(embeddings) != len(metadata):
            raise ValueError("Texts, embeddings, and metadata must have same length")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings for cosine similarity (if using IndexFlatIP)
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(embeddings_array)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store metadata
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        self.ids.extend(ids)
        
        logger.info(f"Added {len(texts)} documents to FAISS index")
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            return []
        
        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.index_type == "IndexFlatIP":
            faiss.normalize_L2(query_array)
        
        # Search
        scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append({
                    "id": self.ids[idx],
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "score": float(score)
                })
        
        return results
    
    def save(self, path: Path):
        """Save FAISS index and metadata."""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "faiss.index"))
        
        # Save metadata
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "texts": self.texts,
                "metadata": self.metadata,
                "ids": self.ids,
                "dimension": self.dimension,
                "index_type": self.index_type
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved FAISS index to {path}")
    
    def load(self, path: Path):
        """Load FAISS index and metadata."""
        # Load FAISS index
        self.index = faiss.read_index(str(path / "faiss.index"))
        
        # Load metadata
        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.texts = data["texts"]
            self.metadata = data["metadata"]
            self.ids = data["ids"]
            self.dimension = data["dimension"]
            self.index_type = data["index_type"]
        
        logger.info(f"Loaded FAISS index from {path}")


class ChromaDBVectorStore(VectorStore):
    """ChromaDB-based vector store implementation."""
    
    def __init__(self, collection_name: str = "research_documents", 
                 persist_directory: Optional[str] = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        logger.info(f"Initialized ChromaDB collection: {collection_name}")
    
    def add_documents(self, texts: List[str], embeddings: List[List[float]], 
                     metadata: List[Dict], ids: Optional[List[str]] = None):
        """Add documents to ChromaDB collection."""
        if len(texts) != len(embeddings) != len(metadata):
            raise ValueError("Texts, embeddings, and metadata must have same length")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata,
            ids=ids
        )
        
        logger.info(f"Added {len(texts)} documents to ChromaDB collection")
    
    def search(self, query_embedding: List[float], k: int = 5, 
              where: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents with optional metadata filtering."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where
        )
        
        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
        
        return formatted_results
    
    def save(self, path: Path):
        """ChromaDB handles persistence automatically if configured."""
        logger.info("ChromaDB persistence is handled automatically")
    
    def load(self, path: Path):
        """ChromaDB loads automatically if persistent client is used."""
        logger.info("ChromaDB loading is handled automatically")


class VectorStoreManager:
    """Manager class for vector stores and embeddings."""
    
    def __init__(self, embedding_manager: EmbeddingManager = None, 
                 vector_store: VectorStore = None,
                 store_type: str = "faiss",
                 store_path: str = None):
        """
        Initialize vector store manager.
        
        Args:
            embedding_manager: Embedding manager instance
            vector_store: Vector store instance
            store_type: Type of vector store ('faiss' or 'chromadb')
            store_path: Path for storing the vector store
        """
        if embedding_manager is None:
            embedding_manager = EmbeddingManager(provider="huggingface")
        
        self.embedding_manager = embedding_manager
        
        if vector_store is None:
            if store_type.lower() == "faiss":
                vector_store = FAISSVectorStore(
                    dimension=self.embedding_manager.dimension,
                    index_type="IndexFlatIP"
                )
            elif store_type.lower() == "chromadb":
                vector_store = ChromaDBVectorStore(
                    collection_name="research_documents",
                    persist_directory=store_path
                )
            else:
                raise ValueError(f"Unknown vector store type: {store_type}")
        
        self.vector_store = vector_store
        self.store_path = store_path
    
    def load_processed_data(self, processed_dir: Path) -> List[Dict]:
        """
        Load processed JSON files from directory.
        
        Args:
            processed_dir: Directory containing processed JSON files
            
        Returns:
            List of document chunks with metadata
        """
        all_chunks = []
        
        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                    all_chunks.extend(chunks)
                logger.info(f"Loaded {len(chunks)} chunks from {json_file.name}")
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        logger.info(f"Total loaded chunks: {len(all_chunks)}")
        return all_chunks
    
    def build_vector_store(self, processed_dir: Path, batch_size: int = 50):
        """
        Build vector store from processed JSON files.
        
        Args:
            processed_dir: Directory containing processed JSON files
            batch_size: Batch size for embedding generation
        """
        # Load processed data
        chunks = self.load_processed_data(processed_dir)
        
        if not chunks:
            logger.warning("No chunks loaded, nothing to process")
            return
        
        # Extract texts and metadata
        texts = [chunk["text"] for chunk in chunks]
        metadata = [chunk["metadata"] for chunk in chunks]
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_manager.embed_batch(texts, batch_size=batch_size)
        
        # Add to vector store
        logger.info("Adding documents to vector store...")
        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadata=metadata
        )
        
        logger.info("Vector store building complete!")
    
    def build_vector_store_from_texts(self, texts: List[str], metadata: List[Dict], 
                                    batch_size: int = 50):
        """
        Build vector store directly from texts and metadata.
        
        Args:
            texts: List of text chunks
            metadata: List of metadata dictionaries
            batch_size: Batch size for embedding generation
        """
        if not texts:
            logger.warning("No texts provided, nothing to process")
            return
        
        if len(texts) != len(metadata):
            raise ValueError("Texts and metadata must have same length")
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_manager.embed_batch(texts, batch_size=batch_size)
        
        # Add to vector store
        logger.info("Adding documents to vector store...")
        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadata=metadata
        )
        
        logger.info("Vector store building complete!")
    
    def search(self, query: str, k: int = 5, **kwargs) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            **kwargs: Additional arguments for vector store search
            
        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_single(query)
        
        # Search vector store
        return self.vector_store.search(query_embedding, k=k, **kwargs)
    
    def save_vector_store(self, path: Path):
        """Save vector store to disk."""
        self.vector_store.save(path)
    
    def load_vector_store(self, path: Path):
        """Load vector store from disk."""
        self.vector_store.load(path)
    
    def get_langchain_retriever(self, k: int = 3):
        """
        Get a LangChain-compatible retriever.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            LangChain BaseRetriever instance
        """
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.documents import Document
        from typing import List
        
        class VectorStoreRetriever(BaseRetriever):
            """Custom retriever for vector store."""
            
            vector_manager: 'VectorStoreManager'
            k: int
            
            def __init__(self, vector_manager: 'VectorStoreManager', k: int = 3):
                super().__init__()
                object.__setattr__(self, 'vector_manager', vector_manager)
                object.__setattr__(self, 'k', k)
            
            def _get_relevant_documents(self, query: str) -> List[Document]:
                results = self.vector_manager.search(query, k=self.k)
                documents = []
                for result in results:
                    doc = Document(
                        page_content=result["text"],
                        metadata=result["metadata"]
                    )
                    documents.append(doc)
                return documents
        
        return VectorStoreRetriever(self, k=k)


def create_vector_store_manager(config: Dict) -> VectorStoreManager:
    """
    Create vector store manager from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VectorStoreManager instance
    """
    # Create embedding manager
    embedding_config = config.get("embeddings", {})
    embedding_manager = EmbeddingManager(**embedding_config)
    
    # Create vector store
    vector_store_config = config.get("vector_store", {})
    store_type = vector_store_config.get("type", "faiss").lower()
    
    if store_type == "faiss":
        vector_store = FAISSVectorStore(
            dimension=embedding_manager.dimension,
            index_type=vector_store_config.get("index_type", "IndexFlatIP")
        )
    elif store_type == "chromadb":
        vector_store = ChromaDBVectorStore(
            collection_name=vector_store_config.get("collection_name", "research_documents"),
            persist_directory=vector_store_config.get("persist_directory")
        )
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
    
    return VectorStoreManager(embedding_manager, vector_store)


# Example usage
if __name__ == "__main__":
    # Test configuration
    config = {
        "embeddings": {
            "provider": "huggingface",
            "model": "all-MiniLM-L6-v2"
        },
        "vector_store": {
            "type": "faiss",
            "index_type": "IndexFlatIP"
        }
    }
    
    # Create manager
    manager = create_vector_store_manager(config)
    
    # Test data
    test_texts = [
        "Linear programming is a method for optimizing linear functions.",
        "Machine learning algorithms can learn from data.",
        "Python is widely used in data science applications."
    ]
    
    test_metadata = [
        {"source": "test1.pdf", "page": 1, "topic": "optimization"},
        {"source": "test2.pdf", "page": 1, "topic": "machine learning"},
        {"source": "test3.pdf", "page": 1, "topic": "programming"}
    ]
    
    # Generate embeddings and add to store
    embeddings = manager.embedding_manager.embed_batch(test_texts)
    manager.vector_store.add_documents(test_texts, embeddings, test_metadata)
    
    # Test search
    results = manager.search("What is optimization?", k=2)
    
    print("Search results:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['score']:.3f}")
        print(f"   Text: {result['text'][:100]}...")
        print(f"   Source: {result['metadata']['source']}")
        print()