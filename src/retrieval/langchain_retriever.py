"""
LangChain Retrieval Module for ScholarAI

This module implements the retrieval component of the RAG system using LangChain
for semantic search over embedded documents.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

# LangChain imports
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports
import sys
sys.path.append('..')
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LangChainEmbeddingAdapter:
    """Adapter to make our custom embeddings work with LangChain."""
    
    def __init__(self, provider: str = "huggingface", model: str = None, **kwargs):
        """
        Initialize embedding adapter.
        
        Args:
            provider: Provider name ('openai', 'huggingface')
            model: Model name
            **kwargs: Additional arguments
        """
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=model or "text-embedding-3-small",
                openai_api_key=kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
            )
        elif self.provider == "huggingface":
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model or "all-MiniLM-L6-v2",
                model_kwargs={'device': 'cuda' if kwargs.get('use_gpu', True) else 'cpu'}
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def get_embeddings(self) -> Embeddings:
        """Get LangChain embeddings object."""
        return self.embeddings


class DocumentRetriever:
    """Main class for document retrieval using LangChain."""
    
    def __init__(self, vector_store: VectorStore, search_kwargs: Optional[Dict] = None):
        """
        Initialize document retriever.
        
        Args:
            vector_store: LangChain vector store
            search_kwargs: Search configuration
        """
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs or {"k": 3}
        self.retriever = self._create_retriever()
    
    def _create_retriever(self) -> BaseRetriever:
        """Create LangChain retriever from vector store."""
        return self.vector_store.as_retriever(search_kwargs=self.search_kwargs)
    
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (overrides default)
            
        Returns:
            List of LangChain Document objects
        """
        if k is not None:
            # Temporarily override k
            original_k = self.search_kwargs.get("k", 3)
            self.search_kwargs["k"] = k
            self.retriever = self._create_retriever()
            
            try:
                results = self.retriever.invoke(query)
            finally:
                # Restore original k
                self.search_kwargs["k"] = original_k
                self.retriever = self._create_retriever()
        else:
            results = self.retriever.invoke(query)
        
        return results
    
    def retrieve_with_scores(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of (Document, score) tuples
        """
        k = k or self.search_kwargs.get("k", 3)
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_retriever(self) -> BaseRetriever:
        """Get the LangChain retriever object."""
        return self.retriever


def load_documents_from_json(processed_dir: Path) -> List[Document]:
    """
    Load documents from processed JSON files.
    
    Args:
        processed_dir: Directory containing processed JSON files
        
    Returns:
        List of LangChain Document objects
    """
    documents = []
    
    for json_file in processed_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            
            for chunk in chunks:
                doc = Document(
                    page_content=chunk["text"],
                    metadata=chunk["metadata"]
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(chunks)} documents from {json_file.name}")
            
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
    
    logger.info(f"Total loaded documents: {len(documents)}")
    return documents


def create_faiss_store(documents: List[Document], embeddings: Embeddings, 
                      save_path: Optional[Path] = None) -> FAISS:
    """
    Create FAISS vector store from documents.
    
    Args:
        documents: List of Document objects
        embeddings: LangChain embeddings
        save_path: Path to save the vector store
        
    Returns:
        FAISS vector store
    """
    logger.info(f"Creating FAISS store from {len(documents)} documents...")
    
    vector_store = FAISS.from_documents(documents, embeddings)
    
    if save_path:
        vector_store.save_local(str(save_path))
        logger.info(f"Saved FAISS store to {save_path}")
    
    return vector_store


def load_faiss_store(store_path: Path, embeddings: Embeddings) -> FAISS:
    """
    Load FAISS vector store from disk.
    
    Args:
        store_path: Path to the saved vector store
        embeddings: LangChain embeddings
        
    Returns:
        FAISS vector store
    """
    logger.info(f"Loading FAISS store from {store_path}")
    return FAISS.load_local(str(store_path), embeddings, allow_dangerous_deserialization=True)


def create_chroma_store(documents: List[Document], embeddings: Embeddings,
                       persist_directory: Path, collection_name: str = "research_docs") -> Chroma:
    """
    Create ChromaDB vector store from documents.
    
    Args:
        documents: List of Document objects
        embeddings: LangChain embeddings
        persist_directory: Directory to persist the database
        collection_name: Name of the collection
        
    Returns:
        Chroma vector store
    """
    logger.info(f"Creating Chroma store from {len(documents)} documents...")
    
    vector_store = Chroma.from_documents(
        documents,
        embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name
    )
    
    logger.info(f"Created Chroma store at {persist_directory}")
    return vector_store


def load_chroma_store(persist_directory: Path, embeddings: Embeddings,
                     collection_name: str = "research_docs") -> Chroma:
    """
    Load ChromaDB vector store from disk.
    
    Args:
        persist_directory: Directory containing the persisted database
        embeddings: LangChain embeddings
        collection_name: Name of the collection
        
    Returns:
        Chroma vector store
    """
    logger.info(f"Loading Chroma store from {persist_directory}")
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name
    )


def create_retriever_from_existing_store(store_path: Path, 
                                       embedding_config: Dict[str, Any],
                                       store_type: str = "faiss",
                                       search_kwargs: Optional[Dict] = None) -> DocumentRetriever:
    """
    Create retriever from existing vector store.
    
    Args:
        store_path: Path to the vector store
        embedding_config: Embedding configuration
        store_type: Type of vector store ('faiss' or 'chroma')
        search_kwargs: Search configuration
        
    Returns:
        DocumentRetriever instance
    """
    # Initialize embeddings
    embedding_adapter = LangChainEmbeddingAdapter(**embedding_config)
    embeddings = embedding_adapter.get_embeddings()
    
    # Load vector store
    if store_type.lower() == "faiss":
        vector_store = load_faiss_store(store_path, embeddings)
    elif store_type.lower() == "chroma":
        vector_store = load_chroma_store(store_path, embeddings)
    else:
        raise ValueError(f"Unsupported store type: {store_type}")
    
    # Create retriever (remove score_threshold for FAISS compatibility)
    default_search_kwargs = {"k": 3}
    if search_kwargs:
        default_search_kwargs.update(search_kwargs)
    
    return DocumentRetriever(vector_store, default_search_kwargs)


def create_retriever_from_json(processed_dir: Path,
                              embedding_config: Dict[str, Any],
                              store_type: str = "faiss",
                              save_path: Optional[Path] = None,
                              search_kwargs: Optional[Dict] = None) -> DocumentRetriever:
    """
    Create retriever from processed JSON files.
    
    Args:
        processed_dir: Directory containing processed JSON files
        embedding_config: Embedding configuration
        store_type: Type of vector store ('faiss' or 'chroma')
        save_path: Path to save the vector store
        search_kwargs: Search configuration
        
    Returns:
        DocumentRetriever instance
    """
    # Load documents
    documents = load_documents_from_json(processed_dir)
    
    if not documents:
        raise ValueError("No documents found in processed directory")
    
    # Initialize embeddings
    embedding_adapter = LangChainEmbeddingAdapter(**embedding_config)
    embeddings = embedding_adapter.get_embeddings()
    
    # Create vector store
    if store_type.lower() == "faiss":
        vector_store = create_faiss_store(documents, embeddings, save_path)
    elif store_type.lower() == "chroma":
        if not save_path:
            save_path = Path("data/vectorstore/chroma")
        vector_store = create_chroma_store(documents, embeddings, save_path)
    else:
        raise ValueError(f"Unsupported store type: {store_type}")
    
    # Create retriever (remove score_threshold for FAISS compatibility)
    default_search_kwargs = {"k": 3}
    if search_kwargs:
        default_search_kwargs.update(search_kwargs)
    
    return DocumentRetriever(vector_store, default_search_kwargs)


def format_retrieval_results(documents: List[Document], 
                           with_scores: bool = False,
                           scores: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """
    Format retrieval results for display.
    
    Args:
        documents: List of retrieved documents
        with_scores: Whether to include scores
        scores: List of similarity scores (if with_scores=True)
        
    Returns:
        List of formatted result dictionaries
    """
    results = []
    
    for i, doc in enumerate(documents):
        result = {
            "text": doc.page_content,
            "metadata": doc.metadata,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown")
        }
        
        if with_scores and scores and i < len(scores):
            result["score"] = scores[i]
        
        results.append(result)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    embedding_config = {
        "provider": "huggingface",
        "model": "all-MiniLM-L6-v2"
    }
    
    search_kwargs = {
        "k": 3
    }
    
    # Test data directory
    processed_dir = Path("../../data/processed")
    
    if processed_dir.exists():
        # Create retriever from JSON
        print("Creating retriever from JSON files...")
        retriever = create_retriever_from_json(
            processed_dir=processed_dir,
            embedding_config=embedding_config,
            store_type="faiss",
            save_path=Path("../../data/vectorstore/langchain_faiss"),
            search_kwargs=search_kwargs
        )
        
        # Test retrieval
        test_queries = [
            "What is linear programming?",
            "How do you solve optimization problems?",
            "What is the maximum flow problem?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Retrieve documents
            docs = retriever.retrieve(query, k=2)
            
            # Format and display results
            results = format_retrieval_results(docs)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Source: {result['source']} (Page {result['page']})")
                print(f"   Text: {result['text'][:200]}...")
    else:
        print(f"Processed directory not found: {processed_dir}")
        print("Please run the PDF processing pipeline first.")