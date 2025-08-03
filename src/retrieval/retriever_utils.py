"""
Retriever Utility Functions

This module provides convenience functions for loading and using retrievers
in the RAG system.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from .langchain_retriever import (
    create_retriever_from_existing_store,
    create_retriever_from_json,
    DocumentRetriever
)


def load_retriever_from_store(store_path: str = "data/vectorstore/langchain_faiss",
                             provider: str = "huggingface",
                             model: str = "all-MiniLM-L6-v2",
                             k: int = 3,
                             score_threshold: float = 0.0) -> DocumentRetriever:
    """
    Load a retriever from an existing vector store.
    
    Args:
        store_path: Path to the vector store
        provider: Embedding provider ('huggingface', 'openai')
        model: Model name
        k: Number of documents to retrieve
        score_threshold: Minimum similarity score
        
    Returns:
        DocumentRetriever instance
    """
    embedding_config = {
        "provider": provider,
        "model": model
    }
    
    search_kwargs = {
        "k": k,
        "score_threshold": score_threshold
    }
    
    return create_retriever_from_existing_store(
        store_path=Path(store_path),
        embedding_config=embedding_config,
        store_type="faiss",
        search_kwargs=search_kwargs
    )


def get_langchain_retriever(store_path: str = "data/vectorstore/langchain_faiss",
                           provider: str = "huggingface",
                           model: str = "all-MiniLM-L6-v2",
                           k: int = 3) -> BaseRetriever:
    """
    Get a LangChain BaseRetriever object for use in chains.
    
    Args:
        store_path: Path to the vector store
        provider: Embedding provider
        model: Model name
        k: Number of documents to retrieve
        
    Returns:
        LangChain BaseRetriever instance
    """
    retriever = load_retriever_from_store(
        store_path=store_path,
        provider=provider,
        model=model,
        k=k,
        score_threshold=0.0
    )
    
    return retriever.get_retriever()


def quick_search(query: str,
                k: int = 3,
                store_path: str = "data/vectorstore/langchain_faiss",
                provider: str = "huggingface") -> List[Dict[str, Any]]:
    """
    Perform a quick search without setting up a persistent retriever.
    
    Args:
        query: Search query
        k: Number of results to return
        store_path: Path to the vector store
        provider: Embedding provider
        
    Returns:
        List of search results
    """
    retriever = load_retriever_from_store(
        store_path=store_path,
        provider=provider,
        k=k,
        score_threshold=0.0
    )
    
    docs_with_scores = retriever.retrieve_with_scores(query, k=k)
    
    results = []
    for doc, score in docs_with_scores:
        results.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": score,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown")
        })
    
    return results


def format_search_results(results: List[Dict[str, Any]], 
                         max_text_length: int = 200) -> str:
    """
    Format search results for display.
    
    Args:
        results: List of search results
        max_text_length: Maximum length of text preview
        
    Returns:
        Formatted string
    """
    if not results:
        return "No results found."
    
    formatted = []
    for i, result in enumerate(results, 1):
        text_preview = result["text"][:max_text_length]
        if len(result["text"]) > max_text_length:
            text_preview += "..."
        
        formatted.append(
            f"{i}. Score: {result['score']:.3f}\n"
            f"   Source: {result['source']} (Page {result['page']})\n"
            f"   Text: {text_preview}\n"
        )
    
    return "\n".join(formatted)


# Convenience functions for common use cases
def search_linear_programming(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Search for linear programming related content."""
    return quick_search(f"linear programming {query}", k=k)


def search_optimization(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Search for optimization related content."""
    return quick_search(f"optimization {query}", k=k)


def search_algorithms(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Search for algorithm related content."""
    return quick_search(f"algorithm {query}", k=k)


# Example usage
if __name__ == "__main__":
    # Test the utility functions
    print("🧪 Testing Retriever Utilities")
    print("=" * 40)
    
    # Test quick search
    print("\n🔍 Testing quick search...")
    results = quick_search("What is linear programming?", k=2)
    
    if results:
        print(f"Found {len(results)} results:")
        print(format_search_results(results))
    else:
        print("No results found")
    
    # Test specialized searches
    print("\n🔍 Testing specialized searches...")
    
    lp_results = search_linear_programming("definition", k=1)
    if lp_results:
        print(f"Linear Programming search: {lp_results[0]['score']:.3f} - {lp_results[0]['text'][:100]}...")
    
    opt_results = search_optimization("problems", k=1)
    if opt_results:
        print(f"Optimization search: {opt_results[0]['score']:.3f} - {opt_results[0]['text'][:100]}...")
    
    # Test LangChain retriever
    print("\n🔍 Testing LangChain retriever...")
    try:
        langchain_retriever = get_langchain_retriever(k=2)
        docs = langchain_retriever.invoke("maximum flow")
        print(f"LangChain retriever returned {len(docs)} documents")
        if docs:
            print(f"First result: {docs[0].page_content[:100]}...")
    except Exception as e:
        print(f"Error testing LangChain retriever: {e}")
    
    print("\n✅ Utility tests complete!")