"""
Retrieval Module

This module provides LangChain-based retrieval capabilities for the RAG system.
"""

from .langchain_retriever import (
    DocumentRetriever,
    LangChainEmbeddingAdapter,
    create_retriever_from_existing_store,
    create_retriever_from_json,
    format_retrieval_results,
    load_documents_from_json
)

from .retriever_utils import (
    load_retriever_from_store,
    get_langchain_retriever,
    quick_search,
    format_search_results,
    search_linear_programming,
    search_optimization,
    search_algorithms
)

__all__ = [
    'DocumentRetriever',
    'LangChainEmbeddingAdapter', 
    'create_retriever_from_existing_store',
    'create_retriever_from_json',
    'format_retrieval_results',
    'load_documents_from_json',
    'load_retriever_from_store',
    'get_langchain_retriever',
    'quick_search',
    'format_search_results',
    'search_linear_programming',
    'search_optimization',
    'search_algorithms'
]