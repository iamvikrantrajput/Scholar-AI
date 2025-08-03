"""
LLM Module

This module provides language model integrations for the RAG system.
"""

from .qa_chain import (
    create_qa_chain,
    answer_question,
    QuestionAnswerer
)

__all__ = [
    'create_qa_chain',
    'answer_question', 
    'QuestionAnswerer'
]