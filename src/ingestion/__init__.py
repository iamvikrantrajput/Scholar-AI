"""
PDF Ingestion Module

This module provides functionality for processing PDF documents into
structured text chunks suitable for RAG systems.
"""

from .pdf_parser import PDFProcessor

__all__ = ['PDFProcessor']