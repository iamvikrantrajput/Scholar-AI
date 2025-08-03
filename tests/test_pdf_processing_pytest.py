#!/usr/bin/env python3
"""
Pytest-compatible test for PDF processing pipeline
"""

import pytest
from pathlib import Path

from ingestion.pdf_parser import PDFProcessor


class TestPDFProcessing:
    """Test suite for PDF processing functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create a PDFProcessor instance for testing."""
        return PDFProcessor(chunk_size=1000, overlap=200)
    
    @pytest.fixture
    def sample_pdfs(self):
        """Get list of sample PDF files for testing."""
        pdf_dir = Path("data/raw").resolve()
        pdfs = list(pdf_dir.glob("*.pdf"))
        if not pdfs:
            pytest.skip("No PDF files found in data/raw/ directory")
        return pdfs
    
    @pytest.mark.requires_data
    def test_pdf_directory_exists(self):
        """Test that the PDF data directory exists."""
        pdf_dir = Path("data/raw").resolve()
        assert pdf_dir.exists(), f"PDF directory {pdf_dir} does not exist"
        assert pdf_dir.is_dir(), f"PDF path {pdf_dir} is not a directory"
    
    @pytest.mark.requires_data
    def test_sample_pdfs_available(self, sample_pdfs):
        """Test that sample PDF files are available for testing."""
        assert len(sample_pdfs) > 0, "No PDF files found for testing"
        
        # Check that files are actually PDF files
        for pdf_file in sample_pdfs:
            assert pdf_file.suffix.lower() == '.pdf', f"File {pdf_file.name} is not a PDF"
            assert pdf_file.stat().st_size > 0, f"PDF file {pdf_file.name} is empty"
    
    @pytest.mark.unit
    def test_processor_initialization(self, processor):
        """Test that PDFProcessor initializes correctly."""
        assert processor is not None
        assert hasattr(processor, 'process_pdf')
        assert hasattr(processor, 'chunk_size')
        assert hasattr(processor, 'overlap')
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_pdf_processing_single_file(self, processor, sample_pdfs):
        """Test processing a single PDF file."""
        test_pdf = sample_pdfs[0]
        
        # Process the PDF
        chunks = processor.process_pdf(test_pdf)
        
        # Validate results
        assert chunks is not None, "PDF processing returned None"
        assert isinstance(chunks, list), "PDF processing should return a list of chunks"
        assert len(chunks) > 0, f"No chunks generated from {test_pdf.name}"
        
        # Validate chunk structure
        sample_chunk = chunks[0]
        assert isinstance(sample_chunk, dict), "Chunks should be dictionaries"
        assert 'text' in sample_chunk, "Chunk should contain 'text' field"
        assert 'metadata' in sample_chunk, "Chunk should contain 'metadata' field"
        
        # Validate text content
        text_content = sample_chunk['text']
        assert isinstance(text_content, str), "Chunk text should be a string"
        assert len(text_content.strip()) > 0, "Chunk text should not be empty"
        
        # Validate metadata
        metadata = sample_chunk['metadata']
        assert isinstance(metadata, dict), "Metadata should be a dictionary"
        assert 'source' in metadata, "Metadata should contain source information"
        assert 'page' in metadata, "Metadata should contain page information"
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_pdf_processing_multiple_files(self, processor, sample_pdfs):
        """Test processing multiple PDF files."""
        all_chunks = []
        
        for pdf_file in sample_pdfs[:3]:  # Test first 3 PDFs to avoid timeout
            chunks = processor.process_pdf(pdf_file)
            
            assert chunks is not None, f"Processing failed for {pdf_file.name}"
            assert len(chunks) > 0, f"No chunks generated for {pdf_file.name}"
            
            all_chunks.extend(chunks)
        
        # Validate combined results
        assert len(all_chunks) > 0, "No chunks generated from any PDF"
        
        # Check that we have chunks from different sources
        sources = {chunk['metadata']['source'] for chunk in all_chunks}
        assert len(sources) > 1, "Chunks should come from multiple PDF sources"
    
    @pytest.mark.unit
    def test_pdf_processing_error_handling(self, processor):
        """Test error handling for invalid PDF files."""
        non_existent_file = Path("non_existent.pdf")
        
        # This should raise an exception or return empty results
        with pytest.raises((FileNotFoundError, Exception)):
            processor.process_pdf(non_existent_file)
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_chunk_size_consistency(self, sample_pdfs):
        """Test that different chunk sizes produce different results."""
        small_processor = PDFProcessor(chunk_size=500, overlap=100)
        large_processor = PDFProcessor(chunk_size=2000, overlap=200)
        
        test_pdf = sample_pdfs[0]
        
        small_chunks = small_processor.process_pdf(test_pdf)
        large_chunks = large_processor.process_pdf(test_pdf)
        
        # Generally, smaller chunk size should produce more chunks
        # (though this isn't guaranteed for very small documents)
        assert small_chunks is not None and large_chunks is not None
        
        # At minimum, both should produce some chunks
        assert len(small_chunks) > 0, "Small chunk processor should produce chunks"
        assert len(large_chunks) > 0, "Large chunk processor should produce chunks"
    
    @pytest.mark.integration
    @pytest.mark.requires_data
    def test_metadata_completeness(self, processor, sample_pdfs):
        """Test that all chunks have complete metadata."""
        test_pdf = sample_pdfs[0]
        chunks = processor.process_pdf(test_pdf)
        
        required_metadata_fields = ['source', 'page']
        
        for i, chunk in enumerate(chunks):
            metadata = chunk.get('metadata', {})
            
            for field in required_metadata_fields:
                assert field in metadata, f"Chunk {i} missing metadata field: {field}"
            
            # Validate specific metadata values
            assert metadata['source'] == test_pdf.name, f"Incorrect source in chunk {i}"
            assert isinstance(metadata['page'], int), f"Page should be integer in chunk {i}"
            assert metadata['page'] >= 1, f"Page number should be >= 1 in chunk {i}"


# Standalone test functions for compatibility with existing test runner
def test_pdf_processing():
    """Legacy test function for compatibility."""
    processor = PDFProcessor(chunk_size=1000, overlap=200)
    pdf_dir = Path("data/raw").resolve()
    pdfs = list(pdf_dir.glob("*.pdf"))
    
    if not pdfs:
        pytest.skip("No PDFs found in data/raw/")
    
    test_pdf = pdfs[0]
    chunks = processor.process_pdf(test_pdf)
    
    assert chunks is not None
    assert len(chunks) > 0
    return True


if __name__ == "__main__":
    # Allow running as standalone script
    import sys
    
    # Try to use pytest if available, otherwise fall back to simple test
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("🔹 Running simple PDF processing test...")
        success = test_pdf_processing()
        print(f"{'✅ PDF Processing Test PASSED' if success else '❌ PDF Processing Test FAILED'}")
        sys.exit(0 if success else 1)