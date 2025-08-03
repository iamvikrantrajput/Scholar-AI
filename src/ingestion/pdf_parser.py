"""
PDF Ingestion Module for AI Research Assistant

This module handles the extraction, cleaning, chunking, and storage of PDF documents
for the RAG-based research assistant system.
"""

import fitz  # PyMuPDF
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Main class for processing PDF documents into structured chunks."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Number of words per chunk
            overlap: Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
            
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=]', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[\.]{3,}', '...', text)
        text = re.sub(r'[-]{2,}', '--', text)
        
        # Strip and ensure single spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_text_from_page(self, page) -> str:
        """
        Extract text from a single PDF page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Cleaned text from the page
        """
        try:
            # Get text from page
            text = page.get_text()
            
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"Error extracting text from page: {e}")
            return ""
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        words = text.split()
        
        # If a text is shorter than chunk size, return as single chunk
        if len(words) <= self.chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(words):
            # Get chunk of words
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk = ' '.join(chunk_words)
            
            chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start += self.chunk_size - self.overlap
            
            # Break if we've reached the end
            if end >= len(words):
                break
                
        return chunks
    
    def create_chunk_metadata(self, filename: str, page_num: int, 
                            chunk_index: int, total_chunks: int,
                            word_count: int, tags: Optional[List[str]] = None) -> Dict:
        """
        Create metadata for a text chunk.
        
        Args:
            filename: Source PDF filename
            page_num: Page number (1-indexed)
            chunk_index: Index of chunk within the page
            total_chunks: Total chunks from this page
            word_count: Number of words in this chunk
            tags: Optional tags for categorization
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "source": filename,
            "page": page_num,
            "chunk_index": chunk_index,
            "total_chunks_from_page": total_chunks,
            "word_count": word_count,
            "processed_at": datetime.now().isoformat(),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap
        }
        
        if tags:
            metadata["tags"] = tags
            
        return metadata
    
    def process_pdf(self, pdf_path: Path, tags: Optional[List[str]] = None) -> List[Dict]:
        """
        Process a single PDF file into chunks with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            tags: Optional tags for all chunks from this PDF
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        chunks_with_metadata = []
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text from page
                page_text = self.extract_text_from_page(page)
                
                if not page_text:
                    logger.warning(f"No text found on page {page_num + 1} of {pdf_path.name}")
                    continue
                
                # Split page text into chunks
                page_chunks = self.chunk_text(page_text)
                
                # Create metadata for each chunk
                for chunk_idx, chunk in enumerate(page_chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        metadata = self.create_chunk_metadata(
                            filename=pdf_path.name,
                            page_num=page_num + 1,  # 1-indexed
                            chunk_index=chunk_idx,
                            total_chunks=len(page_chunks),
                            word_count=len(chunk.split()),
                            tags=tags
                        )
                        
                        chunks_with_metadata.append({
                            "text": chunk,
                            "metadata": metadata
                        })
            
            doc.close()
            logger.info(f"Processed {len(chunks_with_metadata)} chunks from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            
        return chunks_with_metadata
    
    def save_chunks(self, chunks: List[Dict], output_path: Path) -> None:
        """
        Save chunks to JSON file.
        
        Args:
            chunks: List of chunk dictionaries
            output_path: Path where to save the JSON file
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(chunks)} chunks to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving chunks to {output_path}: {e}")
    
    def process_directory(self, input_dir: Path, output_dir: Path, 
                         file_pattern: str = "*.pdf") -> Dict[str, int]:
        """
        Process all PDF files in a directory.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save processed JSON files
            file_pattern: Glob pattern for PDF files
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing PDFs from {input_dir}")
        
        pdf_files = list(input_dir.glob(file_pattern))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {input_dir}")
            return {"processed": 0, "total_chunks": 0, "failed": 0}
        
        stats = {"processed": 0, "total_chunks": 0, "failed": 0}
        
        for pdf_path in pdf_files:
            try:
                # Process the PDF
                chunks = self.process_pdf(pdf_path)
                
                if chunks:
                    # Create output filename
                    output_filename = pdf_path.stem + ".json"
                    output_path = output_dir / output_filename
                    
                    # Save chunks
                    self.save_chunks(chunks, output_path)
                    
                    stats["processed"] += 1
                    stats["total_chunks"] += len(chunks)
                else:
                    logger.warning(f"No chunks extracted from {pdf_path.name}")
                    stats["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                stats["failed"] += 1
        
        logger.info(f"Processing complete: {stats}")
        return stats


def main():
    """Main function to run PDF processing."""
    # Set up paths
    raw_data_dir = Path("data/raw")
    processed_data_dir = Path("data/processed")
    
    # Initialize processor
    processor = PDFProcessor(chunk_size=1000, overlap=200)
    
    # Process all PDFs
    stats = processor.process_directory(raw_data_dir, processed_data_dir)
    
    print(f"\n📄 PDF Processing Complete!")
    print(f"✅ Successfully processed: {stats['processed']} files")
    print(f"📊 Total chunks created: {stats['total_chunks']}")
    print(f"❌ Failed to process: {stats['failed']} files")


if __name__ == "__main__":
    main()