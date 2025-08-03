#!/usr/bin/env python3
"""
Test PDF processing pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from ingestion.pdf_parser import PDFProcessor

def test_pdf_processing():
    """Test the PDF processing pipeline."""
    print("🔹 Testing PDF Parsing & Chunking Pipeline")
    print("=" * 50)
    
    # Initialize processor
    processor = PDFProcessor(chunk_size=1000, overlap=200)
    
    # Check for PDFs
    pdf_dir = Path("data/raw").resolve()
    pdfs = list(pdf_dir.glob("*.pdf"))
    
    print(f"📂 Looking in: {pdf_dir.absolute()}")
    print(f"📄 Found PDFs: {len(pdfs)}")
    
    if not pdfs:
        print("⚠️  No PDFs found in data/raw/")
        return False
    
    # Test processing one PDF
    test_pdf = pdfs[0]
    print(f"🧪 Testing with: {test_pdf.name}")
    
    try:
        chunks = processor.process_pdf(test_pdf)
        print(f"✅ Successfully processed {test_pdf.name}")
        print(f"📊 Generated {len(chunks)} chunks")
        
        # Check chunk structure
        if chunks:
            sample_chunk = chunks[0]
            print(f"📝 Sample chunk keys: {list(sample_chunk.keys())}")
            print(f"📄 Sample metadata: {sample_chunk.get('metadata', {})}")
            print(f"📐 Sample text length: {len(sample_chunk.get('text', ''))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pdf_processing()
    print(f"\n{'✅ PDF Processing Test PASSED' if success else '❌ PDF Processing Test FAILED'}")