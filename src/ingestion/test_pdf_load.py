import fitz  # PyMuPDF
from pathlib import Path

def test_load_sample():
    sample_pdf = Path("data/raw/sample.pdf")
    assert sample_pdf.exists(), "❌ Please add a sample PDF to data/raw first"
    
    doc = fitz.open(sample_pdf)
    print(f"✅ Loaded {len(doc)} pages from sample.pdf")

if __name__ == "__main__":
    test_load_sample()