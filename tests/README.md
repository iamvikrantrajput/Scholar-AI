# 🧪 Test Suite - AI Research Assistant

This directory contains comprehensive tests for all components of the AI Research Assistant system.

## Test Structure

```
tests/
├── README.md                    # This file
├── test_pdf_processing.py       # PDF parsing and chunking pipeline
├── test_vector_store.py         # Embedding generation and vector storage
├── test_qa_pipeline.py          # End-to-end QA system with mock LLM
├── test_custom_prompts.py       # Custom prompt template system
├── test_streamlit_components.py # Streamlit UI components
├── test_langchain_retrieval.py  # LangChain retrieval integration
├── test_qa_system.py           # QA chain functionality
└── test_prompt_comparison.py   # Prompt style comparison
```

## Running Tests

### Individual Tests
```bash
# Run a specific test
poetry run python tests/test_pdf_processing.py
poetry run python tests/test_vector_store.py
poetry run python tests/test_qa_pipeline.py
```

### All Tests
```bash
# Run all tests with the test runner
poetry run python run_tests.py

# Or use the comprehensive validation
poetry run python final_validation.py
```

### Test Categories

#### 1. **Core Pipeline Tests**
- `test_pdf_processing.py` - Tests PDF parsing, text extraction, and chunking
- `test_vector_store.py` - Tests embedding generation and FAISS/ChromaDB storage
- `test_qa_pipeline.py` - Tests end-to-end question answering with citations

#### 2. **Integration Tests**
- `test_langchain_retrieval.py` - Tests LangChain retriever integration
- `test_qa_system.py` - Tests QA chain with real/mock LLMs
- `test_streamlit_components.py` - Tests Streamlit UI components

#### 3. **Feature Tests**
- `test_custom_prompts.py` - Tests custom prompt template system
- `test_prompt_comparison.py` - Compares different response styles

## Test Requirements

### Data Requirements
Tests expect the following data structure:
```
data/
├── raw/           # PDF files for testing (4 sample files)
├── processed/     # JSON chunks from processed PDFs
└── vectorstore/   # FAISS indexes and metadata
```

### Environment Requirements
- Python 3.10+
- Poetry virtual environment
- All project dependencies installed
- Optional: GPU for faster embedding tests

## Test Output Examples

### Successful Test Run
```
🔹 Testing PDF Parsing & Chunking Pipeline
==================================================
📂 Looking in: /path/to/data/raw
📄 Found PDFs: 4
🧪 Testing with: L01.pdf
✅ Successfully processed L01.pdf
📊 Generated 15 chunks
📝 Sample chunk keys: ['text', 'metadata']
📄 Sample metadata: {'source': 'L01.pdf', 'page': 1}
📐 Sample text length: 985

✅ PDF Processing Test PASSED
```

### Failed Test Example
```
❌ Error in vector store creation: No module named 'faiss'
Vector Store Test FAILED
```

## Test Configuration

### Mock LLM Responses
Tests use `FakeListLLM` from LangChain for consistent, fast testing:
```python
mock_responses = [
    "Based on the provided context, linear programming is...",
    "The dual of a linear program provides...",
    "Optimization techniques include..."
]
```

### Test Data Paths
Tests automatically resolve paths relative to project root:
```python
pdf_dir = Path("data/raw").resolve()
processed_dir = Path("data/processed").resolve()
vector_store_dir = Path("data/vectorstore/test_faiss").resolve()
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'X'**
```bash
# Ensure Poetry environment is active
poetry shell
poetry install
```

**No PDFs found in data/raw/**
```bash
# Ensure test data exists
ls data/raw/*.pdf
# If missing, add sample PDFs to data/raw/
```

**Vector store creation fails**
```bash
# Check if processed data exists
ls data/processed/*.json
# Run PDF processing first if needed
poetry run python tests/test_pdf_processing.py
```

**Permission errors**
```bash
# Ensure write permissions for data directories
chmod -R 755 data/
```

### Performance Notes

- **PDF Processing**: ~2-5 seconds per PDF
- **Vector Store Creation**: ~10-30 seconds depending on data size
- **QA Pipeline**: ~1-3 seconds with mock LLM
- **Full Test Suite**: ~60-120 seconds

### GPU Acceleration
If CUDA GPU is available:
- Embedding generation will automatically use GPU
- Expect 2-3x faster vector store creation
- Monitor GPU memory usage with `nvidia-smi`

## Adding New Tests

### Test Template
```python
#!/usr/bin/env python3
"""
Test description
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_your_functionality():
    """Test your specific functionality."""
    print("🔹 Testing Your Functionality")
    print("=" * 40)
    
    try:
        # Your test logic here
        print("✅ Test passed")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_your_functionality()
    print(f"{'✅ Test PASSED' if success else '❌ Test FAILED'}")
```

### Test Guidelines
1. **Descriptive names**: Use clear, descriptive test function names
2. **Error handling**: Always wrap tests in try/except blocks
3. **Clear output**: Use emojis and clear status messages
4. **Cleanup**: Clean up any temporary files created during tests
5. **Documentation**: Add docstrings explaining what each test does

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install Poetry
        run: pip install poetry
      - name: Install dependencies
        run: poetry install
      - name: Run tests
        run: poetry run python run_tests.py
```

### Test Coverage
Current test coverage:
- ✅ PDF Processing Pipeline
- ✅ Vector Store Creation
- ✅ QA System Integration
- ✅ Custom Prompts
- ✅ Streamlit Components
- ✅ LangChain Retrieval
- ✅ Mock LLM Testing

## Contributing

When adding new features:
1. Write corresponding tests in `tests/`
2. Update this README if needed
3. Ensure all tests pass with `run_tests.py`
4. Add your test to the validation suite

---

**Need help?** Check the main [README](../README.md) or run the [validation script](../final_validation.py) to diagnose issues.