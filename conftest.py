"""
Pytest configuration and shared fixtures for AI Research Assistant tests.
"""

import sys
import pytest
from pathlib import Path
from typing import List, Dict, Any

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Get the data directory path."""
    return project_root / "data"


@pytest.fixture(scope="session")
def raw_data_dir(data_dir):
    """Get the raw data directory path."""
    return data_dir / "raw"


@pytest.fixture(scope="session")
def processed_data_dir(data_dir):
    """Get the processed data directory path."""
    return data_dir / "processed"


@pytest.fixture(scope="session")
def vector_store_dir(data_dir):
    """Get the vector store directory path."""
    return data_dir / "vectorstore"


@pytest.fixture(scope="session")
def sample_pdfs(raw_data_dir):
    """Get list of sample PDF files for testing."""
    if not raw_data_dir.exists():
        pytest.skip(f"Raw data directory {raw_data_dir} does not exist")
    
    pdfs = list(raw_data_dir.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"No PDF files found in {raw_data_dir}")
    
    return pdfs


@pytest.fixture(scope="session")
def sample_processed_files(processed_data_dir):
    """Get list of processed JSON files for testing."""
    if not processed_data_dir.exists():
        pytest.skip(f"Processed data directory {processed_data_dir} does not exist")
    
    json_files = list(processed_data_dir.glob("*.json"))
    if not json_files:
        pytest.skip(f"No processed JSON files found in {processed_data_dir}")
    
    return json_files


@pytest.fixture
def mock_llm_responses():
    """Provide mock LLM responses for testing."""
    return [
        "Based on the provided context, linear programming is a mathematical optimization technique used to find the optimal solution within a set of linear constraints. It is commonly used in operations research and mathematical optimization problems. (source: L01.pdf, page 1)",
        "The dual of a linear program provides important insights into the optimization problem structure. The dual theorem establishes the relationship between primal and dual optimal values. (source: L02.pdf, page 1)",
        "Optimization techniques in linear programming include the simplex method and interior point methods, which efficiently solve large-scale problems. (source: L03.pdf, page 1)"
    ]


@pytest.fixture
def test_config():
    """Provide test configuration settings."""
    return {
        "embeddings": {
            "provider": "huggingface",
            "model": "all-MiniLM-L6-v2"
        },
        "vector_store": {
            "type": "faiss",
            "index_type": "IndexFlatIP"
        },
        "llm": {
            "provider": "mock",
            "temperature": 0.1
        },
        "chunk_size": 1000,
        "overlap": 200,
        "batch_size": 5
    }


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for component interaction")
    config.addinivalue_line("markers", "e2e: End-to-end tests for complete workflows")
    config.addinivalue_line("markers", "slow: Tests that take more than 10 seconds")
    config.addinivalue_line("markers", "requires_data: Tests that require sample data files")
    config.addinivalue_line("markers", "requires_gpu: Tests that benefit from GPU acceleration")
    config.addinivalue_line("markers", "requires_api_key: Tests that require API keys")


def pytest_collection_modifyitems(config, items):
    """Modify test items based on markers and conditions."""
    # Add skip markers based on conditions
    skip_no_data = pytest.mark.skip(reason="No test data available")
    skip_no_gpu = pytest.mark.skip(reason="No GPU available")
    skip_no_api_key = pytest.mark.skip(reason="No API key configured")
    
    # Check for data availability
    data_dir = Path("data")
    has_data = (data_dir / "raw").exists() and list((data_dir / "raw").glob("*.pdf"))
    
    # Check for GPU (simplified check)
    has_gpu = False
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        pass
    
    # Check for API keys
    import os
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    
    for item in items:
        # Skip tests requiring data if no data available
        if "requires_data" in item.keywords and not has_data:
            item.add_marker(skip_no_data)
        
        # Skip GPU tests if no GPU available
        if "requires_gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_no_gpu)
        
        # Skip API tests if no API key available
        if "requires_api_key" in item.keywords and not has_openai_key:
            item.add_marker(skip_no_api_key)


# Custom assertions
def assert_valid_chunk(chunk: Dict[str, Any]) -> None:
    """Assert that a chunk has valid structure."""
    assert isinstance(chunk, dict), "Chunk should be a dictionary"
    assert 'text' in chunk, "Chunk should contain 'text' field"
    assert 'metadata' in chunk, "Chunk should contain 'metadata' field"
    
    text = chunk['text']
    assert isinstance(text, str), "Chunk text should be a string"
    assert len(text.strip()) > 0, "Chunk text should not be empty"
    
    metadata = chunk['metadata']
    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    assert 'source' in metadata, "Metadata should contain source information"


def assert_valid_qa_response(response: Dict[str, Any]) -> None:
    """Assert that a QA response has valid structure."""
    assert isinstance(response, dict), "Response should be a dictionary"
    assert 'answer' in response, "Response should contain 'answer' field"
    
    answer = response['answer']
    assert isinstance(answer, str), "Answer should be a string"
    assert len(answer.strip()) > 0, "Answer should not be empty"


# Pytest helpers for test discovery
def pytest_sessionstart(session):
    """Actions to perform at the start of test session."""
    print("\n🧪 Starting AI Research Assistant Test Suite")
    print("=" * 60)


def pytest_sessionfinish(session, exitstatus):
    """Actions to perform at the end of test session."""
    status = "✅ PASSED" if exitstatus == 0 else "❌ FAILED"
    print(f"\n{status}: Test suite completed with exit status {exitstatus}")
    print("=" * 60)