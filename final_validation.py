#!/usr/bin/env python3
"""
🔹 Final Project Validation & Setup Check
Complete end-to-end validation of the AI Research Assistant
"""

import sys
import warnings
import os
import logging

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set logging levels to reduce noise
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

sys.path.append('src')

from pathlib import Path
import subprocess
from typing import Dict, List, Tuple

def run_validation_check(name: str, check_function) -> Tuple[bool, str]:
    """Run a validation check and return result."""
    try:
        result = check_function()
        if isinstance(result, tuple):
            success, message = result
        else:
            success, message = result, "✅ Passed"
        return success, message
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def check_environment() -> Tuple[bool, str]:
    """Check Python version and Poetry setup."""
    import sys
    
    # Check Python version
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        return False, f"❌ Python {version.major}.{version.minor} found, need 3.10+"
    
    # Check Poetry
    try:
        result = subprocess.run(['poetry', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, "❌ Poetry not found"
    except:
        return False, "❌ Poetry not accessible"
    
    return True, f"✅ Python {version.major}.{version.minor}.{version.micro} + Poetry"

def check_dependencies() -> Tuple[bool, str]:
    """Check critical dependencies."""
    critical_deps = [
        'streamlit', 'langchain', 'faiss', 'sentence_transformers', 
        'pymupdf', 'openai', 'chromadb'
    ]
    
    missing = []
    for dep in critical_deps:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing.append(dep)
    
    if missing:
        return False, f"❌ Missing: {', '.join(missing)}"
    
    return True, f"✅ All {len(critical_deps)} critical dependencies"

def check_project_structure() -> Tuple[bool, str]:
    """Check project folder structure."""
    required_paths = [
        'src/app/streamlit_chat.py',
        'src/app/cli_chat.py', 
        'src/ingestion/pdf_parser.py',
        'src/vectorstore/embeddings.py',
        'src/vectorstore/vector_store.py',
        'src/llm/qa_chain.py',
        'src/llm/prompts.py',
        'src/retrieval/langchain_retriever.py',
        'data/raw',
        'data/processed',
        'data/vectorstore',
        'pyproject.toml'
    ]
    
    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)
    
    if missing:
        return False, f"❌ Missing: {', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}"
    
    return True, f"✅ All {len(required_paths)} required paths"

def check_data_files() -> Tuple[bool, str]:
    """Check for test data."""
    raw_pdfs = list(Path('data/raw').glob('*.pdf'))
    processed_json = list(Path('data/processed').glob('*.json'))
    vector_stores = list(Path('data/vectorstore').iterdir())
    
    if not raw_pdfs:
        return False, "❌ No PDFs in data/raw/"
    
    if not processed_json:
        return False, "❌ No processed JSON files"
        
    if not vector_stores:
        return False, "❌ No vector stores found"
    
    return True, f"✅ {len(raw_pdfs)} PDFs, {len(processed_json)} processed, {len(vector_stores)} stores"

def check_pdf_processing() -> Tuple[bool, str]:
    """Test PDF processing pipeline."""
    from ingestion.pdf_parser import PDFProcessor
    
    processor = PDFProcessor()
    pdfs = list(Path('data/raw').glob('*.pdf'))
    
    if not pdfs:
        return False, "❌ No test PDFs available"
    
    try:
        chunks = processor.process_pdf(pdfs[0])
        if not chunks:
            return False, "❌ No chunks generated"
        return True, f"✅ Generated {len(chunks)} chunks from {pdfs[0].name}"
    except Exception as e:
        return False, f"❌ Processing failed: {str(e)[:50]}"

def check_vector_store() -> Tuple[bool, str]:
    """Test vector store functionality."""
    from vectorstore.vector_store import create_vector_store_manager
    
    try:
        config = {
            "embeddings": {"provider": "huggingface", "model": "all-MiniLM-L6-v2"},
            "vector_store": {"type": "faiss", "index_type": "IndexFlatIP"}
        }
        
        vector_manager = create_vector_store_manager(config)
        results = vector_manager.search("test query", k=1)
        
        return True, f"✅ Vector search returned {len(results)} results"
    except Exception as e:
        return False, f"❌ Vector store failed: {str(e)[:50]}"

def check_qa_pipeline() -> Tuple[bool, str]:
    """Test QA pipeline with mock LLM."""
    from langchain_community.llms import FakeListLLM
    from retrieval import get_langchain_retriever
    from llm.qa_chain import QuestionAnswerer
    
    try:
        mock_llm = FakeListLLM(responses=["Test response with citation (source: test.pdf, page 1)"])
        retriever = get_langchain_retriever(k=1)
        
        qa_system = QuestionAnswerer(
            llm=mock_llm,
            retriever=retriever,
            prompt_style="academic"
        )
        
        response = qa_system.answer("What is a test?")
        
        if not response or not response.get('answer'):
            return False, "❌ No response generated"
            
        return True, f"✅ QA pipeline generated response"
    except Exception as e:
        return False, f"❌ QA pipeline failed: {str(e)}"

def check_streamlit_app() -> Tuple[bool, str]:
    """Test Streamlit app components."""
    try:
        from app.streamlit_chat import ScholarAIApp
        app = ScholarAIApp()
        return True, "✅ Streamlit app components loaded"
    except Exception as e:
        return False, f"❌ Streamlit app failed: {str(e)[:50]}"

def check_cli_interface() -> Tuple[bool, str]:
    """Test CLI interface."""
    try:
        from app.cli_chat import single_question_mode
        # Just test import, not execution
        return True, "✅ CLI interface ready"
    except Exception as e:
        return False, f"❌ CLI failed: {str(e)[:50]}"

def check_custom_prompts() -> Tuple[bool, str]:
    """Test custom prompt system."""
    try:
        from llm.prompts import PromptTemplateManager
        
        manager = PromptTemplateManager()
        templates = manager.list_templates()
        
        if len(templates) < 5:
            return False, f"❌ Only {len(templates)} prompt templates"
            
        return True, f"✅ {len(templates)} prompt templates available"
    except Exception as e:
        return False, f"❌ Prompts failed: {str(e)[:50]}"

def main():
    """Run complete validation suite."""
    print("🔹 AI Research Assistant - Final Validation")
    print("=" * 60)
    print("Checking all components for end-to-end functionality...")
    print()
    
    # Define all validation checks
    checks = [
        ("Environment Setup", check_environment),
        ("Dependencies", check_dependencies), 
        ("Project Structure", check_project_structure),
        ("Data Files", check_data_files),
        ("PDF Processing", check_pdf_processing),
        ("Vector Store", check_vector_store),
        ("QA Pipeline", check_qa_pipeline),
        ("Custom Prompts", check_custom_prompts),
        ("Streamlit App", check_streamlit_app),
        ("CLI Interface", check_cli_interface)
    ]
    
    # Run all checks
    results = []
    for name, check_func in checks:
        print(f"🔍 {name}...", end=" ")
        success, message = run_validation_check(name, check_func)
        results.append((name, success, message))
        print(message)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"📊 Validation Results: {passed}/{total} checks passed")
    print()
    
    if passed == total:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print()
        print("✅ Your AI Research Assistant is fully functional and ready to use!")
        print()
        print("🚀 Launch Commands:")
        print("   Web Interface:  poetry run streamlit run streamlit_app.py")
        print("   CLI Interface:  poetry run python chat.py --llm-provider huggingface")
        print("   Quick Question: poetry run python ask.py \"your question\"")
        print()
        print("📚 Documentation:")
        print("   README.md - Complete setup and usage guide")
        # print("   STREAMLIT_GUIDE.md - Detailed web interface guide")
        # print("   examples.md - Command examples and workflows")

    else:
        print("⚠️  Some validation checks failed:")
        print()
        for name, success, message in results:
            if not success:
                print(f"   • {name}: {message}")
        print()
        print("🔧 Please fix the failing components before using the system.")
    
    print("\n" + "=" * 60)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)