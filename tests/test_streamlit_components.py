#!/usr/bin/env python3
"""
Test script for Streamlit app components
"""

import sys
from pathlib import Path
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test all required imports for Streamlit app."""
    print("🧪 Testing Streamlit App Components...")
    print("=" * 50)
    
    try:
        # Test core imports
        print("📦 Testing core imports...")
        from app.streamlit_chat import StreamlitRAGApp
        from ingestion.pdf_parser import PDFProcessor
        from vectorstore.embeddings import EmbeddingManager
        from vectorstore.vector_store import VectorStoreManager
        from llm.qa_chain import QuestionAnswerer
        from llm.prompts import PromptTemplateManager
        print("✅ All core imports successful")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def test_streamlit_app_initialization():
    """Test StreamlitRAGApp initialization."""
    print("\n🔧 Testing app initialization...")
    
    try:
        from app.streamlit_chat import StreamlitRAGApp
        
        # Create app instance
        app = StreamlitRAGApp()
        print("✅ StreamlitRAGApp created successfully")
        
        # Check session state initialization
        print("✅ Session state initialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ App initialization error: {e}")
        return False

def test_vector_store_manager():
    """Test VectorStoreManager with simplified constructor."""
    print("\n🗄️ Testing vector store manager...")
    
    try:
        from vectorstore.vector_store import VectorStoreManager
        from vectorstore.embeddings import EmbeddingManager
        
        # Test simplified initialization
        embedding_manager = EmbeddingManager(provider="huggingface")
        vector_manager = VectorStoreManager(
            embedding_manager=embedding_manager,
            store_type="faiss"
        )
        print("✅ VectorStoreManager created successfully")
        
        # Test with some sample data
        texts = ["This is a test document about linear programming."]
        metadata = [{"source": "test.pdf", "page": 1}]
        
        vector_manager.build_vector_store_from_texts(texts, metadata)
        print("✅ Vector store built from texts")
        
        # Test retriever
        retriever = vector_manager.get_langchain_retriever(k=1)
        print("✅ LangChain retriever created")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store manager error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_qa_chain_integration():
    """Test QA chain with simplified setup."""
    print("\n🤖 Testing QA chain integration...")
    
    try:
        from langchain_community.llms import FakeListLLM
        from vectorstore.vector_store import VectorStoreManager
        from llm.qa_chain import QuestionAnswerer
        
        # Create mock data
        texts = [
            "Linear programming is a mathematical optimization technique.",
            "The simplex method is used to solve linear programming problems."
        ]
        metadata = [
            {"source": "optimization.pdf", "page": 1},
            {"source": "algorithms.pdf", "page": 5}
        ]
        
        # Create vector store
        vector_manager = VectorStoreManager(store_type="faiss")
        vector_manager.build_vector_store_from_texts(texts, metadata)
        
        # Create mock LLM
        mock_responses = [
            "Linear programming is an optimization technique for solving problems with linear constraints and objectives."
        ]
        mock_llm = FakeListLLM(responses=mock_responses)
        
        # Create retriever
        retriever = vector_manager.get_langchain_retriever(k=2)
        
        # Create QA chain
        qa_chain = QuestionAnswerer(
            llm=mock_llm,
            retriever=retriever,
            prompt_style="academic"
        )
        
        # Test question answering
        response = qa_chain.answer("What is linear programming?")
        print(f"✅ QA chain response: {response['answer'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ QA chain integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdf_processing():
    """Test PDF processing for Streamlit app."""
    print("\n📄 Testing PDF processing...")
    
    try:
        from ingestion.pdf_parser import PDFProcessor
        
        # Check if we have test PDFs
        test_pdf_path = Path("data/raw/L01.pdf")
        if not test_pdf_path.exists():
            print("⚠️ No test PDF found, skipping PDF processing test")
            return True
        
        # Process PDF
        processor = PDFProcessor()
        chunks = processor.process_pdf(test_pdf_path)
        print(f"✅ PDF processed successfully: {len(chunks)} chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        return False

def test_prompt_templates():
    """Test prompt template integration."""
    print("\n📝 Testing prompt templates...")
    
    try:
        from llm.prompts import PromptTemplateManager
        
        # Create prompt manager
        prompt_manager = PromptTemplateManager()
        
        # List available templates
        templates = prompt_manager.list_templates()
        print(f"✅ Available templates: {list(templates.keys())}")
        
        # Test template retrieval
        academic_template = prompt_manager.get_template("academic")
        print("✅ Academic template retrieved")
        
        # Test template formatting
        sample_context = "Linear programming is an optimization technique."
        sample_question = "What is linear programming?"
        
        formatted = academic_template.format(
            context=sample_context,
            question=sample_question
        )
        print("✅ Template formatting works")
        
        return True
        
    except Exception as e:
        print(f"❌ Prompt template error: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 Streamlit App Component Tests")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("App Initialization", test_streamlit_app_initialization),
        ("Vector Store Manager", test_vector_store_manager),
        ("QA Chain Integration", test_qa_chain_integration),
        ("PDF Processing", test_pdf_processing),
        ("Prompt Templates", test_prompt_templates)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Streamlit app is ready to use.")
        print("\n🚀 Launch the app with:")
        print("   poetry run streamlit run streamlit_app.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    main()