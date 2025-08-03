#!/usr/bin/env python3
"""
Test script for the QA system with fallback options
"""

import sys
from pathlib import Path
sys.path.append('src')

from llm.qa_chain import answer_question, create_qa_chain


def test_with_openai():
    """Test with OpenAI if API key is available."""
    print("🧪 Testing with OpenAI...")
    try:
        response = answer_question("What is linear programming?")
        print("✅ OpenAI test successful!")
        print(f"Answer: {response['answer'][:100]}...")
        print(f"Sources: {len(response.get('sources', []))}")
        return True
    except Exception as e:
        print(f"❌ OpenAI test failed: {e}")
        return False


def test_with_mock_llm():
    """Test with a mock LLM for basic functionality."""
    print("\n🧪 Testing with Mock LLM...")
    
    try:
        # Import LangChain's FakeListLLM for testing
        from langchain_community.llms import FakeListLLM
        from retrieval import get_langchain_retriever
        from llm.qa_chain import QuestionAnswerer
        
        # Create a mock LLM that returns predefined responses
        mock_responses = [
            "Linear programming is a mathematical optimization technique used to find the best solution from a set of linear constraints.",
            "Optimization problems are solved using various algorithms including simplex method and interior-point methods.",
            "The maximum flow problem seeks to find the maximum flow through a network from source to sink."
        ]
        
        mock_llm = FakeListLLM(responses=mock_responses)
        
        # Get retriever
        retriever = get_langchain_retriever(k=3)
        
        # Create QA chain
        qa_chain = QuestionAnswerer(
            llm=mock_llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        # Test questions
        test_questions = [
            "What is linear programming?",
            "How do you solve optimization problems?",
            "Explain the maximum flow problem."
        ]
        
        for question in test_questions:
            print(f"\n🔍 Question: {question}")
            response = qa_chain.answer(question)
            print(f"💡 Answer: {response['answer']}")
            print(f"📚 Sources: {len(response.get('sources', []))} documents")
            
            if response.get('sources'):
                for i, source in enumerate(response['sources'][:2], 1):
                    print(f"   {i}. {source['source']} (Page {source['page']})")
        
        print("\n✅ Mock LLM test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Mock LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_retrieval_only():
    """Test just the retrieval component."""
    print("\n🧪 Testing Retrieval Only...")
    
    try:
        from retrieval import quick_search
        
        test_queries = [
            "What is linear programming?",
            "maximum flow problem",
            "optimization algorithms"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            results = quick_search(query, k=2)
            
            if results:
                print(f"📊 Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Score: {result['score']:.3f} - {result['source']}")
            else:
                print("❌ No results found")
        
        print("\n✅ Retrieval test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🤖 AI Research Assistant QA System Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Retrieval only
    if test_retrieval_only():
        tests_passed += 1
    
    # Test 2: Mock LLM
    if test_with_mock_llm():
        tests_passed += 1
    
    # Test 3: OpenAI (if available)
    if test_with_openai():
        tests_passed += 1
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! QA system is ready.")
    elif tests_passed >= 1:
        print("⚠️ Partial success. Basic functionality works.")
    else:
        print("❌ All tests failed. Check configuration.")


if __name__ == "__main__":
    main()