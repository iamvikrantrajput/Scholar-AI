#!/usr/bin/env python3
"""
Test complete QA pipeline with mock LLM
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from langchain_community.llms import FakeListLLM
from retrieval import get_langchain_retriever
from llm.qa_chain import QuestionAnswerer

def test_qa_pipeline():
    """Test the complete QA pipeline."""
    print("🔹 Testing Complete QA Pipeline")
    print("=" * 40)
    
    try:
        # Create mock LLM with realistic responses
        mock_responses = [
            "Based on the provided context, linear programming is a mathematical optimization technique used to find the optimal solution within a set of linear constraints. It is commonly used in operations research and mathematical optimization problems. (source: L01.pdf, page 1)",
            "The dual of a linear program provides important insights into the optimization problem structure. The dual theorem establishes the relationship between primal and dual optimal values. (source: L02.pdf, page 1)",
            "Optimization techniques in linear programming include the simplex method and interior point methods, which efficiently solve large-scale problems. (source: L03.pdf, page 1)"
        ]
        
        mock_llm = FakeListLLM(responses=mock_responses)
        print("✅ Mock LLM created")
        
        # Create retriever
        retriever = get_langchain_retriever(k=3)
        print("✅ Retriever created")
        
        # Create QA chain
        qa_system = QuestionAnswerer(
            llm=mock_llm,
            retriever=retriever,
            return_source_documents=True,
            prompt_style="academic"
        )
        print("✅ QA system initialized")
        
        # Test questions
        test_questions = [
            "What is linear programming?",
            "What is the dual of a linear program?",
            "What optimization techniques are used?"
        ]
        
        print("\n🧪 Testing Questions:")
        print("-" * 30)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Question: {question}")
            
            response = qa_system.answer(question)
            
            print(f"   Answer: {response['answer']}")
            print(f"   Sources: {len(response.get('sources', []))} documents")
            
            # Show source details
            if response.get('sources'):
                for j, source in enumerate(response['sources'][:2], 1):
                    print(f"     {j}. {source['source']} (Page {source['page']})")
        
        print("\n✅ QA Pipeline Test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error in QA pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qa_pipeline()
    print(f"\n{'🎉 Complete QA Pipeline WORKING!' if success else '❌ QA Pipeline Test FAILED'}")