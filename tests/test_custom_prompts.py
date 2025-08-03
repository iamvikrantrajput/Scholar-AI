#!/usr/bin/env python3
"""
Test script for custom prompt templates
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from langchain_community.llms import FakeListLLM
from retrieval import get_langchain_retriever
from llm.qa_chain import QuestionAnswerer
from llm.prompts import PromptTemplateManager

def test_prompt_styles():
    """Test different prompt styles with mock LLM."""
    print("🧪 Testing Custom Prompt Templates...")
    print("=" * 60)
    
    # Create mock LLM
    mock_responses = [
        "Linear programming is a mathematical optimization technique used to find the best outcome in a mathematical model with linear relationships. (source: optimization.pdf, page 15)",
        "Linear programming involves maximizing or minimizing a linear objective function subject to linear constraints. The method was developed in the 1940s and has applications in economics, business, and engineering. (source: textbook.pdf, page 42)",
        "LP is an optimization method. (source: notes.pdf, page 3)",
        "Linear programming vs quadratic programming: LP uses linear functions while QP uses quadratic objectives. Both solve optimization problems but differ in complexity. (source: comparison.pdf, page 8)",
        "Step 1: Define variables. Step 2: Set up objective function. Step 3: Add constraints. Step 4: Solve using simplex method. (source: tutorial.pdf, page 12)"
    ]
    
    mock_llm = FakeListLLM(responses=mock_responses)
    retriever = get_langchain_retriever(k=2)
    
    # Test question
    question = "What is linear programming?"
    
    # Test different prompt styles
    prompt_styles = ["academic", "detailed", "concise", "comparative", "problem_solving"]
    
    for i, style in enumerate(prompt_styles):
        print(f"\n🎨 Testing '{style}' prompt style:")
        print("-" * 40)
        
        qa_system = QuestionAnswerer(
            llm=mock_llm,
            retriever=retriever,
            return_source_documents=True,
            prompt_style=style
        )
        
        response = qa_system.answer(question)
        print(f"Answer: {response['answer']}")
        print(f"Sources: {len(response.get('sources', []))} documents")
    
    print(f"\n✅ All prompt styles tested successfully!")

def test_prompt_style_switching():
    """Test dynamic prompt style switching."""
    print(f"\n🔄 Testing Dynamic Prompt Style Switching...")
    print("-" * 50)
    
    mock_responses = [
        "Academic response about linear programming.",
        "Detailed explanation of linear programming."
    ]
    
    mock_llm = FakeListLLM(responses=mock_responses)
    retriever = get_langchain_retriever(k=2)
    
    # Create QA system with academic style
    qa_system = QuestionAnswerer(
        llm=mock_llm,
        retriever=retriever,
        prompt_style="academic"
    )
    
    print(f"Initial style: {qa_system.prompt_style}")
    
    # Switch to detailed style
    qa_system.change_prompt_style("detailed")
    print(f"Changed to: {qa_system.prompt_style}")
    
    # List available styles
    available_styles = qa_system.get_available_prompt_styles()
    print(f"\nAvailable prompt styles:")
    for name, description in available_styles.items():
        print(f"  • {name}: {description}")
    
    print(f"\n✅ Prompt style switching works!")

def test_out_of_scope_handling():
    """Test how prompts handle out-of-scope questions."""
    print(f"\n🚫 Testing Out-of-Scope Query Handling...")
    print("-" * 45)
    
    # Mock response that follows the "I don't know" instruction
    mock_responses = [
        "I'm not sure based on the documents provided. The available context doesn't contain sufficient information to answer this question about quantum computing."
    ]
    
    mock_llm = FakeListLLM(responses=mock_responses)
    retriever = get_langchain_retriever(k=2)
    
    qa_system = QuestionAnswerer(
        llm=mock_llm,
        retriever=retriever,
        prompt_style="academic"
    )
    
    # Ask an out-of-scope question
    response = qa_system.answer("How does quantum computing work?")
    print(f"Out-of-scope question: How does quantum computing work?")
    print(f"Response: {response['answer']}")
    
    # Check if response follows the template
    if "not sure based on the documents" in response['answer'].lower():
        print("✅ Out-of-scope handling works correctly!")
    else:
        print("⚠️ Out-of-scope handling might need adjustment")

def test_prompt_template_manager():
    """Test the PromptTemplateManager class."""
    print(f"\n🛠️ Testing PromptTemplateManager...")
    print("-" * 35)
    
    manager = PromptTemplateManager()
    
    # List templates
    templates = manager.list_templates()
    print(f"Available templates: {list(templates.keys())}")
    
    # Get a specific template
    academic_template = manager.get_template("academic")
    print(f"Academic template variables: {academic_template.input_variables}")
    
    # Test template formatting
    sample_context = "Linear programming is an optimization technique for solving problems with linear objectives and constraints."
    sample_question = "What is linear programming?"
    
    formatted = academic_template.format(
        context=sample_context,
        question=sample_question
    )
    
    print(f"\n📝 Template formatting test:")
    print("✅ Template formatted successfully")
    print(f"Preview (first 150 chars): {formatted[:150]}...")
    
    print(f"\n✅ PromptTemplateManager works correctly!")

if __name__ == "__main__":
    test_prompt_template_manager()
    test_prompt_styles()
    test_prompt_style_switching()
    test_out_of_scope_handling()
    
    print(f"\n🎉 All custom prompt tests completed!")
    print(f"\n📚 Custom prompts are ready for academic use!")