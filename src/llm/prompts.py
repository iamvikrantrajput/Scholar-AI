"""
Custom Prompt Templates for RAG System

This module contains well-designed prompt templates for the AI Research Assistant
to ensure academic, grounded, and well-cited responses.
"""

from langchain_core.prompts import PromptTemplate
from typing import Dict, Any


# Academic Research Assistant Prompt Template
ACADEMIC_QA_TEMPLATE = """
You are an AI research assistant that helps students and researchers understand academic content. Your role is to provide clear, accurate, and well-cited answers based strictly on the provided course documents.

INSTRUCTIONS:
- Use ONLY the provided context to answer questions
- Maintain an academic and professional tone
- Be concise but thorough in your explanations
- Always cite your sources using the format: (source: filename, page X)
- If the context doesn't contain enough information to answer the question, respond with: "I'm not sure based on the documents provided. The available context doesn't contain sufficient information to answer this question."
- Do not make assumptions or add information not present in the context
- When explaining concepts, break them down clearly for better understanding

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# Detailed Explanation Prompt Template
DETAILED_EXPLANATION_TEMPLATE = """
You are an expert AI tutor specializing in academic research assistance. Your goal is to provide comprehensive explanations that help students deeply understand complex topics.

GUIDELINES:
- Base your response entirely on the provided context documents
- Provide detailed explanations with examples when available in the context
- Structure your answer with clear headings or bullet points when appropriate
- Include relevant citations for each major point: (source: filename, page X)
- If the question requires information not in the context, clearly state: "Based on the provided documents, I cannot fully address this question. Additional information would be needed."
- Connect related concepts when they appear in the context

CONTEXT DOCUMENTS:
{context}

STUDENT QUESTION: {question}

DETAILED EXPLANATION:"""


# Concise Summary Prompt Template
CONCISE_SUMMARY_TEMPLATE = """
You are a research assistant focused on providing clear, concise answers. Your responses should be direct and to the point while maintaining academic accuracy.

RULES:
- Answer using only information from the provided context
- Keep responses brief but complete
- Include essential citations: (source: filename, page X)
- If context is insufficient, respond: "Insufficient information in the provided documents."
- Focus on the most important points relevant to the question

CONTEXT:
{context}

QUESTION: {question}

CONCISE ANSWER:"""


# Comparative Analysis Prompt Template
COMPARATIVE_ANALYSIS_TEMPLATE = """
You are an AI research assistant specializing in comparative analysis. Help users understand similarities, differences, and relationships between concepts found in the academic documents.

APPROACH:
- Analyze only the information present in the provided context
- Compare and contrast concepts, methods, or theories as found in the documents
- Highlight key similarities and differences
- Provide specific citations for each comparison point: (source: filename, page X)
- If insufficient information exists for comparison, state: "The provided documents don't contain enough information for a comprehensive comparison."
- Organize comparisons clearly with structured formatting

CONTEXT MATERIALS:
{context}

COMPARISON QUESTION: {question}

COMPARATIVE ANALYSIS:"""


# Problem-Solving Prompt Template
PROBLEM_SOLVING_TEMPLATE = """
You are an AI teaching assistant helping students solve academic problems step-by-step. Guide them through solutions using only the methods and information available in the course materials.

METHODOLOGY:
- Use only techniques and information from the provided context
- Break down solutions into clear, logical steps
- Explain the reasoning behind each step
- Reference specific methods or formulas with citations: (source: filename, page X)
- If the problem requires methods not covered in the context, explain: "This problem requires methods not covered in the available documents."
- Encourage understanding rather than just providing answers

COURSE MATERIALS:
{context}

PROBLEM: {question}

STEP-BY-STEP SOLUTION:"""


class PromptTemplateManager:
    """Manager class for handling different prompt templates."""
    
    def __init__(self):
        self.templates = {
            "academic": self._create_academic_template(),
            "detailed": self._create_detailed_template(),
            "concise": self._create_concise_template(),
            "comparative": self._create_comparative_template(),
            "problem_solving": self._create_problem_solving_template()
        }
        self.default_template = "academic"

    def _create_academic_template(self) -> PromptTemplate:
        """Create the standard academic QA prompt template."""
        return PromptTemplate(
            template=ACADEMIC_QA_TEMPLATE,
            input_variables=["context", "question"]
        )

    def _create_detailed_template(self) -> PromptTemplate:
        """Create the detailed explanation prompt template."""
        return PromptTemplate(
            template=DETAILED_EXPLANATION_TEMPLATE,
            input_variables=["context", "question"]
        )
    
    def _create_concise_template(self) -> PromptTemplate:
        """Create the concise summary prompt template."""
        return PromptTemplate(
            template=CONCISE_SUMMARY_TEMPLATE,
            input_variables=["context", "question"]
        )
    
    def _create_comparative_template(self) -> PromptTemplate:
        """Create the comparative analysis prompt template."""
        return PromptTemplate(
            template=COMPARATIVE_ANALYSIS_TEMPLATE,
            input_variables=["context", "question"]
        )
    
    def _create_problem_solving_template(self) -> PromptTemplate:
        """Create the problem-solving prompt template."""
        return PromptTemplate(
            template=PROBLEM_SOLVING_TEMPLATE,
            input_variables=["context", "question"]
        )
    
    def get_template(self, template_name: str = None) -> PromptTemplate:
        """
        Get a specific prompt template.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            PromptTemplate instance
        """
        if template_name is None:
            template_name = self.default_template
        
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        
        return self.templates[template_name]
    
    def list_templates(self) -> Dict[str, str]:
        """
        List all available templates with descriptions.
        
        Returns:
            Dictionary mapping template names to descriptions
        """
        return {
            "academic": "Standard academic QA with citations and bounded responses",
            "detailed": "Comprehensive explanations with examples and structure",
            "concise": "Brief, direct answers focusing on key points",
            "comparative": "Comparative analysis between concepts or methods",
            "problem_solving": "Step-by-step problem solving guidance"
        }
    
    def create_custom_template(self, template_string: str, name: str) -> PromptTemplate:
        """
        Create and register a custom template.
        
        Args:
            template_string: The prompt template string
            name: Name to register the template under
            
        Returns:
            PromptTemplate instance
        """
        template = PromptTemplate(
            template=template_string,
            input_variables=["context", "question"]
        )
        self.templates[name] = template
        return template


# Factory functions for easy access
def get_academic_prompt() -> PromptTemplate:
    """Get the standard academic QA prompt template."""
    return PromptTemplate(
        template=ACADEMIC_QA_TEMPLATE,
        input_variables=["context", "question"]
    )


def get_detailed_prompt() -> PromptTemplate:
    """Get the detailed explanation prompt template."""
    return PromptTemplate(
        template=DETAILED_EXPLANATION_TEMPLATE,
        input_variables=["context", "question"]
    )


def get_concise_prompt() -> PromptTemplate:
    """Get the concise summary prompt template."""
    return PromptTemplate(
        template=CONCISE_SUMMARY_TEMPLATE,
        input_variables=["context", "question"]
    )


def get_comparative_prompt() -> PromptTemplate:
    """Get the comparative analysis prompt template."""
    return PromptTemplate(
        template=COMPARATIVE_ANALYSIS_TEMPLATE,
        input_variables=["context", "question"]
    )


def get_problem_solving_prompt() -> PromptTemplate:
    """Get the problem-solving prompt template."""
    return PromptTemplate(
        template=PROBLEM_SOLVING_TEMPLATE,
        input_variables=["context", "question"]
    )


# Default prompt template for backward compatibility
DEFAULT_PROMPT = get_academic_prompt()


# Example usage and testing
if __name__ == "__main__":
    # Test prompt template creation
    print("🧪 Testing Prompt Templates...")
    
    manager = PromptTemplateManager()
    
    print("\n📋 Available Templates:")
    for name, description in manager.list_templates().items():
        print(f"  • {name}: {description}")
    
    # Test template retrieval
    print(f"\n🔧 Testing template retrieval...")
    academic_template = manager.get_template("academic")
    print(f"Academic template variables: {academic_template.input_variables}")
    
    # Test template formatting
    print(f"\n📝 Testing template formatting...")
    sample_context = "Linear programming is an optimization technique..."
    sample_question = "What is linear programming?"
    
    formatted = academic_template.format(
        context=sample_context,
        question=sample_question
    )
    
    print("✅ Formatted prompt preview:")
    print("-" * 60)
    print(formatted[:1000] + "...")
    print("-" * 60)
    
    print("\n✅ Prompt templates ready for use!")