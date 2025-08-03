"""
Question Answering Chain Module

This module implements the RetrievalQA chain for the AI Research Assistant,
combining retrievers with language models for grounded question answering.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

# LangChain imports
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline

# Local imports
import sys
sys.path.append('..')
try:
    from retrieval.langchain_retriever import create_retriever_from_existing_store
except ImportError:
    # Fallback for when running from different directories
    from src.retrieval.langchain_retriever import create_retriever_from_existing_store
from .prompts import PromptTemplateManager, get_academic_prompt
from dotenv import load_dotenv

# Load environment variables
from pathlib import Path
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)  # override=True ensures .env values take precedence

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_api_key(key: str) -> str:
    """Clean API key by removing quotes and whitespace."""
    if not key:
        return key
    return key.strip().strip('"').strip("'")


class LLMProvider:
    """Factory class for creating different LLM providers."""
    
    @staticmethod
    def create_openai_llm(model: str = "gpt-3.5-turbo", 
                         temperature: float = 0.1,
                         api_key: Optional[str] = None) -> BaseLanguageModel:
        """Create OpenAI LLM."""
        try:
            # Clean the API key
            cleaned_key = clean_api_key(api_key or os.getenv("OPENAI_API_KEY"))
            
            if model.startswith("gpt-"):
                # Use ChatOpenAI for chat models
                return ChatOpenAI(
                    model_name=model,
                    temperature=temperature,
                    openai_api_key=cleaned_key,
                    max_tokens=1000
                )
            else:
                # Use OpenAI for completion models
                return OpenAI(
                    model_name=model,
                    temperature=temperature,
                    openai_api_key=cleaned_key,
                    max_tokens=1000
                )
        except Exception as e:
            logger.error(f"Failed to create OpenAI LLM: {e}")
            raise
    
    @staticmethod
    def create_huggingface_llm(model: str = "microsoft/DialoGPT-medium",
                              temperature: float = 0.1,
                              max_length: int = 1000,
                              api_key: Optional[str] = None) -> BaseLanguageModel:
        """Create HuggingFace LLM."""
        try:
            # Try HuggingFace Endpoint first (for hosted models)
            cleaned_hf_key = clean_api_key(api_key or os.getenv("HUGGINGFACE_API_KEY"))
            if cleaned_hf_key:
                try:
                    # Use newer HuggingFace parameters - try different parameter combinations
                    return HuggingFaceEndpoint(
                        repo_id=model,
                        temperature=temperature,
                        max_new_tokens=max_length,
                        huggingfacehub_api_token=cleaned_hf_key
                    )
                except Exception as api_error:
                    logger.warning(f"HuggingFace API failed with max_new_tokens: {api_error}")
                    try:
                        # Fallback without max_new_tokens
                        return HuggingFaceEndpoint(
                            repo_id=model,
                            temperature=temperature,
                            huggingfacehub_api_token=cleaned_hf_key
                        )
                    except Exception as api_error2:
                        logger.warning(f"HuggingFace API failed completely: {api_error2}")
                        # Don't raise the error, fall through to demo mode
                        logger.info("Falling back to demo mode due to HuggingFace API issues")
            else:
                # Use simple fake LLM for demo when no API key
                logger.warning("No HuggingFace API key found, using demo mode")
                from langchain_community.llms import FakeListLLM
                
                demo_responses = [
                    "Linear programming is a mathematical optimization technique used to solve problems with linear constraints. It finds the best solution by maximizing or minimizing an objective function subject to linear constraints.",
                    "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It uses algorithms to identify patterns and make predictions.",
                    "Python is a versatile programming language widely used in data science, web development, and automation due to its simplicity and extensive libraries.",
                    "Based on the available information, I can provide a general answer. However, for more detailed responses, please configure a proper language model."
                ]
                
                return FakeListLLM(responses=demo_responses)
        except Exception as e:
            logger.error(f"Failed to create HuggingFace LLM: {e}")
            raise
    
    @classmethod
    def create_llm(cls, provider: str = "openai", **kwargs) -> BaseLanguageModel:
        """Create LLM based on provider."""
        if provider.lower() == "openai":
            return cls.create_openai_llm(**kwargs)
        elif provider.lower() == "huggingface":
            return cls.create_huggingface_llm(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")


class QuestionAnswerer:
    """Main class for handling question answering with retrieval."""
    
    def __init__(self, 
                 llm: BaseLanguageModel,
                 retriever: BaseRetriever,
                 return_source_documents: bool = True,
                 chain_type: str = "stuff",
                 prompt_template: Optional[PromptTemplate] = None,
                 prompt_style: str = "academic"):
        """
        Initialize QuestionAnswerer.
        
        Args:
            llm: Language model
            retriever: Document retriever
            return_source_documents: Whether to return source documents
            chain_type: Type of QA chain ('stuff', 'map_reduce', 'refine', 'map_rerank')
            prompt_template: Custom prompt template (overrides prompt_style)
            prompt_style: Style of prompt to use ('academic', 'detailed', 'concise', etc.)
        """
        self.llm = llm
        self.retriever = retriever
        self.return_source_documents = return_source_documents
        self.chain_type = chain_type
        self.prompt_style = prompt_style
        
        # Set up prompt template
        if prompt_template is not None:
            self.prompt_template = prompt_template
            logger.info("Using custom prompt template")
        else:
            # Use prompt manager to get styled prompt
            prompt_manager = PromptTemplateManager()
            self.prompt_template = prompt_manager.get_template(prompt_style)
            logger.info(f"Using '{prompt_style}' prompt style")
        
        # Create the RetrievalQA chain with custom prompt
        chain_kwargs = {}
        if self.chain_type == "stuff":
            chain_kwargs["prompt"] = self.prompt_template
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.chain_type,
            retriever=self.retriever,
            return_source_documents=self.return_source_documents,
            chain_type_kwargs=chain_kwargs,
            verbose=True
        )
        
        logger.info(f"QuestionAnswerer initialized with {type(llm).__name__} and {type(retriever).__name__}")
    
    def answer(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the RetrievalQA chain.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer and sources
        """
        logger.info(f"Answering question: {question}")
        
        try:
            # Run the QA chain
            result = self.qa_chain.invoke({"query": question})
            
            # Format the response
            response = {
                "question": question,
                "answer": result.get("result", ""),
                "timestamp": datetime.now().isoformat(),
                "sources": []
            }
            
            # Add source documents if available
            if self.return_source_documents and "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", "unknown"),
                        "word_count": doc.metadata.get("word_count", "unknown")
                    }
                    response["sources"].append(source_info)
            
            logger.info(f"Generated answer with {len(response['sources'])} sources")
            return response
            
        except StopIteration as e:
            # Handle StopIteration specifically (common with HuggingFace models)
            error_msg = "The language model stopped generating text unexpectedly. This might be due to model limitations or API issues. Try using OpenAI provider or a different question."
            logger.error(f"StopIteration error answering question: {error_msg}")
            
            return {
                "question": question,
                "answer": f"I apologize, but I encountered an error while processing your question: {error_msg}",
                "timestamp": datetime.now().isoformat(),
                "sources": [],
                "error": error_msg
            }
        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error of type {type(e).__name__}"
            logger.error(f"Error answering question: {error_msg}")
            
            # Also log the full traceback for debugging
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            return {
                "question": question,
                "answer": f"I apologize, but I encountered an error while processing your question: {error_msg}",
                "timestamp": datetime.now().isoformat(),
                "sources": [],
                "error": error_msg
            }
    
    def answer_with_custom_prompt(self, question: str, prompt_template: str) -> Dict[str, Any]:
        """
        Answer a question with a custom prompt template.
        
        Args:
            question: User's question
            prompt_template: Custom prompt template
            
        Returns:
            Dictionary containing answer and sources
        """
        # Create custom QA chain with prompt
        custom_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        custom_qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.chain_type,
            retriever=self.retriever,
            return_source_documents=self.return_source_documents,
            chain_type_kwargs={"prompt": custom_prompt}
        )
        
        # Temporarily replace the chain
        original_chain = self.qa_chain
        self.qa_chain = custom_qa_chain
        
        try:
            result = self.answer(question)
            return result
        finally:
            # Restore original chain
            self.qa_chain = original_chain
    
    def change_prompt_style(self, new_style: str):
        """
        Change the prompt style and rebuild the QA chain.
        
        Args:
            new_style: New prompt style ('academic', 'detailed', 'concise', etc.)
        """
        logger.info(f"Changing prompt style from '{self.prompt_style}' to '{new_style}'")
        
        prompt_manager = PromptTemplateManager()
        self.prompt_template = prompt_manager.get_template(new_style)
        self.prompt_style = new_style
        
        # Rebuild the QA chain with new prompt
        chain_kwargs = {}
        if self.chain_type == "stuff":
            chain_kwargs["prompt"] = self.prompt_template
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=self.chain_type,
            retriever=self.retriever,
            return_source_documents=self.return_source_documents,
            chain_type_kwargs=chain_kwargs,
            verbose=True
        )
        
        logger.info(f"QA chain rebuilt with '{new_style}' prompt style")
    
    def get_available_prompt_styles(self) -> Dict[str, str]:
        """
        Get available prompt styles and their descriptions.
        
        Returns:
            Dictionary mapping style names to descriptions
        """
        prompt_manager = PromptTemplateManager()
        return prompt_manager.list_templates()


def create_qa_chain(llm_provider: str = "openai",
                   llm_model: str = None,
                   retriever_path: str = "data/vectorstore/langchain_faiss",
                   temperature: float = 0.1,
                   prompt_style: str = "academic",
                   **kwargs) -> QuestionAnswerer:
    """
    Create a complete QA chain with LLM and retriever.
    
    Args:
        llm_provider: LLM provider ('openai' or 'huggingface')
        llm_model: Specific model name
        retriever_path: Path to the vector store
        temperature: LLM temperature
        prompt_style: Style of prompt to use ('academic', 'detailed', 'concise', etc.)
        **kwargs: Additional arguments for LLM creation
        
    Returns:
        QuestionAnswerer instance
    """
    # Set default models
    if llm_model is None:
        if llm_provider == "openai":
            llm_model = "gpt-3.5-turbo"
        elif llm_provider == "huggingface":
            llm_model = "microsoft/DialoGPT-medium"
    
    # Create LLM
    logger.info(f"Creating {llm_provider} LLM with model {llm_model}")
    
    # Filter kwargs for LLM creation
    llm_kwargs = {k: v for k, v in kwargs.items() 
                  if k in ['api_key', 'max_length', 'max_tokens']}
    
    llm = LLMProvider.create_llm(
        provider=llm_provider,
        model=llm_model,
        temperature=temperature,
        **llm_kwargs
    )
    
    # Create retriever
    logger.info(f"Loading retriever from {retriever_path}")
    
    # Check if retriever path exists, if not create a demo store
    retriever_path_obj = Path(retriever_path)
    if not retriever_path_obj.exists():
        logger.warning(f"Retriever path {retriever_path} does not exist. Creating demo vector store...")
        
        # Create a demo vector store with sample documents
        from retrieval.langchain_retriever import LangChainEmbeddingAdapter, create_faiss_store
        from langchain_core.documents import Document
        
        # Sample documents for demo
        demo_docs = [
            Document(
                page_content="Linear programming is a mathematical optimization technique used to find the best solution to problems with linear constraints. It involves maximizing or minimizing a linear objective function subject to linear equality and inequality constraints. Linear programming is widely used in operations research, economics, and engineering.",
                metadata={"source": "demo_math.pdf", "page": 1}
            ),
            Document(
                page_content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions. Common applications include image recognition, natural language processing, and predictive analytics.",
                metadata={"source": "demo_ai.pdf", "page": 1}
            ),
            Document(
                page_content="Python is a high-level, interpreted programming language known for its simplicity and readability. It is widely used in data science, web development, automation, and scientific computing. Python's extensive library ecosystem makes it popular for machine learning, data analysis, and rapid prototyping.",
                metadata={"source": "demo_programming.pdf", "page": 1}
            )
        ]
        
        # Create embeddings and vector store
        embedding_adapter = LangChainEmbeddingAdapter(
            provider="huggingface",
            model="all-MiniLM-L6-v2"
        )
        embeddings = embedding_adapter.get_embeddings()
        
        # Create demo vector store
        retriever_path_obj.parent.mkdir(parents=True, exist_ok=True)
        vector_store = create_faiss_store(demo_docs, embeddings, retriever_path_obj)
        
        logger.info(f"Created demo vector store at {retriever_path_obj}")
    
    embedding_config = {
        "provider": "huggingface" if llm_provider == "huggingface" else "huggingface",  # Default to HF for consistency
        "model": "all-MiniLM-L6-v2"
    }
    
    retriever_obj = create_retriever_from_existing_store(
        store_path=retriever_path_obj,
        embedding_config=embedding_config,
        store_type="faiss",
        search_kwargs={"k": kwargs.get('retriever_k', 3)}
    )
    
    retriever = retriever_obj.get_retriever()
    
    # Create QA chain
    qa_chain = QuestionAnswerer(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type=kwargs.get('chain_type', 'stuff'),
        prompt_style=prompt_style
    )
    
    return qa_chain


# Global QA chain instance (lazy loading)
_global_qa_chain = None


def get_default_qa_chain() -> QuestionAnswerer:
    """Get or create the default QA chain."""
    global _global_qa_chain
    
    if _global_qa_chain is None:
        logger.info("Initializing default QA chain")
        try:
            # Try OpenAI first
            _global_qa_chain = create_qa_chain(
                llm_provider="openai",
                llm_model="gpt-3.5-turbo",
                temperature=0.1
            )
        except Exception as e:
            logger.warning(f"Failed to create OpenAI QA chain: {e}")
            logger.info("Falling back to HuggingFace")
            try:
                _global_qa_chain = create_qa_chain(
                    llm_provider="huggingface",
                    llm_model="microsoft/DialoGPT-medium",
                    temperature=0.1
                )
            except Exception as e:
                logger.error(f"Failed to create any QA chain: {e}")
                raise
    
    return _global_qa_chain


def answer_question(question: str, 
                   qa_chain: Optional[QuestionAnswerer] = None) -> Dict[str, Any]:
    """
    Answer a question using the QA chain.
    
    Args:
        question: User's question
        qa_chain: Optional QA chain instance (uses default if None)
        
    Returns:
        Dictionary containing answer and sources
    """
    if qa_chain is None:
        qa_chain = get_default_qa_chain()
    
    return qa_chain.answer(question)


def format_qa_response(response: Dict[str, Any], 
                      show_sources: bool = True,
                      max_source_text: int = 200) -> str:
    """
    Format QA response for display.
    
    Args:
        response: QA response dictionary
        show_sources: Whether to show source information
        max_source_text: Maximum length of source text to display
        
    Returns:
        Formatted string
    """
    formatted = []
    
    # Add question and answer
    formatted.append(f"❓ Question: {response['question']}")
    formatted.append(f"💡 Answer: {response['answer']}")
    
    # Add sources if requested and available
    if show_sources and response.get('sources'):
        formatted.append(f"\n📚 Sources ({len(response['sources'])} documents):")
        
        for i, source in enumerate(response['sources'], 1):
            source_text = source['text'][:max_source_text]
            if len(source['text']) > max_source_text:
                source_text += "..."
            
            formatted.append(
                f"\n{i}. 📄 {source['source']} (Page {source['page']})\n"
                f"   📝 {source_text}"
            )
    
    # Add timestamp
    if 'timestamp' in response:
        formatted.append(f"\n🕐 Answered at: {response['timestamp']}")
    
    return "\n".join(formatted)


# Example usage and testing
if __name__ == "__main__":
    # Test the QA chain
    print("🧪 Testing QA Chain...")
    
    try:
        # Create QA chain
        qa_chain = create_qa_chain(
            llm_provider="openai",  # Try OpenAI first
            temperature=0.1
        )
        
        # Test questions
        test_questions = [
            "What is linear programming?",
            "How do you solve optimization problems?",
            "Explain the maximum flow problem."
        ]
        
        for question in test_questions:
            print(f"\n🔍 Testing: {question}")
            response = qa_chain.answer(question)
            print(format_qa_response(response, max_source_text=100))
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Error testing QA chain: {e}")
        import traceback
        traceback.print_exc()