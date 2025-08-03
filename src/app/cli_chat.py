#!/usr/bin/env python3
"""
CLI Chat Interface for ScholarAI

This script provides a command-line interface for testing the RetrievalQA system.
"""

# Load environment variables first
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import argparse
import logging
from typing import Optional

from llm.qa_chain import create_qa_chain, answer_question, format_qa_response
from ingestion.pdf_parser import PDFProcessor
from retrieval.langchain_retriever import create_retriever_from_json


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def process_raw_documents(verbose: bool = False) -> bool:
    """
    Process all PDFs in data/raw directory and create/update vector store.
    Always refreshes processed data and vector store to match current raw files.
    
    Args:
        verbose: Enable verbose logging
        
    Returns:
        True if documents were processed successfully, False otherwise
    """
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed") 
    vectorstore_dir = Path("data/vectorstore/langchain_faiss")
    
    if not raw_dir.exists():
        if verbose:
            print("📂 No data/raw directory found, skipping document processing")
        return True
    
    # Find PDF files
    pdf_files = list(raw_dir.glob("*.pdf"))
    if not pdf_files:
        if verbose:
            print("📄 No PDF files found in data/raw directory")
        # Clean up existing processed/vector data if no PDFs
        if processed_dir.exists():
            import shutil
            shutil.rmtree(processed_dir)
            print("🧹 Cleaned up processed directory (no PDFs in raw)")
        if vectorstore_dir.exists():
            import shutil
            shutil.rmtree(vectorstore_dir)
            print("🧹 Cleaned up vector store directory (no PDFs in raw)")
        return True
    
    print(f"📚 Found {len(pdf_files)} PDF file(s) in data/raw directory")
    
    # Always clean and recreate directories for fresh processing
    print("🧹 Cleaning previous processed data and vector store...")
    if processed_dir.exists():
        import shutil
        shutil.rmtree(processed_dir)
    if vectorstore_dir.exists():
        import shutil  
        shutil.rmtree(vectorstore_dir)
    
    # Create fresh directories
    processed_dir.mkdir(parents=True, exist_ok=True)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        
        # Process PDFs
        print("🔄 Processing PDF documents...")
        processor = PDFProcessor()
        all_chunks = []
        
        for pdf_file in pdf_files:
            print(f"   📖 Processing {pdf_file.name}...")
            try:
                chunks = processor.process_pdf(pdf_file)
                all_chunks.extend(chunks)
                
                # Save processed chunks
                processed_file = processed_dir / f"{pdf_file.stem}.json"
                import json
                with open(processed_file, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, ensure_ascii=False, indent=2)
                
                print(f"      ✅ Created {len(chunks)} chunks")
                
            except Exception as e:
                print(f"      ❌ Error processing {pdf_file.name}: {e}")
                if verbose:
                    import traceback
                    print(f"         {traceback.format_exc()}")
                continue
        
        if not all_chunks:
            print("❌ No chunks were created from PDF processing")
            return False
        
        print(f"📊 Total chunks created: {len(all_chunks)}")
        
        # Create vector store using LangChain format
        print("🧠 Creating vector store...")
        from retrieval.langchain_retriever import create_retriever_from_json
        
        # Save chunks as JSON files first
        processed_files = []
        for pdf_file in pdf_files:
            processed_file = processed_dir / f"{pdf_file.stem}.json"
            if processed_file.exists():
                processed_files.append(processed_file)
        
        if processed_files:
            # Create retriever using LangChain format
            embedding_config = {
                "provider": "huggingface",
                "model": "all-MiniLM-L6-v2"
            }
            
            retriever = create_retriever_from_json(
                processed_dir=processed_dir,
                embedding_config=embedding_config,
                store_type="faiss",
                save_path=vectorstore_dir,
                search_kwargs={"k": 3}
            )
        
        print("✅ Vector store created successfully!")
        print(f"📁 Ready to answer questions about {len(pdf_files)} document(s)")
        return True
        
    except Exception as e:
        print(f"❌ Error during document processing: {e}")
        if verbose:
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
        return False


def single_question_mode(question: str, 
                        llm_provider: str = "openai",
                        llm_model: Optional[str] = None,
                        temperature: float = 0.1,
                        retriever_k: int = 3,
                        show_sources: bool = True,
                        prompt_style: str = "academic",
                        verbose: bool = False) -> None:
    """
    Answer a single question and exit.
    
    Args:
        question: The question to answer
        llm_provider: LLM provider to use
        llm_model: Specific model name
        temperature: LLM temperature
        retriever_k: Number of documents to retrieve
        show_sources: Whether to show source documents
        prompt_style: Style of prompt to use
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    
    print("🎓 ScholarAI - Single Question Mode")
    print("=" * 60)
    
    # First, process any documents in data/raw
    print("📂 Checking for documents to process...")
    if not process_raw_documents(verbose):
        print("❌ Failed to process documents. Exiting.")
        return
    
    print()  # Add some spacing
    
    try:
        # Create QA chain
        print(f"🚀 Initializing {llm_provider} QA chain...")
        qa_chain = create_qa_chain(
            llm_provider=llm_provider,
            llm_model=llm_model,
            temperature=temperature,
            retriever_k=retriever_k,
            prompt_style=prompt_style
        )
        
        print("✅ QA chain initialized successfully")
        print(f"🔍 Processing question: '{question}'")
        print("-" * 60)
        
        # Answer the question
        response = qa_chain.answer(question)
        
        # Format and display response
        formatted_response = format_qa_response(
            response, 
            show_sources=show_sources,
            max_source_text=300
        )
        
        print(formatted_response)
        
        # Show retrieval statistics
        if 'sources' in response:
            print(f"\n📊 Retrieved {len(response['sources'])} source documents")
            
        # Check for errors
        if 'error' in response:
            print(f"\n⚠️ Error Details: {response['error']}")
            
            # Also show debugging info if available
            if verbose:
                print(f"\n🔍 Debug Info:")
                print(f"  - Question: {response.get('question', 'N/A')}")
                print(f"  - Timestamp: {response.get('timestamp', 'N/A')}")
                print(f"  - Sources found: {len(response.get('sources', []))}")
                if response.get('sources'):
                    for i, source in enumerate(response['sources'], 1):
                        print(f"    {i}. {source.get('source', 'Unknown')} (Page {source.get('page', 'N/A')})")
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def interactive_chat_mode(llm_provider: str = "openai",
                         llm_model: Optional[str] = None,
                         temperature: float = 0.1,
                         retriever_k: int = 3,
                         show_sources: bool = True,
                         prompt_style: str = "academic",
                         verbose: bool = False) -> None:
    """
    Start an interactive chat session.
    
    Args:
        llm_provider: LLM provider to use
        llm_model: Specific model name
        temperature: LLM temperature
        retriever_k: Number of documents to retrieve
        show_sources: Whether to show source documents
        prompt_style: Style of prompt to use
        verbose: Enable verbose logging
    """
    setup_logging(verbose)
    
    print("🎓 ScholarAI - Interactive Chat Mode")
    print("=" * 60)
    
    # First, process any documents in data/raw
    print("📂 Checking for documents to process...")
    if not process_raw_documents(verbose):
        print("❌ Failed to process documents. Exiting.")
        return
    
    print()
    print("Ask questions about your documents. Type 'quit' or 'exit' to stop.")
    print("Type 'help' for available commands.\n")
    
    try:
        # Create QA chain
        print(f"🚀 Initializing {llm_provider} QA chain...")
        qa_chain = create_qa_chain(
            llm_provider=llm_provider,
            llm_model=llm_model,
            temperature=temperature,
            retriever_k=retriever_k,
            prompt_style=prompt_style
        )
        
        print("✅ QA chain initialized successfully")
        print("💬 Ready for questions!\n")
        
        session_count = 0
        
        while True:
            try:
                # Get user input
                question = input("❓ You: ").strip()
                
                # Handle special commands
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'help':
                    show_help()
                    continue
                elif question.lower() == 'stats':
                    show_session_stats(session_count)
                    continue
                elif question.lower().startswith('sources'):
                    show_sources = not show_sources
                    print(f"📚 Source display: {'ON' if show_sources else 'OFF'}")
                    continue
                elif not question:
                    continue
                
                # Answer the question
                print(f"\n🤔 Thinking...")
                response = qa_chain.answer(question)
                
                # Format and display response
                formatted_response = format_qa_response(
                    response,
                    show_sources=show_sources,
                    max_source_text=200
                )
                
                print(f"\n{formatted_response}")
                print("-" * 60)
                
                session_count += 1
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error answering question: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
                print("-" * 60)
        
        print(f"\n👋 Chat session ended. Answered {session_count} questions.")
        
    except Exception as e:
        print(f"\n❌ Failed to initialize QA chain: {e}")
        if verbose:
            import traceback
            traceback.print_exc()


def show_help():
    """Show available commands."""
    print("\n📖 Available Commands:")
    print("  help       - Show this help message")
    print("  sources    - Toggle source document display")
    print("  stats      - Show session statistics")
    print("  quit/exit  - Exit the chat")
    print("\n💡 Tips:")
    print("  - Documents from data/raw/ are automatically processed")
    print("  - Ask specific questions about your documents")
    print("  - Try: 'What is the main topic of this paper?'")
    print("  - Try: 'Summarize the key findings'")
    print("  - Use natural language - the AI understands context")
    print()


def show_session_stats(question_count: int):
    """Show session statistics."""
    print(f"\n📊 Session Statistics:")
    print(f"  Questions answered: {question_count}")
    print(f"  Status: Active")
    print()


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="ScholarAI CLI Chat Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat mode
  python src/app/cli_chat.py
  
  # Answer a single question
  python src/app/cli_chat.py --question "What is linear programming?"
  
  # Use HuggingFace model
  python src/app/cli_chat.py --llm-provider huggingface --question "Explain optimization"
  
  # Verbose mode with custom settings
  python src/app/cli_chat.py --verbose --temperature 0.2 --retriever-k 5
        """
    )
    
    # Main arguments
    parser.add_argument("--question", "-q", type=str,
                       help="Single question to answer (non-interactive mode)")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Start interactive chat mode")
    
    # LLM configuration
    parser.add_argument("--llm-provider", choices=["openai", "huggingface"],
                       default="openai", help="LLM provider to use")
    parser.add_argument("--llm-model", type=str,
                       help="Specific model name (default: gpt-3.5-turbo for OpenAI)")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="LLM temperature (0.0-1.0)")
    
    # Retrieval configuration
    parser.add_argument("--retriever-k", type=int, default=3,
                       help="Number of documents to retrieve")
    parser.add_argument("--no-sources", action="store_true",
                       help="Don't show source documents")
    
    # Prompt configuration
    parser.add_argument("--prompt-style", 
                       choices=["academic", "detailed", "concise", "comparative", "problem_solving"],
                       default="academic",
                       help="Style of prompt to use for responses")
    
    # Other options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.question:
        # Single question mode
        single_question_mode(
            question=args.question,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            temperature=args.temperature,
            retriever_k=args.retriever_k,
            show_sources=not args.no_sources,
            prompt_style=args.prompt_style,
            verbose=args.verbose
        )
    else:
        # Interactive mode (default)
        interactive_chat_mode(
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            temperature=args.temperature,
            retriever_k=args.retriever_k,
            show_sources=not args.no_sources,
            prompt_style=args.prompt_style,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()