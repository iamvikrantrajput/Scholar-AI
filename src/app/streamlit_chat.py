#!/usr/bin/env python3
"""
Streamlit Chat UI for ScholarAI

A modern web interface for PDF upload and academic Q&A using RAG.
"""

import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# Configure Streamlit page
st.set_page_config(
    page_title="ScholarAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables first
from dotenv import load_dotenv
from pathlib import Path
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path, override=True)

# Import backend modules
import sys
sys.path.append('..')

try:
    from ingestion.pdf_parser import PDFProcessor
    from vectorstore.embeddings import EmbeddingManager
    from vectorstore.vector_store import VectorStoreManager
    from llm.qa_chain import QuestionAnswerer, create_qa_chain
    from llm.prompts import PromptTemplateManager
    from retrieval.langchain_retriever import DocumentRetriever
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


class ScholarAIApp:
    """Main ScholarAI Streamlit Application Class."""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'uploaded_files' not in st.session_state:
            st.session_state.uploaded_files = []
        
        if 'qa_chain' not in st.session_state:
            st.session_state.qa_chain = None
        
        if 'vector_store' not in st.session_state:
            st.session_state.vector_store = None
        
        if 'processed_docs' not in st.session_state:
            st.session_state.processed_docs = []
        
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())[:8]
    
    def render_header(self):
        """Render the main header."""
        st.title("🎓 ScholarAI")
        st.markdown("""
        **Upload your academic PDFs and ask questions!** 
        ScholarAI provides grounded answers with proper citations.
        """)
    
    def render_sidebar(self):
        """Render the sidebar with file upload and settings."""
        with st.sidebar:
            st.header("📁 Document Upload")
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload PDF documents",
                type=['pdf'],
                accept_multiple_files=True,
                help="Drag and drop PDF files here or click to browse"
            )
            
            # Process uploaded files
            if uploaded_files:
                self.handle_file_upload(uploaded_files)
            
            # Display uploaded files
            if st.session_state.uploaded_files:
                st.subheader("📚 Uploaded Documents")
                for i, file_info in enumerate(st.session_state.uploaded_files):
                    with st.expander(f"📄 {file_info['name']}", expanded=False):
                        st.write(f"**Size:** {file_info['size']:,} bytes")
                        st.write(f"**Uploaded:** {file_info['timestamp']}")
                        st.write(f"**Status:** {file_info['status']}")
                        if file_info['chunks']:
                            st.write(f"**Chunks:** {len(file_info['chunks'])}")
            
            st.divider()
            
            # Settings
            st.header("⚙️ Settings")
            
            # LLM Provider selection
            llm_provider = st.selectbox(
                "LLM Provider",
                ["huggingface", "openai"],
                index=0,
                help="Choose the language model provider"
            )
            
            # Prompt style selection
            prompt_manager = PromptTemplateManager()
            available_styles = list(prompt_manager.list_templates().keys())
            
            prompt_style = st.selectbox(
                "Response Style",
                available_styles,
                index=0,
                help="Choose the style of AI responses"
            )
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                retriever_k = st.slider(
                    "Documents to retrieve",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Number of document chunks to retrieve for each question"
                )
                
                temperature = st.slider(
                    "Response creativity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.1,
                    step=0.1,
                    help="Higher values make responses more creative but less focused"
                )
                
                show_sources = st.checkbox(
                    "Show source citations",
                    value=True,
                    help="Display source documents and page numbers"
                )
            
            # API Key inputs (if needed)
            st.divider()
            st.header("🔑 API Keys")
            
            # Show current status
            openai_configured = bool(os.getenv("OPENAI_API_KEY"))
            hf_configured = bool(os.getenv("HUGGINGFACE_API_KEY"))
            
            status_msg = "**Current Status:**\n"
            status_msg += f"- OpenAI: {'✅ Configured' if openai_configured else '❌ Not configured'}\n"
            status_msg += f"- HuggingFace: {'✅ Configured' if hf_configured else '❌ Not configured'}"
            
            if not openai_configured and not hf_configured:
                status_msg += "\n\n⚠️ **Note:** Without API keys, the system will use a basic demo mode with limited functionality."
            
            st.markdown(status_msg)
            
            if llm_provider == "openai":
                openai_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Enter your OpenAI API key for best results"
                )
                if openai_key:
                    os.environ["OPENAI_API_KEY"] = openai_key
                    st.success("✅ OpenAI API key configured!")
            
            huggingface_key = st.text_input(
                "HuggingFace API Key",
                type="password",
                help="Optional: Enter your HuggingFace API key for better models"
            )
            if huggingface_key:
                os.environ["HUGGINGFACE_API_KEY"] = huggingface_key
                st.success("✅ HuggingFace API key configured!")
            
            # Store settings in session state
            st.session_state.llm_provider = llm_provider
            st.session_state.prompt_style = prompt_style
            st.session_state.retriever_k = retriever_k
            st.session_state.temperature = temperature
            st.session_state.show_sources = show_sources
            
            # Clear chat button
            st.divider()
            if st.button("🗑️ Clear Chat", type="secondary"):
                st.session_state.messages = []
                st.rerun()
    
    def handle_file_upload(self, uploaded_files):
        """Handle PDF file uploads and processing."""
        for uploaded_file in uploaded_files:
            # Check if file already uploaded
            if any(f['name'] == uploaded_file.name for f in st.session_state.uploaded_files):
                continue
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Process PDF
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    processor = PDFProcessor()
                    chunks = processor.process_pdf(Path(tmp_path))
                
                # Add to session state
                file_info = {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'status': 'Processed',
                    'chunks': chunks,
                    'path': tmp_path
                }
                
                st.session_state.uploaded_files.append(file_info)
                st.session_state.processed_docs.extend(chunks)
                
                # Clear existing QA chain to rebuild with new documents
                st.session_state.qa_chain = None
                st.session_state.vector_store = None
                
                st.success(f"✅ Processed {uploaded_file.name} ({len(chunks)} chunks)")
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                file_info = {
                    'name': uploaded_file.name,
                    'size': uploaded_file.size,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'status': f'Error: {str(e)}',
                    'chunks': [],
                    'path': tmp_path
                }
                st.session_state.uploaded_files.append(file_info)
            
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
    
    def build_qa_chain(self):
        """Build or rebuild the QA chain with current documents."""
        if not st.session_state.processed_docs:
            return None
        
        if st.session_state.qa_chain is not None:
            return st.session_state.qa_chain
        
        try:
            with st.spinner("🔧 Building AI system..."):
                # Use the simpler create_qa_chain approach
                # Save documents to temporary JSON file for processing
                temp_dir = Path(f"temp_data_{st.session_state.session_id}")
                temp_dir.mkdir(exist_ok=True)
                
                # Save processed docs to JSON
                temp_json = temp_dir / "documents.json"
                with open(temp_json, "w", encoding="utf-8") as f:
                    import json
                    json.dump(st.session_state.processed_docs, f, ensure_ascii=False, indent=2)
                
                # Create vector store directory
                vector_dir = temp_dir / "vectorstore"
                vector_dir.mkdir(exist_ok=True)
                
                # Create vector store directly from documents using LangChain
                from retrieval.langchain_retriever import LangChainEmbeddingAdapter, create_faiss_store
                from langchain_core.documents import Document
                
                # Convert processed docs to LangChain Documents
                documents = []
                for chunk in st.session_state.processed_docs:
                    doc = Document(
                        page_content=chunk["text"],
                        metadata=chunk["metadata"]
                    )
                    documents.append(doc)
                
                # Create embeddings
                embedding_adapter = LangChainEmbeddingAdapter(
                    provider="huggingface",
                    model="all-MiniLM-L6-v2"
                )
                embeddings = embedding_adapter.get_embeddings()
                
                # Create FAISS vector store
                vector_store = create_faiss_store(documents, embeddings, vector_dir)
                
                # Create LLM based on user selection
                from llm.qa_chain import LLMProvider
                
                try:
                    llm_provider = st.session_state.get('llm_provider', 'huggingface')
                    
                    if llm_provider == "openai" and os.getenv("OPENAI_API_KEY"):
                        llm = LLMProvider.create_openai_llm(
                            model="gpt-3.5-turbo",
                            temperature=st.session_state.get('temperature', 0.1)
                        )
                    else:
                        # Use HuggingFace as fallback or default
                        try:
                            # Try to use HuggingFace Endpoint first
                            if os.getenv("HUGGINGFACE_API_KEY"):
                                llm = LLMProvider.create_huggingface_llm(
                                    model="microsoft/DialoGPT-medium",
                                    temperature=st.session_state.get('temperature', 0.1),
                                    max_length=512,
                                    api_key=os.getenv("HUGGINGFACE_API_KEY")
                                )
                            else:
                                # Use a simple text generation model locally
                                from langchain_huggingface import HuggingFacePipeline
                                from transformers import pipeline
                                
                                # Use a model that's better for QA
                                text_generator = pipeline(
                                    "text2text-generation",
                                    model="google/flan-t5-small",  # Better for QA tasks
                                    max_length=256,
                                    temperature=st.session_state.get('temperature', 0.1),
                                    do_sample=True
                                )
                                
                                llm = HuggingFacePipeline(pipeline=text_generator)
                            
                        except Exception as hf_error:
                            st.warning(f"Could not load HuggingFace model: {hf_error}")
                            # Fallback to a simple mock LLM with better responses
                            from langchain_community.llms import FakeListLLM
                            
                            mock_responses = [
                                "Based on the uploaded documents, I can provide relevant information. However, for more detailed analysis, please configure a proper LLM provider (OpenAI or HuggingFace) in the settings.",
                                "I found relevant content in your documents that relates to your question. The system is currently running in demonstration mode.",
                                "Your question can be answered using the document content you've provided. Please note this is a simplified response mode."
                            ]
                            
                            llm = FakeListLLM(responses=mock_responses)
                            
                except Exception as e:
                    st.error(f"Error creating LLM: {e}")
                    # Final fallback
                    from langchain_community.llms import FakeListLLM
                    llm = FakeListLLM(responses=["I encountered an error setting up the language model. Please check your configuration."])
                
                # Create retriever directly from vector store
                from retrieval.langchain_retriever import DocumentRetriever
                
                retriever = DocumentRetriever(
                    vector_store=vector_store,
                    search_kwargs={"k": st.session_state.get('retriever_k', 3)}
                )
                
                qa_chain = QuestionAnswerer(
                    llm=llm,
                    retriever=retriever.get_retriever(),
                    return_source_documents=True,
                    prompt_style=st.session_state.get('prompt_style', 'academic')
                )
                
                st.session_state.qa_chain = qa_chain
                st.session_state.temp_dir = temp_dir
                
                return qa_chain
                
        except Exception as e:
            st.error(f"❌ Error building QA system: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            return None
    
    def render_chat_interface(self):
        """Render the main chat interface."""
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        # Show answer
                        st.markdown(f"**💡 Answer:**")
                        st.markdown(message["content"])
                        
                        # Show sources inline if available
                        if st.session_state.get('show_sources', True) and message.get("sources"):
                            sources = message["sources"]
                            st.markdown("---")
                            st.markdown(f"**📚 Sources ({len(sources)} documents):**")
                            
                            for i, source in enumerate(sources, 1):
                                source_name = source.get('source', 'Unknown Document')
                                page_num = source.get('page', 'N/A')
                                text_preview = source.get('text', '')
                                
                                if len(text_preview) > 200:
                                    text_preview = text_preview[:200] + "..."
                                
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"**{i}. {source_name}**")
                                        st.caption(text_preview)
                                    with col2:
                                        st.markdown(f"**Page {page_num}**")
                                
                                if i < len(sources):
                                    st.markdown("")
                    else:
                        # User messages
                        st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            if not st.session_state.processed_docs:
                st.warning("⚠️ Please upload some PDF documents first!")
                return
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = self.get_ai_response(prompt)
                
                if response:
                    # Display the answer prominently
                    answer_text = response["answer"]
                    sources = response.get("sources", [])
                    
                    if answer_text:
                        # Show the main answer
                        st.markdown(f"**💡 Answer:**")
                        st.markdown(answer_text)
                        
                        # Show sources inline if available
                        if st.session_state.get('show_sources', True) and sources:
                            st.markdown("---")
                            st.markdown(f"**📚 Sources ({len(sources)} documents):**")
                            
                            for i, source in enumerate(sources, 1):
                                # Create a more compact source display
                                source_name = source.get('source', 'Unknown Document')
                                page_num = source.get('page', 'N/A')
                                text_preview = source.get('text', '')
                                
                                # Truncate long text for cleaner display
                                if len(text_preview) > 200:
                                    text_preview = text_preview[:200] + "..."
                                
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"**{i}. {source_name}**")
                                        st.caption(text_preview)
                                    with col2:
                                        st.markdown(f"**Page {page_num}**")
                                
                                if i < len(sources):  # Add separator except for last item
                                    st.markdown("")
                    else:
                        st.warning("❌ No answer was generated. Please try rephrasing your question.")
                        
                        # Show debug info if sources exist but no answer
                        if sources:
                            st.info("ℹ️ Found relevant documents but couldn't generate a synthesized answer.")
                            with st.expander("🔍 Retrieved document chunks"):
                                for i, source in enumerate(sources, 1):
                                    st.write(f"**Chunk {i} from {source.get('source', 'Unknown')}:**")
                                    st.code(source.get("text", "")[:300] + "..." if len(source.get("text", "")) > 300 else source.get("text", ""))
                    
                    # Add assistant message to session state (simplified)
                    assistant_message = {
                        "role": "assistant",
                        "content": answer_text if answer_text else "I couldn't generate an answer for that question.",
                        "sources": sources
                    }
                    st.session_state.messages.append(assistant_message)
                else:
                    error_msg = "❌ Sorry, I couldn't process your question. Please try again."
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    def get_ai_response(self, question: str) -> Optional[Dict[str, Any]]:
        """Get AI response for a question."""
        qa_chain = self.build_qa_chain()
        
        if not qa_chain:
            return None
        
        try:
            response = qa_chain.answer(question)
            return response
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return None
    
    
    def render_footer(self):
        """Render footer with app info."""
        st.divider()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Session ID:** " + st.session_state.session_id)
        
        with col2:
            st.markdown(f"**Documents:** {len(st.session_state.uploaded_files)}")
        
        with col3:
            st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    
    def run(self):
        """Run the main Streamlit app."""
        self.render_header()
        
        # Create two columns: sidebar and main content
        self.render_sidebar()
        
        # Main content area
        if not st.session_state.uploaded_files:
            # Welcome screen
            st.markdown("""
            ### 👋 Welcome to ScholarAI!
            
            **Get started by:**
            1. 📁 Upload PDF documents using the sidebar
            2. ⚙️ Configure your preferences 
            3. ❓ Ask questions about your documents
            
            **Features:**
            - 🎓 AI-powered academic question answering
            - 📚 Proper source citations with page numbers
            - 🎨 Multiple response styles (academic, detailed, concise)
            - 💬 Interactive chat interface
            - 🔍 Advanced document retrieval
            
            **Example questions:**
            - "What is the main topic of this document?"
            - "Summarize the key findings and methodology"
            - "Compare different approaches mentioned in the papers"
            - "What are the limitations of this research?"
            """)
        else:
            # Chat interface
            self.render_chat_interface()
        
        # Footer
        self.render_footer()


def main():
    """Main entry point for the Streamlit app."""
    app = ScholarAIApp()
    app.run()


if __name__ == "__main__":
    main()