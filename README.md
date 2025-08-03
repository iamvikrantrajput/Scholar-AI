# 🎓 ScholarAI

A powerful **AI-powered research assistant** built with **Retrieval-Augmented Generation (RAG)** for academic document analysis and intelligent question answering. Upload your research papers, academic documents, or any PDFs and get AI-generated responses with accurate source citations. Built with Python, LangChain, Streamlit, and state-of-the-art language models.

## ✨ Key Features

- 📄 **Smart PDF Processing**: Advanced text extraction with intelligent chunking and metadata preservation
- 🧠 **Multiple LLM Support**: OpenAI GPT models, HuggingFace transformers with automatic fallbacks
- 🔍 **Semantic Search**: High-performance vector similarity search using FAISS or ChromaDB
- 🎓 **Academic-Focused QA**: Context-aware responses optimized for scholarly content
- 🎨 **Multiple Response Styles**: Academic, detailed, concise, comparative, and problem-solving modes
- 📚 **Automatic Source Citations**: Page numbers, document references, and confidence scores
- 🌐 **Modern Web Interface**: Clean, intuitive Streamlit chat interface with real-time responses
- 💻 **Command Line Tools**: CLI for automation, batch processing, and integration
- 🔧 **Modular Architecture**: Clean, extensible codebase with comprehensive error handling
- 🔑 **Flexible API Configuration**: Support for OpenAI, HuggingFace, and demo modes

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/scholar-ai.git
cd scholar-ai

# Install dependencies with Poetry (recommended)
poetry install

# Or use pip
pip install -r requirements.txt

# Activate the Poetry environment
poetry shell
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```bash
# Optional: OpenAI API key for best results
OPENAI_API_KEY="your_openai_api_key_here"

# Optional: HuggingFace API key for additional models
HUGGINGFACE_API_KEY="your_huggingface_token_here"

# Default Configuration
DEFAULT_LLM_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_VECTOR_STORE=faiss
DEFAULT_PROMPT_STYLE=academic

# Advanced Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_SEARCH_RESULTS=5
```

### 3. Launch ScholarAI

```bash
# Web Interface (Recommended)
streamlit run streamlit_app.py

# CLI Interface
python ask.py "What is machine learning?"

# Interactive Chat
python chat.py
```

### 4. Usage Workflow

1. **📁 Upload Documents**: Use the web interface or place PDFs in `data/raw/`
2. **🔄 Auto-Processing**: Documents are automatically chunked and embedded
3. **❓ Ask Questions**: Use natural language queries about your documents
4. **💡 Get Answers**: Receive AI-generated responses with source citations

## 📁 Project Structure

```
scholar-ai/
├── 📂 data/
│   ├── raw/                         # Upload your PDFs here
│   ├── processed/                   # Processed JSON chunks with metadata
│   └── vectorstore/                 # FAISS/ChromaDB indexes
├── 📂 src/
│   ├── app/
│   │   ├── streamlit_chat.py        # 🌐 Web interface (main UI)
│   │   └── cli_chat.py              # 💻 Command-line interface
│   ├── ingestion/
│   │   └── pdf_parser.py            # 📄 PDF processing & chunking
│   ├── vectorstore/
│   │   ├── embeddings.py            # 🧠 Embedding generation
│   │   └── vector_store.py          # 🔍 Vector storage & search
│   ├── llm/
│   │   ├── qa_chain.py              # 🤖 QA system & LLM management
│   │   └── prompts.py               # 📝 Custom prompt templates
│   └── retrieval/
│       └── langchain_retriever.py   # 🔗 Document retrieval logic
├── 📜 streamlit_app.py              # Streamlit launcher
├── 📜 chat.py                       # CLI launcher
├── 📜 ask.py                        # Quick question tool
├── 📂 tests/                        # Comprehensive test suite
├── 📜 run.sh                        # Automation script
├── 📜 run_tests.py                  # Test runner
├── 📜 final_validation.py           # End-to-end validation
└── 📜 pyproject.toml               # Poetry dependencies
```

## 🛠️ Usage Examples

### Web Interface (Recommended)

```bash
# Launch the ScholarAI web application
streamlit run streamlit_app.py

# Open browser to: http://localhost:8501
```

**Web Interface Features:**
- 📤 Drag-and-drop PDF upload
- 💬 Real-time chat interface  
- 🎛️ Response style selection
- 🔧 LLM provider switching
- 📚 Inline source citations
- 🔑 API key management
- 📊 Document processing status

### Command Line Interface

```bash
# Quick single question
python ask.py "What is linear programming?"

# Interactive chat mode
python chat.py --llm-provider openai

# Batch processing with specific settings
python chat.py \
  --question "Explain neural networks" \
  --prompt-style academic \
  --llm-provider huggingface \
  --verbose
```

### Response Styles

Choose from 5 specialized response modes:

- **🎓 Academic**: Professional, scholarly responses with formal citations
- **📖 Detailed**: Comprehensive explanations with examples and context
- **⚡ Concise**: Brief, direct answers focusing on essential information
- **⚖️ Comparative**: Side-by-side analysis of concepts and approaches
- **🔧 Problem-solving**: Step-by-step guidance and practical solutions

## ⚙️ Configuration Options

### LLM Providers

| Provider | Models | Cost | Setup | Quality |
|----------|--------|------|-------|---------|
| **OpenAI** | GPT-3.5, GPT-4 | Paid API | API Key Required | Excellent |
| **HuggingFace** | Various OSS Models | Free/Paid | API Key Optional | Good |
| **Demo Mode** | Mock Responses | Free | No Setup | Basic |

### Vector Stores

| Store | Use Case | Performance | Persistence |
|-------|----------|-------------|-------------|
| **FAISS** | Fast similarity search | Excellent | File-based |
| **ChromaDB** | Persistent database | Good | SQLite-based |

### Embedding Models

| Provider | Model | Dimensions | Speed | Quality |
|----------|-------|------------|-------|---------|
| **HuggingFace** | all-MiniLM-L6-v2 | 384 | Fast | Good |
| **OpenAI** | text-embedding-3-small | 1536 | Fast | Excellent |

## 🧪 Current Status & Features

### ✅ Implemented Features
- **PDF Processing**: Complete pipeline with chunking and metadata
- **Vector Storage**: FAISS and ChromaDB support
- **Multiple LLMs**: OpenAI, HuggingFace, and demo modes
- **Web Interface**: Modern Streamlit UI with real-time chat
- **CLI Tools**: Interactive and single-question modes
- **Source Citations**: Automatic page-level references
- **Response Styles**: 5 different academic response modes
- **Error Handling**: Graceful fallbacks and user feedback
- **Environment Management**: Secure API key handling

### 🔄 Recent Updates
- **Project Renamed**: ScholarGPT → ScholarAI
- **Enhanced UI**: Cleaner answer display with inline sources
- **Fixed API Issues**: Resolved OpenAI and HuggingFace parameter conflicts
- **Improved Fallbacks**: Better demo mode when no API keys provided
- **Streamlined Sources**: Single answer with organized citations

## 📋 System Requirements

### Minimum Requirements
- **OS**: macOS, Linux, Windows (WSL recommended)
- **Python**: 3.10+ (3.12 recommended)
- **RAM**: 8GB minimum, 16GB recommended for local models
- **Storage**: 2GB for models and data

### Optional Enhancements
- **GPU**: CUDA-compatible for faster local embeddings
- **API Keys**: OpenAI for premium models, HuggingFace for additional options
- **Docker**: For containerized deployment (roadmap)

## 🚀 Advanced Usage

### Custom Prompt Development

```python
# Create custom prompt templates
from src.llm.prompts import PromptTemplateManager

manager = PromptTemplateManager()
manager.add_template("research", """
You are a research assistant specializing in {domain}.
Context: {context}
Question: {question}
Provide a detailed analysis with methodology and implications.
""")
```

### Programmatic API

```python
# Use ScholarAI programmatically
from src.llm.qa_chain import create_qa_chain

qa_system = create_qa_chain(
    llm_provider="openai",
    prompt_style="academic"
)

response = qa_system.answer("What are the key findings?")
print(f"Answer: {response['answer']}")
print(f"Sources: {len(response['sources'])} documents")
```

## 🛠️ Testing & Validation

```bash
# Run complete validation suite
python final_validation.py

# Run all unit tests
python run_tests.py

# Test specific components
python -m pytest tests/test_pdf_processing.py
python -m pytest tests/test_vector_store.py
python -m pytest tests/test_qa_pipeline.py
```

## 🔒 Security & Privacy

- **🏠 Local Processing**: HuggingFace models run entirely offline
- **🔐 Secure Keys**: API keys stored in environment variables only
- **🛡️ Data Privacy**: Documents stay on your system (except OpenAI API calls)
- **🗄️ Local Storage**: Vector stores and processed data remain local

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with these amazing technologies:

- **[LangChain](https://langchain.com/)** - RAG framework and LLM orchestration
- **[Streamlit](https://streamlit.io/)** - Interactive web application framework
- **[FAISS](https://faiss.ai/)** - Efficient similarity search and clustering
- **[HuggingFace](https://huggingface.co/)** - Open-source transformer models
- **[OpenAI](https://openai.com/)** - GPT language models and embeddings
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** - Fast PDF processing
- **[Sentence Transformers](https://www.sbert.net/)** - State-of-the-art embeddings

## 🆘 Support & Troubleshooting

### Common Issues

**Q: "Invalid API key" errors**
- Check your `.env` file has correct API keys without quotes
- Restart the application after updating environment variables

**Q: "No documents found" errors**  
- Upload PDFs through the web interface or place in `data/raw/`
- Check that PDFs contain extractable text (not just images)

**Q: Poor answer quality**
- Try different response styles (academic, detailed, etc.)
- Upload more relevant documents for better context
- Consider using OpenAI models for higher quality responses

### Getting Help
- 📖 **Documentation**: Check this README and inline code comments
- 🐛 **Bug Reports**: [Create an issue](https://github.com/your-username/scholar-ai/issues)
- 💬 **Discussions**: [Join the community](https://github.com/your-username/scholar-ai/discussions)

## 🔮 Roadmap

### Planned Features
- [ ] **Multi-modal Support**: Extract and analyze images, tables, charts
- [ ] **Advanced Analytics**: Document similarity, topic clustering
- [ ] **Collaborative Features**: Team document sharing and annotation
- [ ] **API Server**: RESTful API for third-party integrations
- [ ] **Enhanced UI**: Document preview, advanced search filters
- [ ] **Export Options**: PDF reports, citations, summaries

### Technical Improvements
- [ ] **Performance**: Faster embeddings, streaming responses
- [ ] **Scalability**: Support for larger document collections
- [ ] **Monitoring**: Usage analytics and performance metrics
- [ ] **Deployment**: Docker containers, cloud deployment guides

---

## 🚀 **Ready to Get Started?**

```bash
# One-command setup and launch
git clone https://github.com/your-username/scholar-ai.git
cd scholar-ai
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Transform your academic research workflow with AI!** 🎓✨

---

*ScholarAI - Making academic research more efficient and insightful through the power of AI*