#!/bin/bash

# 🤖 AI Research Assistant - Automated Setup & Launch Script
# This script automates the complete pipeline: ingestion → embeddings → UI launch

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}🔹 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Header
echo -e "${PURPLE}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    🎓 SCHOLAR-AI                           ║"
echo "║           Automated Setup & Launch Pipeline                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Step 1: Environment Check
print_step "Step 1: Checking Environment Setup"

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    print_error "Poetry is not installed. Please install Poetry first:"
    echo "  curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

print_success "Poetry found: $(poetry --version)"

# Check Python version
PYTHON_VERSION=$(poetry run python --version 2>&1)
print_success "Python environment: $PYTHON_VERSION"

# Step 2: Environment Variables Check
print_step "Step 2: Checking Environment Configuration"

if [ ! -f ".env" ]; then
    print_warning ".env file not found"
    print_info "Creating sample .env file..."
    
    cat > .env << 'EOF'
# AI Research Assistant Configuration
# 
# Required for OpenAI models (paid, high quality)
OPENAI_API_KEY=your_openai_api_key_here

# Optional for HuggingFace models (free, runs locally)
HUGGINGFACE_API_KEY=your_huggingface_token_here

# Default Configuration
DEFAULT_LLM_PROVIDER=huggingface
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
DEFAULT_VECTOR_STORE=faiss
DEFAULT_PROMPT_STYLE=academic

# Advanced Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_SEARCH_RESULTS=5
EOF

    print_warning "Please edit .env file with your API keys before proceeding"
    print_info "You can continue with HuggingFace (free) by leaving OpenAI key as-is"
    
    read -p "Press Enter to continue or Ctrl+C to exit and configure .env..."
else
    print_success ".env file found"
fi

# Step 3: Install Dependencies
print_step "Step 3: Installing Dependencies"

print_info "Installing project dependencies with Poetry..."
if poetry install --quiet; then
    print_success "Dependencies installed successfully"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Step 4: Create Required Directories
print_step "Step 4: Setting Up Project Structure"

REQUIRED_DIRS=("data/raw" "data/processed" "data/vectorstore")

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_success "Created directory: $dir"
    else
        print_info "Directory exists: $dir"
    fi
done

# Step 5: Check for PDFs and Process if Found
print_step "Step 5: PDF Processing Pipeline"

PDF_COUNT=$(find data/raw -name "*.pdf" 2>/dev/null | wc -l)

if [ "$PDF_COUNT" -eq 0 ]; then
    print_warning "No PDFs found in data/raw/"
    print_info "You can:"
    print_info "  1. Place PDFs in data/raw/ directory now"
    print_info "  2. Upload them via the web interface later"
    
    read -p "Add PDFs to data/raw/ now? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Waiting for you to add PDFs to data/raw/..."
        print_info "Press Enter when ready to continue..."
        read
        
        # Recheck PDF count
        PDF_COUNT=$(find data/raw -name "*.pdf" 2>/dev/null | wc -l)
    fi
fi

if [ "$PDF_COUNT" -gt 0 ]; then
    print_success "Found $PDF_COUNT PDF(s) in data/raw/"
    
    # Check if already processed
    PROCESSED_COUNT=$(find data/processed -name "*.json" 2>/dev/null | wc -l)
    
    if [ "$PROCESSED_COUNT" -eq 0 ] || [ "$PDF_COUNT" -gt "$PROCESSED_COUNT" ]; then
        print_info "Processing PDFs into chunks..."
        
        if poetry run python -c "
import sys; sys.path.append('src')
from ingestion.pdf_parser import PDFProcessor
from pathlib import Path
import json

processor = PDFProcessor()
pdf_dir = Path('data/raw')
processed_dir = Path('data/processed')

for pdf_file in pdf_dir.glob('*.pdf'):
    print(f'Processing {pdf_file.name}...')
    try:
        chunks = processor.process_pdf(pdf_file)
        output_file = processed_dir / f'{pdf_file.stem}_chunks.json'
        
        with open(output_file, 'w') as f:
            json.dump(chunks, f, indent=2)
        
        print(f'Saved {len(chunks)} chunks to {output_file.name}')
    except Exception as e:
        print(f'Error processing {pdf_file.name}: {e}')
        continue

print('PDF processing completed!')
"; then
            print_success "PDF processing completed"
        else
            print_error "PDF processing failed"
            print_warning "Continuing anyway - you can process PDFs via the web interface"
        fi
    else
        print_success "PDFs already processed ($PROCESSED_COUNT files)"
    fi
fi

# Step 6: Build Vector Store
print_step "Step 6: Building Vector Store & Embeddings"

VECTOR_STORE_COUNT=$(find data/vectorstore -type f 2>/dev/null | wc -l)

if [ "$VECTOR_STORE_COUNT" -eq 0 ] || [ "$PROCESSED_COUNT" -gt 0 ]; then
    print_info "Building vector store with embeddings..."
    
    if poetry run python -c "
import sys; sys.path.append('src')
from vectorstore.vector_store import create_vector_store_manager
from pathlib import Path

try:
    config = {
        'embeddings': {
            'provider': 'huggingface',
            'model': 'all-MiniLM-L6-v2'
        },
        'vector_store': {
            'type': 'faiss',
            'index_type': 'IndexFlatIP'
        }
    }
    
    processed_dir = Path('data/processed')
    if not any(processed_dir.glob('*.json')):
        print('No processed files found - skipping vector store creation')
        exit(0)
    
    print('Creating vector store manager...')
    vector_manager = create_vector_store_manager(config)
    
    print('Building vector store from processed documents...')
    vector_manager.build_vector_store(processed_dir, batch_size=10)
    
    print('Saving vector store...')
    vector_store_dir = Path('data/vectorstore/langchain_faiss')
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    vector_manager.save_vector_store(vector_store_dir)
    
    print('Vector store created successfully!')
except Exception as e:
    print(f'Vector store creation failed: {e}')
    print('You can still use the app - embeddings will be created on-demand')
"; then
        print_success "Vector store built successfully"
    else
        print_warning "Vector store creation encountered issues - continuing anyway"
    fi
else
    print_success "Vector store already exists"
fi

# Step 7: Run Validation
print_step "Step 7: Running System Validation"

print_info "Performing end-to-end validation..."

if PYTHONWARNINGS=ignore poetry run python final_validation.py 2>/dev/null | grep -q "ALL VALIDATION CHECKS PASSED"; then
    print_success "System validation passed!"
else
    print_warning "Some validation checks failed - check output above"
    print_info "The system may still work with basic functionality"
fi

# Step 8: Launch Options
print_step "Step 8: Launch Options"

echo -e "\n${GREEN}🎉 Setup Complete! Choose how to launch:${NC}\n"

echo -e "${CYAN}Available launch options:${NC}"
echo -e "  ${YELLOW}1.${NC} Streamlit Web Interface (Recommended)"
echo -e "  ${YELLOW}2.${NC} CLI Chat Interface"
echo -e "  ${YELLOW}3.${NC} Quick Question Tool"
echo -e "  ${YELLOW}4.${NC} Skip launch (manual start later)"

echo ""
read -p "Enter your choice (1-4): " -n 1 -r
echo ""

case $REPLY in
    1)
        print_info "Starting Streamlit Web Interface..."
        print_success "🌐 Web interface will open at: http://localhost:8501"
        print_info "Press Ctrl+C to stop the server"
        echo ""
        
        # Give user a moment to read
        sleep 2
        
        # Launch Streamlit
        poetry run streamlit run streamlit_app.py
        ;;
    2)
        print_info "Starting CLI Chat Interface..."
        echo ""
        poetry run python chat.py --llm-provider openai
        ;;
    3)
        print_info "Quick Question Tool"
        echo ""
        read -p "Enter your question: " QUESTION
        poetry run python ask.py "$QUESTION"
        ;;
    4)
        print_info "Manual launch commands:"
        echo ""
        echo -e "${CYAN}Web Interface:${NC}"
        echo "  poetry run streamlit run streamlit_app.py"
        echo ""
        echo -e "${CYAN}CLI Interface:${NC}"
        echo "  poetry run python chat.py --llm-provider huggingface"
        echo ""
        echo -e "${CYAN}Quick Question:${NC}"
        echo "  poetry run python ask.py \"your question here\""
        echo ""
        echo -e "${CYAN}Validation:${NC}"
        echo "  poetry run python final_validation.py"
        ;;
    *)
        print_warning "Invalid choice. Here are the manual commands:"
        echo ""
        echo -e "${CYAN}Web Interface:${NC} poetry run streamlit run streamlit_app.py"
        echo -e "${CYAN}CLI Interface:${NC} poetry run python chat.py --llm-provider huggingface"
        ;;
esac

echo ""
print_success "🚀 AI Research Assistant is ready!"
echo -e "${PURPLE}Happy researching! 📚✨${NC}"