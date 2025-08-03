#!/usr/bin/env python3
"""
Test embedding and vector store creation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from vectorstore.vector_store import create_vector_store_manager

def test_vector_store_creation():
    """Test the vector store creation pipeline."""
    print("🔹 Testing Embedding & Vector Store Creation")
    print("=" * 50)
    
    # Check if processed data exists
    processed_dir = Path("data/processed").resolve()
    json_files = list(processed_dir.glob("*.json"))
    
    print(f"📂 Looking in: {processed_dir.absolute()}")
    print(f"📄 Found JSON files: {len(json_files)}")
    
    if not json_files:
        print("⚠️  No processed JSON files found in data/processed/")
        print("   Run PDF processing first")
        return False
    
    try:
        # Create vector store manager
        print("🔧 Creating vector store manager...")
        config = {
            "embeddings": {
                "provider": "huggingface",
                "model": "all-MiniLM-L6-v2"
            },
            "vector_store": {
                "type": "faiss",
                "index_type": "IndexFlatIP"
            }
        }
        
        vector_manager = create_vector_store_manager(config)
        print("✅ Vector store manager created")
        
        # Build vector store
        print("🚀 Building vector store...")
        vector_manager.build_vector_store(processed_dir, batch_size=5)
        print("✅ Vector store built successfully")
        
        # Test search
        print("🔍 Testing search functionality...")
        results = vector_manager.search("linear programming", k=2)
        print(f"📊 Search returned {len(results)} results")
        
        if results:
            print(f"🎯 Top result score: {results[0].get('score', 'N/A'):.3f}")
            print(f"📝 Top result source: {results[0].get('metadata', {}).get('source', 'Unknown')}")
        
        # Save vector store
        print("💾 Saving vector store...")
        vector_store_dir = Path("data/vectorstore/test_faiss").resolve()
        vector_store_dir.mkdir(parents=True, exist_ok=True)
        vector_manager.save_vector_store(vector_store_dir)
        print("✅ Vector store saved")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in vector store creation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_store_creation()
    print(f"\n{'✅ Vector Store Test PASSED' if success else '❌ Vector Store Test FAILED'}")