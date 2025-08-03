#!/usr/bin/env python3
"""
Test script for LangChain Retrieval System
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from retrieval.langchain_retriever import (
    create_retriever_from_json,
    create_retriever_from_existing_store,
    format_retrieval_results
)


def test_retriever_creation():
    """Test creating retriever from JSON files."""
    print("🚀 Testing LangChain Retriever Creation...")
    
    # Configuration
    embedding_config = {
        "provider": "huggingface",
        "model": "all-MiniLM-L6-v2"
    }
    
    search_kwargs = {
        "k": 3,
        "score_threshold": 0.1
    }
    
    processed_dir = Path("data/processed")
    save_path = Path("data/vectorstore/langchain_faiss")
    
    if not processed_dir.exists():
        print(f"❌ Processed directory not found: {processed_dir}")
        return None
    
    # Create retriever
    try:
        retriever = create_retriever_from_json(
            processed_dir=processed_dir,
            embedding_config=embedding_config,
            store_type="faiss",
            save_path=save_path,
            search_kwargs=search_kwargs
        )
        print("✅ Retriever created successfully")
        return retriever
    except Exception as e:
        print(f"❌ Error creating retriever: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_retrieval_queries(retriever):
    """Test retrieval with various queries."""
    print("\n🔍 Testing Retrieval Queries...")
    
    test_queries = [
        "What is linear programming?",
        "How do you solve optimization problems?",
        "What is the maximum flow problem?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: '{query}'")
        
        try:
            # Retrieve with scores
            docs_with_scores = retriever.retrieve_with_scores(query, k=2)
            
            print(f"📊 Found {len(docs_with_scores)} results:")
            
            for i, (doc, score) in enumerate(docs_with_scores, 1):
                metadata = doc.metadata
                print(f"\n{i}. Score: {score:.3f}")
                print(f"   Source: {metadata.get('source', 'unknown')} (Page {metadata.get('page', 'unknown')})")
                print(f"   Words: {metadata.get('word_count', 'unknown')}")
                print(f"   Text: {doc.page_content[:200]}...")
                
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")


def main():
    """Main test function."""
    print("🧪 LangChain Retrieval System Test Suite")
    print("=" * 50)
    
    # Test 1: Create retriever from JSON
    retriever = test_retriever_creation()
    
    if retriever:
        # Test 2: Test retrieval queries
        test_retrieval_queries(retriever)
    
    print("\n🎉 Testing complete!")


if __name__ == "__main__":
    main()