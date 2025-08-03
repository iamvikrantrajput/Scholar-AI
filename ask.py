#!/usr/bin/env python3
"""
Simple ask command for ScholarAI
Usage: python ask.py "your question here"
"""

import sys
import os

# Add src to a path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py \"your question here\"")
        print("Example: python ask.py \"What is linear programming?\"")
        return
    
    question = " ".join(sys.argv[1:])
    
    # Import after path setup
    from app.cli_chat import single_question_mode
    
    # Run with sensible defaults - prioritize OpenAI for reliability
    import os
    
    # Prioritize OpenAI for better reliability, fallback to HuggingFace
    if os.getenv("OPENAI_API_KEY"):
        llm_provider = "openai"
        print("🔑 Using OpenAI (most reliable)")
    elif os.getenv("HUGGINGFACE_API_KEY"):
        llm_provider = "huggingface"
        print("🔑 Using HuggingFace (may have some limitations)")
    else:
        llm_provider = "huggingface"  # This will use demo mode
        print("🔑 No API keys found - using demo mode")
    
    try:
        single_question_mode(
            question=question,
            llm_provider=llm_provider,
            prompt_style="academic",
            show_sources=True,
            verbose=True  # Enable verbose to see what's happening
        )
    except Exception as e:
        print(f"\n❌ Critical Error in ask.py: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()