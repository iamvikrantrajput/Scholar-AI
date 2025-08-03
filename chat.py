#!/usr/bin/env python3
"""
Simple wrapper for the ScholarAI CLI
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the CLI
from app.cli_chat import main

if __name__ == "__main__":
    main()