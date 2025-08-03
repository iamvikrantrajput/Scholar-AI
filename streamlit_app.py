#!/usr/bin/env python3
"""
ScholarAI Streamlit app launcher that handles imports correctly
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Change to project root directory
os.chdir(project_root)

# Import and run the app
from app.streamlit_chat import main

if __name__ == "__main__":
    main()