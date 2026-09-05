"""
Convenience runner for Phase 2: LLM APIs & Prompt Engineering.
Enables running `python main.py` directly from either the phase folder
or the project folder.
"""

import asyncio
import os
import sys

# Set working directory and sys.path to the project folder
project_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project")
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

os.chdir(project_dir)

from main import main

if __name__ == "__main__":
    asyncio.run(main())
