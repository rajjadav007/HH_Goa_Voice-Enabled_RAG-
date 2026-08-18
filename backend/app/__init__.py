"""HH Goa 2026 Voice RAG Backend Application Package."""

import os
import sys

# Ensure project root directory is in sys.path for top-level module imports (orchestration, voice, etc.)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

__version__ = "1.0.0"

