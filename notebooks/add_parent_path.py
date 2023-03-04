""" System functions for Notebooks
"""

import sys
from pathlib import Path


def add_parent_path() -> None:
    """Add parent folder to the python path"""
    sys.path.append(str(Path.cwd().parent))
