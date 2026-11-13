"""Compile and check the research article."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_document import build

if __name__ == "__main__":
    build(HERE, max_pages=8)
