"""Sphinx configuration for epimeter (skeleton; filled in at stage 7)."""

import sys
from pathlib import Path

import epimeter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

project = "epimeter"
copyright = "2026, Epiplexity team"
author = "Epiplexity team"
release = epimeter.__version__
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
