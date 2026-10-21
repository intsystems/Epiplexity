"""Sphinx configuration; builds work from any current directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import episynth  # noqa: E402

project = "episynth"
copyright = "2026, Epiplexity team"
author = "Epiplexity team"
version = release = episynth.__version__
root_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_ivar = True
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_title = "episynth documentation"
html_baseurl = "https://intsystems.github.io/Epiplexity/"
html_context = {
    "display_github": True,
    "github_user": "intsystems",
    "github_repo": "Epiplexity",
    "github_version": "main",
    "conf_py_path": "/doc/source/",
}
