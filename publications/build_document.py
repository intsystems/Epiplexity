"""Build a publication and verify references, layout, and its page budget."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def build(folder: Path, max_pages: int):
    folder = folder.resolve()
    for command in [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]:
        result = subprocess.run(command, cwd=folder, capture_output=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise SystemExit(result.stdout + result.stderr)
    log = (folder / "main.log").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Output written on main\.pdf \((\d+) pages?", log)
    if match is None:
        raise SystemExit("Could not verify the PDF page count")
    pages = int(match.group(1))
    warnings = re.findall(r"[^\n]*(?:undefined|Overfull|LaTeX Error)[^\n]*", log)
    if pages > max_pages or warnings:
        raise SystemExit(f"Build check failed: {pages} pages (limit {max_pages}); {warnings}")
    bibliography = (folder / "main.bbl").read_text(encoding="utf-8")
    record = {"pages": pages, "max_pages": max_pages, "includes_bibliography_and_appendix": True,
              "bibliography_entries": len(re.findall(r"\\bibitem", bibliography)),
              "unresolved_references": False, "overfull_boxes": False,
              "sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                         for name in ("main.pdf", "main.tex", "references.bib")}}
    (folder / "data" / "build_verification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Built {folder.name}/main.pdf: {pages} pages, {record['bibliography_entries']} references.")
