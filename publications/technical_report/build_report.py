"""Compile the bibliography and report, rejecting unresolved references or >7 pages."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=HERE, capture_output=True, encoding="utf-8", errors="replace")
        if result.returncode:
            raise SystemExit(result.stdout + result.stderr)
    log = (HERE / "main.log").read_text(encoding="utf-8", errors="replace")
    count = re.search(r"Output written on main\.pdf \((\d+) pages?", log)
    if count is None:
        raise SystemExit("Could not verify PDF page count")
    pages = int(count.group(1))
    warnings = re.findall(r"[^\n]*(?:undefined|Overfull|LaTeX Error)[^\n]*", log)
    if pages > 7 or warnings:
        raise SystemExit(f"Report gate failed: {pages} pages; {warnings}")
    bibliography = (HERE / "main.bbl").read_text(encoding="utf-8")
    record = {
        "pages": pages,
        "includes_bibliography_and_appendix": True,
        "bibliography_entries": len(re.findall(r"\\bibitem", bibliography)),
        "unresolved_references": False,
        "overfull_boxes": False,
        "pdf_sha256": hashlib.sha256((HERE / "main.pdf").read_bytes()).hexdigest(),
        "tex_sha256": hashlib.sha256((HERE / "main.tex").read_bytes()).hexdigest(),
        "bib_sha256": hashlib.sha256((HERE / "references.bib").read_bytes()).hexdigest(),
    }
    (HERE / "data" / "build_verification.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Built main.pdf: {pages} pages, {record['bibliography_entries']} references, no unresolved references or overfull boxes")
