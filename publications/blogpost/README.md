# Blog post

Read [epiplexity.md](epiplexity.md) directly on GitHub. The article connects 
epiplexity and prequential coding to our synthetic time-series
transfer study. The [technical report](../technical_report/README.md) is the
source for our experimental claims.

## Local preview

With Pandoc installed, from this directory:

```bash
python preview.py
```

Open `preview.html` in a browser. The preview loads the site frame and styles
when available; without access to the site it reuses the existing frame or a
fallback. Styles and MathJax may still require Internet access.

## Export for the intsystems website

The website needs image paths rooted at `/images/blog/epiplexity/`. Export a
separate copy, leaving the repository Markdown's relative links intact:

```bash
python - <<'PY'
from pathlib import Path

text = Path("epiplexity.md").read_text()
text = text.replace("](images/", "](/images/blog/epiplexity/")
text = text.replace("cover: images/", "cover: /images/blog/epiplexity/")
Path("/tmp/epiplexity.md").write_text(text)
PY
```

In the separate `intsystems.github.io` repository, copy the exported file to
`_blogs/epiplexity.md` and the contents of `images/` to
`images/blog/epiplexity/`. Publication on that website is separate from the
episynth documentation workflow. The navbar/reference-anchor adjustment for
that site is described in `preview.py`.
