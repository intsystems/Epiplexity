"""Render epiplexity.md into preview.html: the post as it will look on intsystems.github.io.

    python preview.py        # then open preview.html in Firefox

To publish, copy epiplexity.md to _blogs/epiplexity.md and images/* to images/blog/epiplexity/ in the
intsystems.github.io repository; the post then appears at /materials/blog/epiplexity/.

Citations link to the reference list (<a id="ref-..."> anchors). The site navbar is fixed, and the site
CSS offsets only headings when a link jumps to them, so a clicked reference would land under the navbar.
The same pull request should add, inside `.blog-post__content { ... }` in _sass/_blog.scss:

    // in-page link targets (references) clear the fixed navbar, as headings do
    [id] {
        scroll-margin-top: 120px;
    }

The preview applies this rule (NAVBAR_OFFSET below), so it shows the post as the site will show it then.

Needs pandoc and an internet connection: the page frame (navbar, footer), the site CSS, the fonts and
MathJax come from the live site and its CDN. Without a connection a plain frame is used.

The site renders posts with kramdown, which passes every $$...$$ span to MathJax verbatim: as \\( \\) inline,
as \\[ \\] when the span opens a line of its own. This script does the same, so the math in the preview is the
math the site will show. Single-dollar math is not safe on the site (kramdown eats \\{ and \\|, and turns _
into italics), which is why the post uses $$ everywhere.
"""
import datetime
import html
import io
import os
import re
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
POST = os.path.join(HERE, "epiplexity.md")
OUT = os.path.join(HERE, "preview.html")
SITE = "https://intsystems.github.io"
SLUG = "epiplexity"
FRAME_URL = SITE + "/materials/blog/spherical-loss-family/"   # any live post serves as the frame
NAVBAR_OFFSET = "<style>.blog-post__content [id] { scroll-margin-top: 120px; }</style>"  # proposed site rule

FALLBACK = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title></title>
<link rel="stylesheet" href="https://intsystems.github.io/style.css">
<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']],
processEscapes:true,processEnvironments:true}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script></head>
<body><main id="main" class="container"><div class="content"><article></article></div></main></body></html>"""


def front_matter(text):
    head, body = re.match(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", text, re.S).groups()
    fm, key = {}, None
    for line in head.splitlines():
        item = re.match(r"^\s+-\s+(.*)$", line)
        pair = re.match(r"^(\w+):\s*(.*)$", line)
        if item and key:
            fm[key].append(item.group(1).strip().strip('"'))
        elif pair:
            key, val = pair.groups()
            fm[key] = val.strip().strip('"') if val.strip() else []
    return fm, body


def protect_math(body):
    """Swap every $$...$$ span for a token; return the text and the HTML each token stands for."""
    store = {}

    def swap(kind, tex, prefix=""):
        token = "MATHTOKEN%04dX" % len(store)
        tex = re.sub(r"(^|\n)>[ \t]?", r"\1", tex)                 # drop blockquote markers inside
        tex = html.escape(tex.strip() if kind == "display" else tex, quote=False)
        store[token] = (kind, ("\\[%s\\]" if kind == "display" else "\\(%s\\)") % tex)
        return prefix + token

    # a display span opens with $$ alone at the start of a (possibly quoted) line and closes the same way
    body = re.sub(r"(?m)^(>[ \t]?)?\$\$[ \t]*\n(.*?)\n(?:>[ \t]?)?\$\$[ \t]*$",
                  lambda m: swap("display", m.group(2), m.group(1) or ""), body, flags=re.S)
    body = re.sub(r"\$\$(.+?)\$\$", lambda m: swap("inline", m.group(1)), body, flags=re.S)
    return body, store


def to_html(markdown):
    for fmt in ("gfm+smart", "gfm"):
        run = subprocess.run(["pandoc", "-f", fmt, "-t", "html5", "--wrap=none"], input=markdown,
                             capture_output=True, text=True, encoding="utf-8")
        if run.returncode == 0:
            return run.stdout
    raise SystemExit("pandoc failed:\n" + run.stderr)


def frame():
    try:
        page = urllib.request.urlopen(FRAME_URL, timeout=20).read().decode("utf-8", "replace")
    except OSError:
        return FALLBACK
    page = re.sub(r'((?:href|src)=")/(?!/)', r"\1" + SITE + "/", page)       # root-relative -> live site
    page = page.replace("_blogs/spherical-loss-family.md", "_blogs/%s.md" % SLUG)
    return page


def main():
    fm, body = front_matter(io.open(POST, encoding="utf-8").read())
    body, store = protect_math(body)
    content = to_html(body)
    for token, (kind, tex) in store.items():
        if kind == "display":
            content = content.replace("<p>%s</p>" % token, tex)
        content = content.replace(token, tex)
    content = content.replace('src="/images/blog/%s/' % SLUG, 'src="images/')

    date = datetime.date.fromisoformat(str(fm["date"]))
    esc = lambda s: html.escape(s, quote=False)
    tags = "".join("<li>%s</li>" % esc(t) for t in fm.get("tags", []))
    article = (
        '<article class="blog-post" itemscope itemtype="https://schema.org/BlogPosting">\n'
        '<header class="blog-post__header">\n'
        '<h1 class="blog-post__title page-heading" itemprop="headline">%s</h1>\n'
        '<ul class="blog-post__tags" itemprop="keywords">%s</ul>\n'
        '<div class="blog-post__meta"><time datetime="%s">%s</time>'
        '<span aria-hidden="true">&middot;</span><span>%s min</span>'
        '<span aria-hidden="true">&middot;</span><span itemprop="author">%s</span></div>\n'
        '<p class="blog-post__summary" itemprop="description">%s</p>\n'
        '</header>\n<div class="blog-post__content" itemprop="articleBody">\n%s\n</div>\n</article>'
        % (esc(fm["title"]), tags, date.isoformat(), date.strftime("%B %d, %Y").replace(" 0", " "),
           fm.get("read_time", ""), esc(", ".join(fm.get("authors", []))), esc(fm.get("summary", "")), content))

    page = frame()
    page = re.sub(r"<article.*?</article>", lambda m: article, page, count=1, flags=re.S)
    page = re.sub(r"<title>.*?</title>", "<title>%s (local preview)</title>" % esc(fm["title"]), page, count=1, flags=re.S)
    page = page.replace("</head>", NAVBAR_OFFSET + "\n</head>", 1)
    io.open(OUT, "w", encoding="utf-8").write(page)
    print("wrote %s: %d math spans, %d images" % (OUT, len(store), content.count("<img")))


if __name__ == "__main__":
    main()
