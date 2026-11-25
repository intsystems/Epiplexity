"""Render epiplexity.md into preview.html: the post as it will look on
intsystems.github.io.

    python preview.py        # then open preview.html in Firefox

Image paths in epiplexity.md are relative so the post renders on GitHub. For
publication
on intsystems.github.io, export a copy with site-relative image paths as
described in
README.md, then copy that Markdown and the images to the site repository.

Citations link to the reference list (<a id="ref-..."> anchors). The site
navbar is fixed, and the site
CSS offsets only headings when a link jumps to them, so a clicked reference
would land under the navbar.
The same pull request should add, inside `.blog-post__content { ... }` in
_sass/_blog.scss:

    // in-page link targets (references) clear the fixed navbar, as headings do
    [id] {
        scroll-margin-top: 120px;
    }

The preview applies this rule (NAVBAR_OFFSET below), so it shows the post as
the site will show it then.

Needs pandoc and an internet connection: the page frame (navbar, footer), the
site CSS, the fonts and
MathJax come from the live site and its CDN. Without a connection a plain frame
is used.

The site renders posts with kramdown, which passes every $$...$$ span to
MathJax verbatim: as \\( \\) inline,
as \\[ \\] when the span opens a line of its own. This script does the same, so
the math in the preview is the
math the site will show. Single-dollar math is not safe on the site (kramdown
eats \\{ and \\|, and turns _
into italics), which is why the post uses $$ everywhere.
"""

import datetime
import html
import re
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
POST = HERE / "epiplexity.md"
OUT = HERE / "preview.html"
SITE = "https://intsystems.github.io"
SLUG = "epiplexity"
FRAME_URL = (
    SITE + "/materials/blog/spherical-loss-family/"
)  # any live post serves as the frame
NAVBAR_OFFSET = (  # proposed site rule
    "<style>.blog-post__content [id] { scroll-margin-top: 120px; }</style>"
)

FALLBACK = (
    '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>'
    '</title>\n'
    '<link rel="stylesheet" href="https://intsystems.github.io/'
    'style.css">\n'
    '<script>'
    "window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displa"
    "yMath:[['$$','$$'],['\\\\[','\\\\]']],\n"
    'processEscapes:true,processEnvironments:true}};</script>\n'
    '<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/'
    'tex-mml-chtml.js"></script></head>\n'
    '<body><main id="main" class="container"><div class="content">'
    '<article></article></div></main></body></html>'
)


def front_matter(text):
    head, body = re.match(
        r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", text, re.S
    ).groups()
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
    """Swap every $$...$$ span for a token; return the text and the HTML each
    token stands for.
    """
    store = {}

    def swap(kind, tex, prefix=""):
        token = f"MATHTOKEN{len(store):04d}X"
        tex = re.sub(
            r"(^|\n)>[ \t]?", r"\1", tex
        )  # drop blockquote markers inside
        tex = html.escape(
            tex.strip() if kind == "display" else tex, quote=False
        )
        store[token] = (
            kind,
            ("\\[%s\\]" if kind == "display" else "\\(%s\\)") % tex,
        )
        return prefix + token

    # a display span opens with $$ alone at the start of a (possibly quoted)
    # line and closes the same way
    body = re.sub(
        r"(?m)^(>[ \t]?)?\$\$[ \t]*\n(.*?)\n(?:>[ \t]?)?\$\$[ \t]*$",
        lambda m: swap("display", m.group(2), m.group(1) or ""),
        body,
        flags=re.S,
    )
    body = re.sub(
        r"\$\$(.+?)\$\$",
        lambda m: swap("inline", m.group(1)),
        body,
        flags=re.S,
    )
    return body, store


def to_html(markdown):
    for fmt in ("gfm+smart", "gfm"):
        run = subprocess.run(
            ["pandoc", "-f", fmt, "-t", "html5", "--wrap=none"],
            input=markdown,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if run.returncode == 0:
            return run.stdout
    raise SystemExit("pandoc failed:\n" + run.stderr)


def frame():
    try:
        page = (
            urllib.request.urlopen(FRAME_URL, timeout=20)
            .read()
            .decode("utf-8", "replace")
        )
    except OSError:
        # Keep the previous site frame when regenerating without network
        # access.
        if OUT.exists():
            cached = OUT.read_text(encoding="utf-8")
            if '<nav' in cached and 'blog-post__content' in cached:
                return cached.replace(NAVBAR_OFFSET, "")
        return FALLBACK
    page = re.sub(
        r'((?:href|src)=")/(?!/)', r"\1" + SITE + "/", page
    )  # root-relative -> live site
    page = page.replace(
        "_blogs/spherical-loss-family.md", f"_blogs/{SLUG}.md"
    )
    return page


def main():
    fm, body = front_matter(POST.read_text(encoding="utf-8"))
    body, store = protect_math(body)
    content = to_html(body)
    for token, (kind, tex) in store.items():
        if kind == "display":
            content = content.replace(f"<p>{token}</p>", tex)
        content = content.replace(token, tex)
    content = content.replace(f'src="/images/blog/{SLUG}/', 'src="images/')

    date = datetime.date.fromisoformat(str(fm["date"]))

    def esc(s):
        return html.escape(s, quote=False)

    tags = "".join(f"<li>{esc(t)}</li>" for t in fm.get("tags", []))
    article = (
        '<article class="blog-post" itemscope'
        ' itemtype="https://schema.org/BlogPosting">\n<header'
        ' class="blog-post__header">\n<h1 class="blog-post__title'
        ' page-heading" itemprop="headline">{}</h1>\n<ul'
        ' class="blog-post__tags" itemprop="keywords">{}</ul>\n<div'
        ' class="blog-post__meta"><time datetime="{}">{}</time><span'
        ' aria-hidden="true">&middot;</span><span>{} min</span><span'
        ' aria-hidden="true">&middot;</span><span'
        ' itemprop="author">{}</span></div>\n<p class="blog-post__summary"'
        ' itemprop="description">{}</p>\n</header>\n<div'
        ' class="blog-post__content"'
        ' itemprop="articleBody">\n{}\n</div>\n</article>'.format(
            esc(fm["title"]),
            tags,
            date.isoformat(),
            date.strftime("%B %d, %Y").replace(" 0", " "),
            fm.get("read_time", ""),
            esc(", ".join(fm.get("authors", []))),
            esc(fm.get("summary", "")),
            content,
        )
    )

    page = frame()
    page = re.sub(
        r"<article.*?</article>", lambda m: article, page, count=1, flags=re.S
    )
    page = re.sub(
        r"<title>.*?</title>",
        "<title>{} (local preview)</title>".format(esc(fm["title"])),
        page,
        count=1,
        flags=re.S,
    )
    page = page.replace("</head>", NAVBAR_OFFSET + "\n</head>", 1)
    page = "\n".join(line.rstrip() for line in page.splitlines()) + "\n"
    OUT.write_text(page, encoding="utf-8")
    print(
        f"wrote {OUT}: {len(store)} math spans, "
        f"{content.count('<img')} images"
    )


if __name__ == "__main__":
    main()
