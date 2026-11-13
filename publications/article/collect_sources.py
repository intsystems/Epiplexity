"""Download arXiv metadata for the article bibliography."""

import concurrent.futures
import json
import re
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "build" / "article_sources"
SOURCES = {
    "epiplexity": "https://arxiv.org/abs/2601.03220",
    "episelect": "https://arxiv.org/abs/2608.11746",
    "mix_dont_pick": "https://arxiv.org/abs/2606.09912",
    "moirai": "https://arxiv.org/abs/2402.02592",
    "gift_eval": "https://arxiv.org/abs/2410.10393",
    "chronos": "https://arxiv.org/abs/2403.07815",
    "prequential": "https://arxiv.org/abs/1802.07044",
    "requential": "https://arxiv.org/abs/2607.11883",
    "chronos2": "https://arxiv.org/abs/2510.15821",
    "tirex": "https://arxiv.org/abs/2505.23719",
    "diffusion_elbo": "https://arxiv.org/abs/2303.00848",
}


def fetch(item):
    key, url = item
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Epiplexity-article/1.0"})
        with urllib.request.urlopen(request, timeout=35) as response:
            html = response.read().decode("utf-8")
        (OUT / (key + ".html")).write_text(html, encoding="utf-8")
        metadata = re.findall(r'<meta\s+name="(citation_[^"]+)"\s+content="([^"]*)"', html)
        return {"key": key, "url": url, "metadata": metadata, "status": "retrieved"}
    except Exception as exc:
        return {"key": key, "url": url, "status": "unavailable", "error": str(exc)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, SOURCES.items()))
    (OUT / "metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    for record in records:
        print(json.dumps(record, ensure_ascii=True))
