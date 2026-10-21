"""Download primary-source metadata for the report; no experiment results are inferred."""

import concurrent.futures
import json
import re
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "build" / "technical_report_sources"
SOURCES = {
    "epiplexity": "https://arxiv.org/abs/2601.03220",
    "episelect": "https://arxiv.org/abs/2608.11746",
    "mix_dont_pick": "https://arxiv.org/abs/2606.09912",
    "moirai": "https://arxiv.org/abs/2402.02592",
    "gift_eval": "https://arxiv.org/abs/2410.10393",
    "chronos": "https://arxiv.org/abs/2403.07815",
    "pfn": "https://arxiv.org/abs/2112.10510",
    "in_context": "https://arxiv.org/abs/2401.15530",
}


def fetch(item):
    key, url = item
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Epiplexity-technical-report/1.0"})
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
