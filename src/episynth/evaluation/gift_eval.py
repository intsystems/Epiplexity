"""GIFT-Eval data (Aksu et al., 2024, arXiv:2410.10393): datasets, horizons,
test windows.

Repeat. `gift_eval.data.Dataset` from github.com/SalesforceAIResearch/gift-eval
(Apache-2.0) free of
gluonts and `datasets`: image of the repository `Salesforce/GiftEval` It reads
directly from arrow
files. The horizon is given by frequency and thermometer, the number of test
windows is the tenth
part of the shortest row of a dataset, but no more than that. 20 ooh M4 One
window; the windows go
in succession increments into the horizon to the end of the row, the history of
the window to the
whole row. `_dim<k>`Seasonality in frequency, like
`gluonts.time_feature.get_seasonality`.
"""

import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pyarrow as pa

SHORT_CONFIGS: tuple[str, ...] = tuple(
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly"
    " electricity/15T electricity/H electricity/D electricity/W solar/10T"
    " solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M"
    " us_births/W saugeenday/D saugeenday/M saugeenday/W"
    " temperature_rain_with_missing kdd_cup_2018_with_missing/H"
    " kdd_cup_2018_with_missing/D car_parts_with_missing restaurant"
    " hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H"
    " LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H"
    " ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T"
    " jena_weather/H jena_weather/D bitbrains_fast_storage/5T"
    " bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H"
    " bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
    .split()
)
"""Names of datasets of the short-term part of the benchmark (term short),
55 pieces.
"""

_PRETTY = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
_TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}
_PRED_LENGTH = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
_M4_PRED_LENGTH = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
_SEASONALITY = {
    "S": 3600,
    "T": 1440,
    "H": 24,
    "D": 1,
    "W": 1,
    "M": 12,
    "B": 5,
    "Q": 4,
    "A": 1,
}
_BASE_ALIASES = {
    "min": "T",
    "T": "T",
    "H": "H",
    "h": "H",
    "D": "D",
    "W": "W",
    "M": "M",
    "ME": "M",
    "MS": "M",
    "Q": "Q",
    "QE": "Q",
    "QS": "Q",
    "A": "A",
    "Y": "A",
    "YE": "A",
    "YS": "A",
    "AS": "A",
    "S": "S",
    "s": "S",
    "B": "B",
}
_TEST_SPLIT = 0.1
_MAX_WINDOWS = 20


def parse_freq(freq: str) -> tuple[str, int]:
    """The base frequency of the {S, T, H, D, W, M, Q, A, B} and the
    multiplier, for example, `"15T"` gives `("T", 15)`.
    """
    m = re.fullmatch(r"(\d*)([A-Za-z]+)(?:-\w+)?", freq)
    assert m, freq
    return _BASE_ALIASES[m.group(2)], int(m.group(1) or 1)


def seasonality(freq: str) -> int:
    """Frequency season according to the gluonts rule: base seasonality divided
    by a multiplier, otherwise 1.
    """
    base, n = parse_freq(freq)
    q, r = divmod(_SEASONALITY.get(base, 1), n)
    return q if r == 0 else 1


def config_name(name: str, freq: str, term: str) -> str:
    """The configuration name in the leaderboard tables of the form
    `saugeen/D/short`: frequency without anchor (`A-DEC` gives `A`).
    """
    key = name.split("/")[0].lower()
    key = _PRETTY.get(key, key)
    base, n = parse_freq(name.split("/")[1] if "/" in name else freq)
    return f"{key}/{n if n > 1 else ''}{base}/{term}"


@dataclass
class TestInstance:
    """One test window: the story before the window and the truth.

    Attributes:
        item_id: The name of the series, in multidimensional datasets with the
            suffix `_dim<k>`.
        window: The window number is in a row from 0.
        history: All row to the `(T,)` float32 window, with possible NaN.
        label: True `(h,)` float32, with possible NaN.
    """

    item_id: str
    window: int
    history: np.ndarray
    label: np.ndarray


class GiftDataset:
    """Dataset GIFT-Eval from the local image repository `Salesforce/GiftEval`.

    Args:
        root: Dataset snapshot directory.
        name: The name of the dataset, as in `SHORT_CONFIGS`, for example
            `"electricity/15T"`.
        term: Therm from {"short", "medium", "long"}.

    Attributes:
        freq: Frequency line of the dataset.
        target_dim: The number of variables in the reference series.
        prediction_length: Horizon.
        windows: Number of test windows per row.
        season: Seasonal frequency.
        config: The configuration name in the leaderboard tables.
    """

    def __init__(
        self, root: str | Path, name: str, term: str = "short"
    ) -> None:
        self.root = Path(root)
        self.name = name
        self.term = term
        self._table = self._read_table(self.root / name)
        self.freq = str(self._table.column("freq")[0].as_py())
        first = self._table.column("target")[0].as_py()
        self.target_dim = len(first) if isinstance(first[0], list) else 1
        base, _ = parse_freq(self.freq)
        pred = _M4_PRED_LENGTH[base] if "m4" in name else _PRED_LENGTH[base]
        self.prediction_length = _TERM_MULTIPLIER[term] * pred
        self.season = seasonality(self.freq)
        self.config = config_name(name, self.freq, term)

    @staticmethod
    def _read_table(path: Path) -> pa.Table:
        files = sorted(path.glob("data-*.arrow"))
        assert files, f"No Arrow files in {path}"
        tables = []
        for f in files:
            with pa.memory_map(str(f)) as source:
                tables.append(pa.ipc.open_stream(source).read_all())
        return pa.concat_tables(tables)

    @cached_property
    def series(self) -> list[tuple[str, np.ndarray]]:
        """One-dimensional series of datasets with names."""
        out = []
        ids = self._table.column("item_id").to_pylist()
        targets = self._table.column("target")
        for item_id, target in zip(ids, targets, strict=True):
            values = target.as_py()
            if self.target_dim == 1:
                out.append(
                    (str(item_id), np.asarray(values, dtype=np.float32))
                )
            else:
                for k, row in enumerate(values):
                    out.append(
                        (
                            f"{item_id}_dim{k}",
                            np.asarray(row, dtype=np.float32),
                        )
                    )
        return out

    @cached_property
    def windows(self) -> int:
        """Number of test windows per row."""
        if "m4" in self.name:
            return 1
        targets = self._table.column("target")
        if self.target_dim == 1:
            min_len = min(len(t) for t in targets.to_pylist())
        else:
            min_len = min(len(t[0]) for t in targets.to_pylist())
        w = math.ceil(_TEST_SPLIT * min_len / self.prediction_length)
        return min(max(1, w), _MAX_WINDOWS)

    def test_instances(self) -> Iterator[TestInstance]:
        """Test windows of all rows in order of rows and windows."""
        h, w = self.prediction_length, self.windows
        for item_id, y in self.series:
            for k in range(w):
                cut = len(y) - h * (w - k)
                yield TestInstance(item_id, k, y[:cut], y[cut: cut + h])

    def __len__(self) -> int:
        """Number of test windows on the dataset."""
        return len(self.series) * self.windows


def context_length_distribution(
    root: str | Path,
    cap: int = 512,
    names: tuple[str, ...] = SHORT_CONFIGS,
    term: str = "short",
) -> tuple[np.ndarray, np.ndarray]:
    """Distribution of the lengths of the available history over test windows
    cropped on top of the `cap`.

    The configurations are weighted equally, the windows within the
    configuration are weighted
    equally, so that the length weights repeat the contribution of tasks to the
    composite metric.
    NaN in history is not counted as its absence: length is taken from the
    number of points to the
    window.

    Args:
        root: Dataset snapshot directory.
        cap: The upper limit of length.
        names: Dataset names.
        term: Term.

    Returns:
        tuple[np.ndarray, np.ndarray]: Ascending lengths of `(k,)` and their
        probabilities `(k,)`.
    """
    counts: dict[int, float] = {}
    for name in names:
        ds = GiftDataset(root, name, term)
        lengths = np.array(
            [min(len(inst.history), cap) for inst in ds.test_instances()]
        )
        values, n = np.unique(lengths, return_counts=True)
        for v, c in zip(values, n, strict=True):
            counts[int(v)] = counts.get(int(v), 0.0) + c / len(lengths) / len(
                names
            )
    lengths = np.array(sorted(counts))
    probs = np.array([counts[int(v)] for v in lengths])
    return lengths, probs / probs.sum()


def dataset_properties(root: str | Path) -> dict[str, dict]:
    """Domains and the number of variable datasets from
    `dataset_properties.json`, if the file is in the image; otherwise empty.
    """
    path = Path(root) / "dataset_properties.json"
    return json.loads(path.read_text()) if path.exists() else {}
