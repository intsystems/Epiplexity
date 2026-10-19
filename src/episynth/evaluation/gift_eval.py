"""Данные GIFT-Eval (Aksu и др., 2024, arXiv:2410.10393): датасеты, горизонты, тестовые окна.

Повторяет `gift_eval.data.Dataset` из github.com/SalesforceAIResearch/gift-eval (Apache-2.0) без
gluonts и `datasets`: снимок репозитория `Salesforce/GiftEval` читается напрямую из arrow-файлов.
Горизонт задаётся частотой и термом, число тестовых окон — десятой долей самого короткого ряда
датасета, но не больше 20 (у M4 одно окно); окна идут подряд с шагом в горизонт до конца ряда,
история окна — весь ряд до него. Многомерные датасеты разворачиваются в одномерные ряды с
суффиксом `_dim<k>`. Сезонность по частоте — как `gluonts.time_feature.get_seasonality`.
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
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H "
    "electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D "
    "us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T "
    "SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H".split()
)
"""Имена датасетов краткосрочной части бенчмарка (терм short), 55 штук."""

_PRETTY = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
_TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}
_PRED_LENGTH = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
_M4_PRED_LENGTH = {"A": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
_SEASONALITY = {"S": 3600, "T": 1440, "H": 24, "D": 1, "W": 1, "M": 12, "B": 5, "Q": 4, "A": 1}
_BASE_ALIASES = {
    "min": "T", "T": "T", "H": "H", "h": "H", "D": "D", "W": "W", "M": "M", "ME": "M", "MS": "M",
    "Q": "Q", "QE": "Q", "QS": "Q", "A": "A", "Y": "A", "YE": "A", "YS": "A", "AS": "A", "S": "S", "s": "S", "B": "B",
}
_TEST_SPLIT = 0.1
_MAX_WINDOWS = 20


def parse_freq(freq: str) -> tuple[str, int]:
    """Базовая частота из {S, T, H, D, W, M, Q, A, B} и множитель, например `"15T"` даёт `("T", 15)`."""
    m = re.fullmatch(r"(\d*)([A-Za-z]+)(?:-\w+)?", freq)
    assert m, freq
    return _BASE_ALIASES[m.group(2)], int(m.group(1) or 1)


def seasonality(freq: str) -> int:
    """Сезонность частоты по правилу gluonts: базовая сезонность, делённая на множитель, иначе 1."""
    base, n = parse_freq(freq)
    q, r = divmod(_SEASONALITY.get(base, 1), n)
    return q if r == 0 else 1


def config_name(name: str, freq: str, term: str) -> str:
    """Имя конфигурации в таблицах лидерборда вида `saugeen/D/short`: частота без якоря (`A-DEC` даёт `A`)."""
    key = name.split("/")[0].lower()
    key = _PRETTY.get(key, key)
    base, n = parse_freq(name.split("/")[1] if "/" in name else freq)
    return f"{key}/{n if n > 1 else ''}{base}/{term}"


@dataclass
class TestInstance:
    """Одно тестовое окно: история до начала окна и правда.

    Attributes:
        item_id: Имя ряда, у многомерных датасетов с суффиксом `_dim<k>`.
        window: Номер окна в ряде от 0.
        history: Весь ряд до окна `(T,)` float32, с возможными NaN.
        label: Правда `(h,)` float32, с возможными NaN.
    """

    item_id: str
    window: int
    history: np.ndarray
    label: np.ndarray


class GiftDataset:
    """Датасет GIFT-Eval из локального снимка репозитория `Salesforce/GiftEval`.

    Args:
        root: Каталог снимка.
        name: Имя датасета, как в `SHORT_CONFIGS`, например `"electricity/15T"`.
        term: Терм из {"short", "medium", "long"}.

    Attributes:
        freq: Строка частоты датасета.
        target_dim: Число переменных исходного ряда.
        prediction_length: Горизонт.
        windows: Число тестовых окон на ряд.
        season: Сезонность по частоте.
        config: Имя конфигурации в таблицах лидерборда.
    """

    def __init__(self, root: str | Path, name: str, term: str = "short") -> None:
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
        assert files, f"нет arrow-файлов в {path}"
        tables = []
        for f in files:
            with pa.memory_map(str(f)) as source:
                tables.append(pa.ipc.open_stream(source).read_all())
        return pa.concat_tables(tables)

    @cached_property
    def series(self) -> list[tuple[str, np.ndarray]]:
        """Одномерные ряды датасета с именами."""
        out = []
        ids = self._table.column("item_id").to_pylist()
        targets = self._table.column("target")
        for item_id, target in zip(ids, targets, strict=True):
            values = target.as_py()
            if self.target_dim == 1:
                out.append((str(item_id), np.asarray(values, dtype=np.float32)))
            else:
                for k, row in enumerate(values):
                    out.append((f"{item_id}_dim{k}", np.asarray(row, dtype=np.float32)))
        return out

    @cached_property
    def windows(self) -> int:
        """Число тестовых окон на ряд."""
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
        """Тестовые окна всех рядов по порядку рядов и окон."""
        h, w = self.prediction_length, self.windows
        for item_id, y in self.series:
            for k in range(w):
                cut = len(y) - h * (w - k)
                yield TestInstance(item_id, k, y[:cut], y[cut : cut + h])

    def __len__(self) -> int:
        """Число тестовых окон датасета."""
        return len(self.series) * self.windows


def context_length_distribution(
    root: str | Path, cap: int = 512, names: tuple[str, ...] = SHORT_CONFIGS, term: str = "short"
) -> tuple[np.ndarray, np.ndarray]:
    """Распределение длин доступной истории по тестовым окнам, обрезанных сверху на `cap`.

    Конфигурации взвешены равновероятно, окна внутри конфигурации тоже, так что веса длин
    повторяют вклад задач в сводную метрику. NaN в истории не учитываются как её отсутствие:
    длина берётся по числу точек до окна.

    Args:
        root: Каталог снимка.
        cap: Верхняя граница длины.
        names: Имена датасетов.
        term: Терм.

    Returns:
        tuple[np.ndarray, np.ndarray]: Длины `(k,)` по возрастанию и их вероятности `(k,)`.
    """
    counts: dict[int, float] = {}
    for name in names:
        ds = GiftDataset(root, name, term)
        lengths = np.array([min(len(inst.history), cap) for inst in ds.test_instances()])
        values, n = np.unique(lengths, return_counts=True)
        for v, c in zip(values, n, strict=True):
            counts[int(v)] = counts.get(int(v), 0.0) + c / len(lengths) / len(names)
    lengths = np.array(sorted(counts))
    probs = np.array([counts[int(v)] for v in lengths])
    return lengths, probs / probs.sum()


def dataset_properties(root: str | Path) -> dict[str, dict]:
    """Домены и число переменных датасетов из `dataset_properties.json`, если файл лежит в снимке; иначе пусто."""
    path = Path(root) / "dataset_properties.json"
    return json.loads(path.read_text()) if path.exists() else {}
