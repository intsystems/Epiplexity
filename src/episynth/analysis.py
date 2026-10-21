r"""Анализ прогонов: площадь над финальным лоссом с её ошибкой, сводки качества, ранговые корреляции.

Площадь прогона $A = \sum_{i=1}^{N} \ell_i - N L_{\mathrm{fin}}$ в натах: $\ell_i$ — лосс $i$-го
окна до шага по нему (`window_loss.f32`), $L_{\mathrm{fin}}$ — средний лосс окна на отложенном
наборе у последнего чекпойнта для выбранного варианта весов (`heldout_<вариант>_1.f32`, точность
обучения). Ошибка площади от конечного отложенного набора равна $N$, умноженному на стандартную
ошибку $L_{\mathrm{fin}}$ по окнам набора; обучающие лоссы входят в площадь как есть, это
свойство прогона, а не оценка. Кривая площади по префиксам $A(n) = \sum_{i \le n} \ell_i - n L_{\mathrm{fin}}$
нужна для предсказания по началу обучения.

Сводки качества по задачам бенчмарка из `summary.csv` прогонов и сезонного наива: средний ранг
среди моделей, среднее геометрическое, арифметическое и медиана отношений к наиву, доля задач
лучше наива. Корреляция Спирмена предсказателя с качеством по генераторам с бутстреп-интервалом
по генераторам.
"""

import csv
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


@dataclass
class RunArea:
    """Площадь прогона и её составляющие.

    Attributes:
        n_windows: Число обучающих окон $N$.
        sum_loss: Сумма лоссов окон до шага, нат.
        l_fin: Финальный лосс, нат на окно.
        l_fin_se: Стандартная ошибка финального лосса по окнам отложенного набора.
        n_heldout: Размер отложенного набора.
        area: Площадь $A$, нат.
        area_se: Ошибка площади от конечного отложенного набора, нат.
        variant: Вариант весов финального лосса: `ema` или `raw`.
    """

    n_windows: int
    sum_loss: float
    l_fin: float
    l_fin_se: float
    n_heldout: int
    area: float
    area_se: float
    variant: str


def run_area(run_dir: str | Path, variant: str = "ema") -> RunArea:
    """Площадь прогона по его каталогу.

    Args:
        run_dir: Каталог прогона с `window_loss.f32` и `heldout_<variant>_1.f32`.
        variant: Вариант весов финального лосса: `ema` или `raw`.

    Returns:
        RunArea: Площадь и составляющие.
    """
    run_dir = Path(run_dir)
    losses = np.fromfile(run_dir / "window_loss.f32", dtype=np.float32).astype(np.float64)
    held = np.fromfile(run_dir / f"heldout_{variant}_1.f32", dtype=np.float32).astype(np.float64)
    n, m = len(losses), len(held)
    l_fin = float(held.mean())
    se = float(held.std(ddof=1) / math.sqrt(m))
    return RunArea(n, float(losses.sum()), l_fin, se, m, float(losses.sum() - n * l_fin), n * se, variant)


def area_curve(run_dir: str | Path, variant: str = "ema", points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Кривая площади по префиксам обучения: номера окон `(points,)` и $A(n)$ в тех же точках."""
    run_dir = Path(run_dir)
    losses = np.fromfile(run_dir / "window_loss.f32", dtype=np.float32).astype(np.float64)
    l_fin = float(np.fromfile(run_dir / f"heldout_{variant}_1.f32", dtype=np.float32).astype(np.float64).mean())
    ns = np.unique(np.linspace(1, len(losses), points).astype(np.int64))
    cum = np.cumsum(losses)
    return ns, cum[ns - 1] - ns * l_fin


def load_summary(path: str | Path) -> dict[str, dict[str, float]]:
    """Метрики `summary.csv` по конфигурациям: `{config: {метрика: значение}}`."""
    out: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["config"]] = {k: float(v) for k, v in row.items() if k not in ("config", "model") and v != ""}
    return out


def quality_table(summaries: Mapping[str, str | Path], naive: str | Path, metric: str = "scaled_crps") -> dict[str, dict[str, float]]:
    """Сводки качества моделей по задачам бенчмарка относительно друг друга и сезонного наива.

    Args:
        summaries: `summary.csv` прогонов по именам моделей.
        naive: `summary.csv` сезонного наива.
        metric: Колонка метрики; меньше — лучше.

    Returns:
        dict[str, dict[str, float]]: По моделям `mean_rank` (средний ранг среди моделей по задачам,
        при равенстве средний), `geo_mean_ratio`, `mean_ratio`, `median_ratio` (отношения к наиву по
        задачам), `share_better` (доля задач, где лучше наива), `n_configs`.
    """
    tables = {name: load_summary(p) for name, p in summaries.items()}
    base = load_summary(naive)
    configs = sorted(set.intersection(*(set(t) for t in tables.values())) & set(base))
    names = list(tables)
    values = np.array([[tables[n][c][metric] for n in names] for c in configs])
    ranks = np.vstack([rankdata(row) for row in values])
    out = {}
    for j, name in enumerate(names):
        ratio = np.array([tables[name][c][metric] / base[c][metric] for c in configs])
        out[name] = {
            "mean_rank": float(ranks[:, j].mean()),
            "geo_mean_ratio": float(np.exp(np.log(ratio).mean())),
            "mean_ratio": float(ratio.mean()),
            "median_ratio": float(np.median(ratio)),
            "share_better": float((ratio < 1).mean()),
            "n_configs": float(len(configs)),
        }
    return out


def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Ранговая корреляция Спирмена с бутстреп-интервалом по объектам (процентильный метод).

    Args:
        x: Предсказатель по объектам.
        y: Качество по объектам.
        n_boot: Число бутстреп-выборок.
        seed: Сид бутстрепа.
        level: Уровень интервала.

    Returns:
        tuple[float, float, float]: Корреляция и границы интервала.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        boots.append(spearmanr(x[idx], y[idx]).statistic)
    lo, hi = np.nanpercentile(boots, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return rho, float(lo), float(hi)


def correlation_table(predictors: Mapping[str, Mapping[str, float]], quality: Mapping[str, float], **kwargs) -> dict[str, dict[str, float]]:
    """Корреляции Спирмена каждого предсказателя с качеством по общим объектам: `rho`, `lo`, `hi`, `n`."""
    out = {}
    for name, pred in predictors.items():
        keys = sorted(set(pred) & set(quality))
        rho, lo, hi = spearman_ci(np.array([pred[k] for k in keys]), np.array([quality[k] for k in keys]), **kwargs)
        out[name] = {"rho": rho, "lo": lo, "hi": hi, "n": float(len(keys))}
    return out
