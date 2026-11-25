r"""Analyze training runs, summarize forecasts, and compute rank correlations.

The area in nats is :math:`A = \sum_{i=1}^{N} \ell_i - N L_{\mathrm{fin}}`.
Here :math:`\ell_i` is the pre-update loss of window :math:`i`
(`window_loss.f32`),
and :math:`L_{\mathrm{fin}}` is the mean held-out loss of the final checkpoint
for the selected weight variant (`heldout_<variant>_1.f32`, at training
precision).
The area's standard error is :math:`N` times the standard error of the held-out
mean. The logged training losses are treated as fixed observations from this
run.
The prefix area :math:`A(n) = \sum_{i \le n} \ell_i - n L_{\mathrm{fin}}`
supports
prediction from early training.

Benchmark summaries compare model `summary.csv` files with seasonal naive. They
report mean model ranks, geometric and arithmetic mean ratios, median ratios,
and the share of tasks beating the baseline. Spearman correlations across
generators use a generator-level bootstrap interval.
"""

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr


@dataclass
class RunArea:
    """Excess loss area and its components.

    Attributes:
        n_windows: Number of training windows :math:`N`.
        sum_loss: Sum of pre-update window losses in nats.
        l_fin: Final held-out loss in nats per window.
        l_fin_se: Standard error of the final held-out mean.
        n_heldout: Number of held-out windows.
        area: Area :math:`A` in nats.
        area_se: Standard error of the area due to the finite held-out set, in
            nats.
        variant: Final weight variant, `ema` or `raw`.
    """

    n_windows: int
    sum_loss: float
    l_fin: float
    l_fin_se: float
    n_heldout: int
    area: float
    area_se: float
    variant: str


def _read_losses(
    path: Path, expected: int | None = None, minimum: int = 1
) -> np.ndarray:
    """Read a float32 vector, checking alignment, count and finite values."""
    if not path.is_file():
        raise ValueError(f"missing loss artifact: {path}")
    if path.stat().st_size % np.dtype(np.float32).itemsize:
        raise ValueError(f"loss artifact is not aligned to float32: {path}")
    values = np.fromfile(path, dtype=np.float32).astype(np.float64)
    if len(values) < minimum:
        raise ValueError(f"at least {minimum} loss values required: {path}")
    if expected is not None and len(values) != expected:
        raise ValueError(
            f"expected {expected} losses, found {len(values)}: {path}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite losses: {path}")
    return values


def _run_losses(run_dir: Path, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Check Trainer completion when metadata exists; otherwise validate
    vectors.
    """
    config_path = run_dir / "config.json"
    n = m = None
    if config_path.exists():
        config = json.loads(config_path.read_text())["train"]
        n, m, batch = (
            config["n_windows"],
            config["heldout_windows"],
            config["batch_size"],
        )
        if (
            any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in (n, m, batch)
            )
            or n <= 0
            or m < 2
            or batch <= 0
            or n % batch
        ):
            raise ValueError(f"invalid training counts: {config_path}")
        # Float32 logs are preallocated before training. Their size alone
        # cannot
        # distinguish an interrupted run from a completed one.
        heldout_path = run_dir / "heldout.jsonl"
        try:
            final = json.loads(
                heldout_path.read_text().rstrip().rsplit("\n", 1)[-1]
            )
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise ValueError(
                f"missing or incomplete final heldout record: {heldout_path}"
            ) from error
        if (
            not isinstance(final, dict)
            or final.get("step") != n // batch
            or final.get("fraction") != 1
        ):
            raise ValueError(
                f"missing or incomplete final heldout record: {heldout_path}"
            )
    elif any(
        (run_dir / name).exists() for name in ("last.pt", "heldout.jsonl")
    ):
        raise ValueError(f"missing Trainer configuration: {config_path}")
    return (
        _read_losses(run_dir / "window_loss.f32", expected=n),
        _read_losses(
            run_dir / f"heldout_{variant}_1.f32", expected=m, minimum=2
        ),
    )


def run_area(run_dir: str | Path, variant: str = "ema") -> RunArea:
    """Compute the excess area from a run directory.

    Trainer runs require configured loss counts and a final heldout.jsonl
    record.
    Historical directories with only loss files receive basic vector
    validation;
    without run metadata, completeness of those files cannot be established.

    Args:
        run_dir: Run directory containing `window_loss.f32` and
            `heldout_<variant>_1.f32`.
        variant: Final weight variant: `ema` or `raw`.

    Returns:
        RunArea: Area and components.
    """
    losses, held = _run_losses(Path(run_dir), variant)
    n, m = len(losses), len(held)
    l_fin = float(held.mean())
    se = float(held.std(ddof=1) / math.sqrt(m))
    return RunArea(
        n,
        float(losses.sum()),
        l_fin,
        se,
        m,
        float(losses.sum() - n * l_fin),
        n * se,
        variant,
    )


def area_curve(
    run_dir: str | Path, variant: str = "ema", points: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Curve area by training prefixes: window numbers `(points,)` and
    :math:`A(n)`.

    Uses the same artifact and completion checks as :func:`run_area`.
    """
    losses, held = _run_losses(Path(run_dir), variant)
    l_fin = float(held.mean())
    ns = np.unique(np.linspace(1, len(losses), points).astype(np.int64))
    cum = np.cumsum(losses)
    return ns, cum[ns - 1] - ns * l_fin


def load_summary(path: str | Path) -> dict[str, dict[str, float]]:
    """Read `summary.csv` as a mapping from configurations to metric values."""
    out: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[row["config"]] = {
                k: float(v)
                for k, v in row.items()
                if k not in ("config", "model") and v != ""
            }
    return out


def quality_table(
    summaries: Mapping[str, str | Path],
    naive: str | Path,
    metric: str = "scaled_crps",
) -> dict[str, dict[str, float]]:
    """Summary of the quality of models on benchmark problems relative to each
    other and seasonal naive.

    Args:
        summaries: `summary.csv` runs by model names.
        naive: `summary.csv` seasonal naive.
        metric: Metric column; less is better.

    Returns:
        dict[str, dict[str, float]: `mean_rank` (average rank among the models
        by task, with
        equality average), `geo_mean_ratio`, `mean_ratio`, `median_ratio`
        (relationships to naive by
        task), `share_better` (share of tasks where naive is better),
        `n_configs`.
    """
    tables = {name: load_summary(p) for name, p in summaries.items()}
    base = load_summary(naive)
    configs = sorted(
        set.intersection(*(set(t) for t in tables.values())) & set(base)
    )
    names = list(tables)
    values = np.array([[tables[n][c][metric] for n in names] for c in configs])
    ranks = np.vstack([rankdata(row) for row in values])
    out = {}
    for j, name in enumerate(names):
        ratio = np.array(
            [tables[name][c][metric] / base[c][metric] for c in configs]
        )
        out[name] = {
            "mean_rank": float(ranks[:, j].mean()),
            "geo_mean_ratio": float(np.exp(np.log(ratio).mean())),
            "mean_ratio": float(ratio.mean()),
            "median_ratio": float(np.median(ratio)),
            "share_better": float((ratio < 1).mean()),
            "n_configs": float(len(configs)),
        }
    return out


def spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Rank correlation of Spearman with the bootstrap interval by objects
    (percentile method).

    Args:
        x: Object fortune teller.
        y: Quality by facility.
        n_boot: Number of bootstrap samples.
        seed: Bootstrap seed.
        level: Interval level.

    Returns:
        tuple[float, float, float]: Correlation and interval boundaries.
        Undefined values
        (inadequate, constant, or infinite inputs) return as NaN. If no
        bootstrap sample determines
        correlation, the boundaries are NaN.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError(
            "Spearman inputs must be one-dimensional arrays of equal length"
        )
    if (
        not isinstance(n_boot, (int, np.integer))
        or n_boot <= 0
        or not 0 < level < 1
    ):
        raise ValueError(
            "n_boot must be positive and level must lie in (0, 1)"
        )
    if (len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all()
            or len(np.unique(x)) < 2 or len(np.unique(y)) < 2):
        return float("nan"), float("nan"), float("nan")
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        boots.append(spearmanr(x[idx], y[idx]).statistic)
    if not boots:
        return rho, float("nan"), float("nan")
    lo, hi = np.nanpercentile(
        boots, [100 * (1 - level) / 2, 100 * (1 + level) / 2]
    )
    return rho, float(lo), float(hi)


def correlation_table(
    predictors: Mapping[str, Mapping[str, float]],
    quality: Mapping[str, float],
    **kwargs: int | float,
) -> dict[str, dict[str, float]]:
    """Correlations of the Spearman of each soothsayer with quality over common
    objects.

    Args:
        predictors: Values of soothsayers by object names.
        quality: Quality by the same object names.
        **kwargs: Parameters :func:`spearman_ci`: ``n_boot`` and ``seed``
            (int),
            ``level`` (float) Transmitted unchanged.

    Returns:
        Dictionary of predictors with fields ``rho``, ``lo``, ``hi`` and ``n``.
    """
    out = {}
    for name, pred in predictors.items():
        keys = sorted(set(pred) & set(quality))
        rho, lo, hi = spearman_ci(
            np.array([pred[k] for k in keys]),
            np.array([quality[k] for k in keys]),
            **kwargs,
        )
        out[name] = {"rho": rho, "lo": lo, "hi": hi, "n": float(len(keys))}
    return out
