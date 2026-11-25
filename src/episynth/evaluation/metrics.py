r"""Metrics of forecast by windows and their reduction by dataset.

For each window, sums are stored at the observable points of the horizon, from
which any dataset
metric is restored precisely: in the definitions GIFT-Eval (gluonts: average at
all points of all
windows, weighted quantile losses as a ratio of sums) and in our scaled form.
The zoom divider is a
seasonal error of the series: the average module of the history difference with
lag in seasonality
(lag) 1if the seasonality is not less than the length of history, as
`seasonal_error` Quantile level
loss :math:`q` for the forecast :math:`\hat y`: :math:`2\,|(y - \hat
y)(\mathbf{1}[\hat y \ge y] -
q)|`. CRPS by sample: :math:`\mathbb{E}|X - y| - \tfrac12 \mathbb{E}|X - X'|`
with an exact formula
for ordered samples.
"""

import math

import numpy as np

QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
INTERVALS: tuple[float, ...] = (0.5, 0.8, 0.9)
_MSIS_ALPHA = 0.05
AGGREGATE_FIELDS: tuple[str, ...] = (
    "n_windows",
    "n_points",
    "n_excluded",
    "MASE[0.5]",
    "MSIS",
    "MAE[0.5]",
    "sMAPE[0.5]",
    "ND[0.5]",
    "MSE[0.5]",
    "MSE[mean]",
    "RMSE[mean]",
    "NRMSE[mean]",
    "mean_weighted_sum_quantile_loss",
    "scaled_crps",
    "rmsse",
    *(
        field
        for c in INTERVALS
        for field in (f"coverage_{c}", f"scaled_width_{c}")
    ),
    "nll_per_point",
)


def seasonal_error(history: np.ndarray, season: int) -> tuple[float, float]:
    """The middle module and the middle square of the seasonal difference of
    history; NaN in history are skipped.
    """
    y = history.astype(np.float64)
    lag = season if season < len(y) else 1
    d = y[lag:] - y[:-lag]
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan")
    return float(np.abs(d).mean()), float((d * d).mean())


def crps_from_samples(samples: np.ndarray, label: np.ndarray) -> np.ndarray:
    """CRPS `(h,)` samples `(n, h)` for the truth of `(h,)`."""
    n = samples.shape[0]
    xs = np.sort(samples, axis=0)
    term1 = np.abs(xs - label[None]).mean(axis=0)
    weights = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    half_spread = (weights * xs).sum(axis=0) / (n * n)
    return term1 - half_spread


def window_stats(
    label: np.ndarray,
    samples: np.ndarray,
    history: np.ndarray,
    season: int,
    nll: np.ndarray | None = None,
    context_length: int | None = None,
) -> dict[str, float]:
    """The sums at the observed points of a single window are sufficient for
    all metrics.

    Args:
        label: `(h,)` and NaN are not counted.
        samples: Samples of the forecast `(n, h)`; the median and quantiles are
            taken from them by linear interpolation.
        history: A row story to a window for seasonal error.
        season: Seasonal.
        nll: Minus the truth log density of the `(h,)` in nats; `None` - the
            model has no density.
        context_length: How many points of history did the model see; the
            `None` does not record.

    Returns:
        dict[str, float]: Fields `n_valid`, `n_label`, `seasonal_error`,
        `seasonal_error_sq`, error
        sums, quantile losses, CRPS, coatings, widths, interval count and log
        density.
    """
    label = label.astype(np.float64)
    valid = np.isfinite(label)
    y = np.where(valid, label, 0.0)
    samples = samples.astype(np.float64)
    quantiles = np.quantile(samples, QUANTILES, axis=0)
    median = quantiles[QUANTILES.index(0.5)]
    mean = samples.mean(axis=0)
    se, se_sq = seasonal_error(history, season)
    denom = np.abs(y) + np.abs(median)
    smape_ok = valid & (denom > 0)
    out: dict[str, float] = {
        "n_valid": float(valid.sum()),
        "n_label": float(len(label)),
        "n_smape": float(smape_ok.sum()),
        "seasonal_error": se,
        "seasonal_error_sq": se_sq,
        "sum_abs_y": float(np.abs(y[valid]).sum()),
        "sum_abs_err_median": float(np.abs(y - median)[valid].sum()),
        "sum_sq_err_median": float(((y - median) ** 2)[valid].sum()),
        "sum_sq_err_mean": float(((y - mean) ** 2)[valid].sum()),
        "sum_smape": float(
            (2 * np.abs(y - median)[smape_ok] / denom[smape_ok]).sum()
        ),
        "sum_crps": float(crps_from_samples(samples, y)[valid].sum()),
    }
    for q, pred in zip(QUANTILES, quantiles, strict=True):
        out[f"ql_{q}"] = float(
            (2 * np.abs((y - pred) * ((pred >= y) - q)))[valid].sum()
        )
    for c in INTERVALS:
        lo, hi = np.quantile(samples, [(1 - c) / 2, (1 + c) / 2], axis=0)
        out[f"cover_{c}"] = float(((lo <= y) & (y <= hi))[valid].sum())
        out[f"width_{c}"] = float((hi - lo)[valid].sum())
    lo, hi = np.quantile(
        samples, [_MSIS_ALPHA / 2, 1 - _MSIS_ALPHA / 2], axis=0
    )
    sis = (
        hi
        - lo
        + 2 / _MSIS_ALPHA * (lo - y) * (y < lo)
        + 2 / _MSIS_ALPHA * (y - hi) * (y > hi)
    )
    out["sum_interval_score"] = float(sis[valid].sum())
    out["sum_nll"] = (
        float(np.where(valid, nll, 0.0).sum())
        if nll is not None
        else float("nan")
    )
    out["context_length"] = (
        float(context_length) if context_length is not None else float("nan")
    )
    return out


def _divide(numerator: float, denominator: float) -> float:
    """Undefined metrics retain their column and are reported as NaN."""
    return float(numerator / denominator) if denominator > 0 else float("nan")


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    """Dataset metrics by window records: definitions GIFT-Eval and scaled
    ours.

    The fields of GIFT-Eval repeat the leaderboard columns (`MASE[0.5]`,
    `sMAPE[0.5]`, `ND[0.5]`, `MAE[0.5]`).
    `MSE[0.5]`, `MSE[mean]`, `RMSE[mean]`, `NRMSE[mean]`, `MSIS`,
    `mean_weighted_sum_quantile_loss`)
    with their own exceptions: points with zero denominator sMAPE and windows
    with zero seasonal
    error MASE and MSIS It's not in the sum or in the account (as the masked
    arithmetic of gluonts
    does). Our fields are counted on the windows with a positive seasonal
    error.`n_excluded` - How
    many windows fell out? `scaled_crps`, `rmsse`, `coverage_<c>`,
    `scaled_width_<c>`,
    `nll_per_point`; MASE coincidental `MASE[0.5]`.

    All fields are present even when no observations remain. Metrics with a
    zero
    denominator or no eligible observations are NaN.

    Args:
        rows: `window_stats` records of all dataset windows.

    Returns:
        dict[str, float]: Dataset metrics and counters `n_windows`, `n_points`,
        `n_excluded`.
    """
    n = sum(r["n_valid"] for r in rows)
    abs_err = np.array([r["sum_abs_err_median"] for r in rows])
    sum_abs_y = sum(r["sum_abs_y"] for r in rows)
    keep = [r for r in rows if r["seasonal_error"] > 0]
    m = sum(r["n_valid"] for r in keep)
    out: dict[str, float] = {
        "n_windows": float(len(rows)),
        "n_points": n,
        "n_excluded": float(len(rows) - len(keep)),
    }
    out["MASE[0.5]"] = _divide(
        sum(r["sum_abs_err_median"] / r["seasonal_error"] for r in keep), m
    )
    out["MSIS"] = _divide(
        sum(r["sum_interval_score"] / r["seasonal_error"] for r in keep), m
    )
    out["MAE[0.5]"] = _divide(abs_err.sum(), n)
    out["sMAPE[0.5]"] = _divide(
        sum(r["sum_smape"] for r in rows), sum(r["n_smape"] for r in rows)
    )
    out["ND[0.5]"] = _divide(abs_err.sum(), sum_abs_y)
    out["MSE[0.5]"] = _divide(sum(r["sum_sq_err_median"] for r in rows), n)
    out["MSE[mean]"] = _divide(sum(r["sum_sq_err_mean"] for r in rows), n)
    out["RMSE[mean]"] = math.sqrt(out["MSE[mean]"])
    out["NRMSE[mean]"] = _divide(out["RMSE[mean]"], _divide(sum_abs_y, n))
    out["mean_weighted_sum_quantile_loss"] = float(
        np.mean(
            [
                _divide(sum(r[f"ql_{q}"] for r in rows), sum_abs_y)
                for q in QUANTILES
            ]
        )
    )
    out["scaled_crps"] = _divide(
        sum(r["sum_crps"] / r["seasonal_error"] for r in keep), m
    )
    out["rmsse"] = math.sqrt(
        _divide(
            sum(
                _divide(r["sum_sq_err_median"], r["seasonal_error_sq"])
                for r in keep
            ),
            m,
        )
    )
    for c in INTERVALS:
        out[f"coverage_{c}"] = _divide(sum(r[f"cover_{c}"] for r in keep), m)
        out[f"scaled_width_{c}"] = _divide(
            sum(r[f"width_{c}"] / r["seasonal_error"] for r in keep), m
        )
    out["nll_per_point"] = _divide(sum(r["sum_nll"] for r in keep), m)
    return out
