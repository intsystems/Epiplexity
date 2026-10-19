r"""Метрики прогноза по окнам и их сведение по датасету.

По каждому окну сохраняются суммы по наблюдаемым точкам горизонта, из которых любая метрика
датасета восстанавливается точно: в определениях GIFT-Eval (gluonts: среднее по всем точкам
всех окон, взвешенные квантильные потери как отношение сумм) и в нашем масштабированном виде.
Делитель масштабирования — сезонная ошибка ряда: средний модуль разности истории с лагом в
сезонность (лаг 1, если сезонность не меньше длины истории), как `seasonal_error` в gluonts.
Квантильная потеря уровня $q$ для прогноза $\hat y$: $2\,|(y - \hat y)(\mathbf{1}[\hat y \ge y] - q)|$.
CRPS по сэмплам: $\mathbb{E}|X - y| - \tfrac12 \mathbb{E}|X - X'|$ с точной формулой по
упорядоченным сэмплам.
"""

import math

import numpy as np

QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
INTERVALS: tuple[float, ...] = (0.5, 0.8, 0.9)
_MSIS_ALPHA = 0.05


def seasonal_error(history: np.ndarray, season: int) -> tuple[float, float]:
    """Средний модуль и средний квадрат сезонной разности истории; NaN в истории пропускаются."""
    y = history.astype(np.float64)
    lag = season if season < len(y) else 1
    d = y[lag:] - y[:-lag]
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan")
    return float(np.abs(d).mean()), float((d * d).mean())


def crps_from_samples(samples: np.ndarray, label: np.ndarray) -> np.ndarray:
    """CRPS `(h,)` по сэмплам `(n, h)` для правды `(h,)`."""
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
    """Суммы по наблюдаемым точкам одного окна, достаточные для всех метрик.

    Args:
        label: Правда `(h,)`, NaN не учитываются.
        samples: Сэмплы прогноза `(n, h)`; медиана и квантили берутся по ним линейной интерполяцией.
        history: История ряда до окна для сезонной ошибки.
        season: Сезонность.
        nll: Минус лог-плотность правды `(h,)` в натах; `None` — у модели нет плотности.
        context_length: Сколько точек истории видела модель; `None` — не записывается.

    Returns:
        dict[str, float]: Поля `n_valid`, `n_label`, `seasonal_error`, `seasonal_error_sq`, суммы
        ошибок, квантильных потерь, CRPS, покрытий, ширин, интервального счёта и лог-плотности.
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
        "sum_smape": float((2 * np.abs(y - median)[smape_ok] / denom[smape_ok]).sum()),
        "sum_crps": float(crps_from_samples(samples, y)[valid].sum()),
    }
    for q, pred in zip(QUANTILES, quantiles, strict=True):
        out[f"ql_{q}"] = float((2 * np.abs((y - pred) * ((pred >= y) - q)))[valid].sum())
    for c in INTERVALS:
        lo, hi = np.quantile(samples, [(1 - c) / 2, (1 + c) / 2], axis=0)
        out[f"cover_{c}"] = float(((lo <= y) & (y <= hi))[valid].sum())
        out[f"width_{c}"] = float((hi - lo)[valid].sum())
    lo, hi = np.quantile(samples, [_MSIS_ALPHA / 2, 1 - _MSIS_ALPHA / 2], axis=0)
    sis = hi - lo + 2 / _MSIS_ALPHA * (lo - y) * (y < lo) + 2 / _MSIS_ALPHA * (y - hi) * (y > hi)
    out["sum_interval_score"] = float(sis[valid].sum())
    out["sum_nll"] = float(np.where(valid, nll, 0.0).sum()) if nll is not None else float("nan")
    out["context_length"] = float(context_length) if context_length is not None else float("nan")
    return out


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    """Метрики датасета по записям окон: определения GIFT-Eval и масштабированные наши.

    Поля GIFT-Eval повторяют колонки лидерборда (`MASE[0.5]`, `sMAPE[0.5]`, `ND[0.5]`, `MAE[0.5]`,
    `MSE[0.5]`, `MSE[mean]`, `RMSE[mean]`, `NRMSE[mean]`, `MSIS`, `mean_weighted_sum_quantile_loss`)
    с их же исключениями: точки с нулевым знаменателем sMAPE и окна с нулевой сезонной ошибкой в
    MASE и MSIS не входят ни в сумму, ни в счёт (так делает маскированная арифметика gluonts).
    Наши поля считаются по окнам с положительной сезонной ошибкой (`n_excluded` — сколько окон
    выпало): `scaled_crps`, `rmsse`, `coverage_<c>`, `scaled_width_<c>`, `nll_per_point`; MASE
    совпадает с `MASE[0.5]`.

    Args:
        rows: Записи `window_stats` всех окон датасета.

    Returns:
        dict[str, float]: Метрики датасета и счётчики `n_windows`, `n_points`, `n_excluded`.
    """
    n = sum(r["n_valid"] for r in rows)
    abs_err = np.array([r["sum_abs_err_median"] for r in rows])
    sum_abs_y = sum(r["sum_abs_y"] for r in rows)
    keep = [r for r in rows if r["seasonal_error"] > 0]
    m = sum(r["n_valid"] for r in keep)
    out: dict[str, float] = {"n_windows": float(len(rows)), "n_points": n, "n_excluded": float(len(rows) - len(keep))}
    if m > 0:
        out["MASE[0.5]"] = sum(r["sum_abs_err_median"] / r["seasonal_error"] for r in keep) / m
        out["MSIS"] = sum(r["sum_interval_score"] / r["seasonal_error"] for r in keep) / m
    out["MAE[0.5]"] = abs_err.sum() / n
    out["sMAPE[0.5]"] = sum(r["sum_smape"] for r in rows) / sum(r["n_smape"] for r in rows)
    out["ND[0.5]"] = abs_err.sum() / sum_abs_y
    out["MSE[0.5]"] = sum(r["sum_sq_err_median"] for r in rows) / n
    out["MSE[mean]"] = sum(r["sum_sq_err_mean"] for r in rows) / n
    out["RMSE[mean]"] = math.sqrt(out["MSE[mean]"])
    out["NRMSE[mean]"] = out["RMSE[mean]"] / (sum_abs_y / n)
    out["mean_weighted_sum_quantile_loss"] = float(np.mean([sum(r[f"ql_{q}"] for r in rows) / sum_abs_y for q in QUANTILES]))
    if m == 0:
        return out
    out["scaled_crps"] = sum(r["sum_crps"] / r["seasonal_error"] for r in keep) / m
    out["rmsse"] = math.sqrt(sum(r["sum_sq_err_median"] / r["seasonal_error_sq"] for r in keep) / m)
    for c in INTERVALS:
        out[f"coverage_{c}"] = sum(r[f"cover_{c}"] for r in keep) / m
        out[f"scaled_width_{c}"] = sum(r[f"width_{c}"] / r["seasonal_error"] for r in keep) / m
    out["nll_per_point"] = sum(r["sum_nll"] for r in keep) / m
    return out
