# Estimator variants and Spearman correlations with forecast performance

Each cell shows Spearman's ρ between an estimate and a performance metric across generators. Lower metric values are better, so negative ρ associates a higher estimate with better models. Brackets give 95% bootstrap intervals where reported.


## All 11 (n = 11)

| Estimator | CRPS / naive (geometric mean) | CRPS / naive (median) | CRPS rank | MASE / naive (geometric mean) | GIFT-Eval NLL rank |
|---|---|---|---|---|---|
| `A_ema` | +0.00 [-0.89, +0.61] | +0.00 [-0.75, +0.67] | -0.01 [-0.74, +0.67] | -0.35 [-0.94, +0.41] | -0.06 [-0.89, +0.69] |
| `A_raw` | +0.19 [-0.67, +0.70] | +0.21 [-0.63, +0.77] | +0.18 [-0.58, +0.72] | -0.09 [-0.74, +0.59] | -0.11 [-0.77, +0.61] |
| `sum_loss` | -0.29 [-0.84, +0.42] | -0.34 [-0.97, +0.32] | -0.46 [-0.98, +0.18] | -0.21 [-0.80, +0.59] | -0.43 [-0.87, +0.21] |
| `L_fin_ema` | -0.22 [-0.83, +0.49] | -0.28 [-0.97, +0.44] | -0.38 [-0.98, +0.28] | -0.15 [-0.79, +0.66] | -0.37 [-0.87, +0.31] |
| `drop_ema` | -0.03 [-0.93, +0.58] | -0.06 [-0.81, +0.64] | -0.07 [-0.79, +0.63] | -0.35 [-0.93, +0.40] | -0.17 [-0.92, +0.59] |

## Excluding chaotic, waveform, stepfunction (n = 8)

| Estimator | CRPS / naive (geometric mean) | CRPS / naive (median) | CRPS rank | MASE / naive (geometric mean) | GIFT-Eval NLL rank |
|---|---|---|---|---|---|
| `A_ema` | -0.83 [-1.00, -0.24] | -0.69 [-1.00, +0.14] | -0.64 [-1.00, +0.14] | -0.83 [-1.00, -0.32] | -0.33 [-1.00, +0.62] |
| `A_raw` | -0.60 [-1.00, +0.23] | -0.62 [-1.00, +0.22] | -0.57 [-0.97, +0.14] | -0.45 [-0.92, +0.62] | -0.50 [-0.95, +0.34] |
| `sum_loss` | -0.24 [-0.85, +0.62] | -0.12 [-1.00, +0.60] | -0.31 [-1.00, +0.48] | +0.10 [-0.67, +0.92] | -0.33 [-0.85, +0.43] |
| `L_fin_ema` | -0.07 [-0.85, +0.70] | +0.00 [-1.00, +0.67] | -0.14 [-1.00, +0.59] | +0.21 [-0.64, +0.97] | -0.24 [-0.85, +0.60] |
| `drop_ema` | -0.90 [-1.00, -0.53] | -0.83 [-1.00, -0.24] | -0.79 [-1.00, -0.14] | -0.81 [-1.00, -0.20] | -0.55 [-1.00, +0.39] |

## Estimator definitions

- `A_ema` — Σℓ − N·L_fin using the final held-out EMA loss (primary estimate)
- `A_raw` — Σℓ − N·L_fin using the final held-out loss of the raw weights
- `A_train_tail` — Σℓ − N times the mean training loss of the last 2% of windows, without a held-out set
- `A_min_ema` — Σℓ − N times the minimum held-out EMA loss across checkpoints
- `A_pos_ema` — Σ max(ℓ − L_fin_ema, 0), the positive excess only
- `A_first_half_ema` — Area above L_fin_ema over the first half of the windows
- `A_second_half_ema` — Area above L_fin_ema over the second half of the windows
- `A_heldout_curve_ema` — Area above L_fin and under the piecewise-linear held-out EMA loss curve over seven checkpoints
- `A_heldout_curve_raw` — The same held-out curve area for raw weights
- `A_rel_ema` — A_ema / (N·(L_start − L_fin)), the area divided by the rectangle spanning the full loss drop
- `sum_loss` — Σℓ without subtraction (code length)
- `L_fin_ema` — Final held-out EMA loss
- `L_fin_raw` — Final held-out loss of the raw weights
- `L_start` — Mean loss of the first 1% of windows
- `drop_ema` — L_start − L_fin_ema

## Estimate values (areas and total loss in million nats, point estimates of loss in nats per window, A_rel dimensionless)

| Generator | A_ema | A_raw | sum_loss | L_fin_ema | drop_ema |
|---|---|---|---|---|---|
| kernelsynth | 926.1 | 383.2 | -1287.3 | -55.34 | 117.73 |
| ets | 138.6 | 62.8 | 9303.8 | 229.13 | 45.52 |
| waveform | 1061.2 | 568.0 | 402.6 | -16.46 | 136.72 |
| sde | 34.6 | 26.7 | 3374.7 | 83.50 | 18.04 |
| stepfunction | 588.3 | 469.1 | -591.3 | -29.49 | 75.69 |
| tsi | 765.9 | 382.5 | -4268.5 | -125.86 | 108.21 |
| arima | 154.5 | 15.4 | 2923.3 | 69.22 | 36.04 |
| fbm | 40.6 | 26.2 | -4133.6 | -104.36 | 16.73 |
| garch | 100.9 | 30.9 | -1804.6 | -47.64 | 25.85 |
| chaotic | 1956.6 | 815.7 | -4484.9 | -161.04 | 222.69 |
| timesynth | 114.9 | 62.8 | 1482.3 | 34.18 | 27.72 |
