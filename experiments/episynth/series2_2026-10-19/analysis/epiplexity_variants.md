# Estimator variants and Spearman correlations with forecast performance

Each cell shows Spearman's ρ between an estimate and a performance metric across generators. Lower metric values are better, so negative ρ associates a higher estimate with better models. Brackets give 95% bootstrap intervals where reported.


## All 11 (n = 11)

| Estimator | CRPS / naive (geometric mean) | CRPS / naive (median) | CRPS rank | MASE / naive (geometric mean) | GIFT-Eval NLL rank |
|---|---|---|---|---|---|
| `A_ema` | +0.00 [-0.89, +0.61] | +0.00 | -0.01 | -0.35 | -0.06 [-0.89, +0.69] |
| `A_raw` | +0.19 [-0.67, +0.70] | +0.21 | +0.18 | -0.09 | -0.11 [-0.77, +0.61] |
| `A_train_tail` | +0.03 [-0.86, +0.61] | +0.00 | +0.00 | -0.25 | -0.09 [-0.86, +0.64] |
| `A_min_ema` | +0.00 [-0.89, +0.61] | +0.00 | -0.01 | -0.35 | -0.06 [-0.89, +0.69] |
| `A_pos_ema` | -0.33 [-0.87, +0.40] | +0.01 | -0.02 | -0.23 | +0.18 [-0.59, +0.75] |
| `A_first_half_ema` | -0.01 [-0.91, +0.61] | -0.05 | -0.05 | -0.32 | -0.14 [-0.89, +0.62] |
| `A_second_half_ema` | +0.03 [-0.85, +0.61] | -0.01 | +0.02 | -0.35 | -0.08 [-0.92, +0.68] |
| `A_heldout_curve_ema` | -0.03 [-0.93, +0.58] | -0.06 | -0.07 | -0.35 | -0.17 [-0.92, +0.59] |
| `A_heldout_curve_raw` | +0.10 [-0.74, +0.66] | +0.16 | +0.15 | -0.16 | +0.00 [-0.72, +0.72] |
| `A_rel_ema` | -0.03 [-0.79, +0.59] | +0.15 | +0.15 | -0.20 | +0.03 [-0.83, +0.75] |
| `sum_loss` | -0.29 [-0.84, +0.42] | -0.34 | -0.46 | -0.21 | -0.43 [-0.87, +0.21] |
| `L_fin_ema` | -0.22 [-0.83, +0.49] | -0.28 | -0.38 | -0.15 | -0.37 [-0.87, +0.31] |
| `L_fin_raw` | -0.29 [-0.84, +0.42] | -0.34 | -0.46 | -0.21 | -0.43 [-0.87, +0.21] |
| `L_start` | -0.22 [-0.79, +0.52] | -0.43 | -0.55 | -0.40 | -0.50 [-0.80, +0.08] |
| `drop_ema` | -0.03 [-0.93, +0.58] | -0.06 | -0.07 | -0.35 | -0.17 [-0.92, +0.59] |

## Excluding Chaotic, StepFunction (n = 9)

| Estimator | CRPS / naive (geometric mean) | CRPS / naive (median) | CRPS rank | MASE / naive (geometric mean) | GIFT-Eval NLL rank |
|---|---|---|---|---|---|
| `A_ema` | -0.28 [-0.98, +0.53] | -0.48 | -0.45 | -0.80 | -0.38 [-0.95, +0.46] |
| `A_raw` | -0.12 [-0.89, +0.67] | -0.38 | -0.35 | -0.53 | -0.53 [-0.95, +0.22] |
| `A_train_tail` | -0.25 [-1.00, +0.55] | -0.48 | -0.43 | -0.65 | -0.43 [-0.95, +0.39] |
| `A_min_ema` | -0.28 [-0.98, +0.53] | -0.48 | -0.45 | -0.80 | -0.38 [-0.95, +0.46] |
| `A_pos_ema` | -0.63 [-1.00, -0.11] | -0.57 | -0.55 | -0.68 | -0.07 [-0.74, +0.69] |
| `A_first_half_ema` | -0.30 [-1.00, +0.51] | -0.57 | -0.52 | -0.75 | -0.50 [-0.95, +0.30] |
| `A_second_half_ema` | -0.23 [-0.95, +0.56] | -0.50 | -0.42 | -0.82 | -0.42 [-1.00, +0.43] |
| `A_heldout_curve_ema` | -0.33 [-1.00, +0.48] | -0.60 | -0.57 | -0.80 | -0.55 [-0.98, +0.23] |
| `A_heldout_curve_raw` | -0.22 [-0.95, +0.56] | -0.47 | -0.40 | -0.60 | -0.37 [-0.89, +0.43] |
| `A_rel_ema` | -0.27 [-0.98, +0.63] | -0.42 | -0.33 | -0.65 | -0.32 [-0.98, +0.63] |
| `sum_loss` | -0.18 [-0.84, +0.53] | -0.13 | -0.30 | +0.07 | -0.27 [-0.74, +0.47] |
| `L_fin_ema` | -0.07 [-0.84, +0.62] | -0.03 | -0.17 | +0.17 | -0.18 [-0.74, +0.69] |
| `L_fin_raw` | -0.18 [-0.84, +0.53] | -0.13 | -0.30 | +0.07 | -0.27 [-0.74, +0.47] |
| `L_start` | -0.23 [-0.89, +0.58] | -0.42 | -0.53 | -0.42 | -0.52 [-0.89, +0.26] |
| `drop_ema` | -0.33 [-1.00, +0.48] | -0.60 | -0.57 | -0.80 | -0.55 [-0.98, +0.23] |

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

| Generator | A_ema | A_raw | A_train_tail | A_min_ema | A_pos_ema | A_first_half_ema | A_second_half_ema | A_heldout_curve_ema | A_heldout_curve_raw | A_rel_ema | sum_loss | L_fin_ema | L_fin_raw | L_start | drop_ema |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kernelsynth | 926.1 | 383.2 | 386.2 | 926.1 | 2381.8 | 620.2 | 305.9 | 563.6 | 371.2 | 0.197 | -1287.3 | -55.34 | -41.76 | 62.39 | 117.73 |
| ets | 138.6 | 62.8 | 63.9 | 138.6 | 3209.9 | 99.3 | 39.3 | 122.5 | 67.6 | 0.076 | 9303.8 | 229.13 | 231.03 | 274.65 | 45.52 |
| waveform | 1061.2 | 568.0 | 538.4 | 1061.2 | 1740.7 | 745.0 | 316.1 | 764.6 | 583.2 | 0.194 | 402.6 | -16.46 | -4.14 | 120.25 | 136.72 |
| sde | 34.6 | 26.7 | 30.0 | 34.6 | 837.6 | 28.6 | 6.1 | 42.3 | 24.2 | 0.048 | 3374.7 | 83.50 | 83.70 | 101.55 | 18.04 |
| stepfunction | 588.3 | 469.1 | 225.4 | 588.3 | 2635.1 | 393.9 | 194.4 | 350.5 | 420.2 | 0.194 | -591.3 | -29.49 | -26.51 | 46.20 | 75.69 |
| tsi | 765.9 | 382.5 | 367.6 | 765.9 | 2993.6 | 530.9 | 235.0 | 547.5 | 402.7 | 0.177 | -4268.5 | -125.86 | -116.27 | -17.65 | 108.21 |
| arima | 154.5 | 15.4 | 74.8 | 154.5 | 2549.9 | 105.7 | 48.8 | 109.1 | 96.4 | 0.107 | 2923.3 | 69.22 | 72.70 | 105.26 | 36.04 |
| fbm | 40.6 | 26.2 | 9.8 | 40.6 | 2020.3 | 28.0 | 12.6 | 30.5 | 17.5 | 0.061 | -4133.6 | -104.36 | -104.00 | -87.63 | 16.73 |
| garch | 100.9 | 30.9 | 52.2 | 100.9 | 1768.4 | 69.1 | 31.8 | 81.8 | 29.1 | 0.098 | -1804.6 | -47.64 | -45.89 | -21.79 | 25.85 |
| chaotic | 1956.6 | 815.7 | 786.5 | 1956.6 | 3737.6 | 1318.6 | 638.0 | 1203.9 | 774.5 | 0.220 | -4484.9 | -161.04 | -132.52 | 61.65 | 222.69 |
| timesynth | 114.9 | 62.8 | 68.7 | 114.9 | 829.5 | 85.5 | 29.4 | 93.4 | 68.0 | 0.104 | 1482.3 | 34.18 | 35.49 | 61.90 | 27.72 |
