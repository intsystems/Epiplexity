# Estimator variants and Spearman correlations with forecast performance

Each cell shows Spearman's ρ between an estimate and a performance metric across generators. Lower metric values are better, so negative ρ associates a higher estimate with better models. Brackets give 95% bootstrap intervals where reported.


## All 11 (n = 11)

| Estimator | CRPS / naive (geometric mean) | CRPS / naive (median) | CRPS rank | MASE / naive (geometric mean) | GIFT-Eval NLL rank |
|---|---|---|---|---|---|
| `A_ema` | -0.18 [-0.82, +0.49] | -0.15 | +0.05 | -0.27 | -0.08 [-0.81, +0.63] |
| `A_raw` | -0.15 [-0.75, +0.52] | -0.10 | +0.09 | -0.23 | -0.05 [-0.76, +0.64] |
| `A_train_tail` | -0.16 [-0.79, +0.49] | -0.12 | +0.08 | -0.25 | -0.06 [-0.79, +0.63] |
| `A_min_ema` | -0.18 [-0.82, +0.49] | -0.15 | +0.05 | -0.27 | -0.08 [-0.81, +0.63] |
| `A_pos_ema` | -0.17 [-0.86, +0.47] | -0.03 | -0.05 | -0.19 | +0.04 [-0.76, +0.70] |
| `A_first_half_ema` | -0.18 [-0.82, +0.49] | -0.15 | +0.05 | -0.27 | -0.08 [-0.81, +0.63] |
| `A_second_half_ema` | -0.19 [-0.85, +0.50] | -0.11 | +0.04 | -0.24 | -0.10 [-0.86, +0.61] |
| `A_heldout_curve_ema` | -0.17 [-0.80, +0.52] | -0.13 | +0.06 | -0.25 | -0.07 [-0.80, +0.64] |
| `A_heldout_curve_raw` | -0.18 [-0.82, +0.49] | -0.15 | +0.05 | -0.27 | -0.08 [-0.81, +0.63] |
| `A_rel_ema` | -0.05 [-0.68, +0.62] | +0.02 | +0.21 | -0.11 | +0.04 [-0.69, +0.70] |
| `sum_loss` | -0.25 [-0.69, +0.37] | -0.31 | -0.30 | -0.09 | -0.27 [-0.75, +0.35] |
| `L_fin_ema` | -0.13 [-0.70, +0.44] | -0.20 | -0.22 | -0.01 | -0.20 [-0.77, +0.40] |
| `L_fin_raw` | -0.17 [-0.72, +0.41] | -0.26 | -0.25 | -0.07 | -0.25 [-0.79, +0.35] |
| `L_start` | -0.29 [-0.75, +0.31] | -0.35 | -0.31 | -0.26 | -0.31 [-0.70, +0.30] |
| `drop_ema` | -0.21 [-0.94, +0.47] | -0.15 | -0.09 | -0.33 | -0.17 [-0.92, +0.58] |

## Excluding Chaotic, StepFunction (n = 9)

| Estimator | CRPS / naive (geometric mean) | CRPS / naive (median) | CRPS rank | MASE / naive (geometric mean) | GIFT-Eval NLL rank |
|---|---|---|---|---|---|
| `A_ema` | -0.73 [-0.98, -0.16] | -0.67 | -0.30 | -0.73 | -0.38 [-1.00, +0.32] |
| `A_raw` | -0.68 [-0.93, -0.11] | -0.60 | -0.23 | -0.65 | -0.33 [-0.86, +0.37] |
| `A_train_tail` | -0.70 [-0.94, -0.14] | -0.62 | -0.25 | -0.68 | -0.35 [-0.94, +0.35] |
| `A_min_ema` | -0.73 [-0.98, -0.16] | -0.67 | -0.30 | -0.73 | -0.38 [-1.00, +0.32] |
| `A_pos_ema` | -0.75 [-1.00, -0.23] | -0.50 | -0.50 | -0.67 | -0.33 [-0.86, +0.47] |
| `A_first_half_ema` | -0.73 [-0.98, -0.16] | -0.67 | -0.30 | -0.73 | -0.38 [-1.00, +0.32] |
| `A_second_half_ema` | -0.75 [-1.00, -0.21] | -0.60 | -0.33 | -0.68 | -0.43 [-1.00, +0.30] |
| `A_heldout_curve_ema` | -0.72 [-0.93, -0.11] | -0.65 | -0.28 | -0.70 | -0.37 [-0.94, +0.34] |
| `A_heldout_curve_raw` | -0.73 [-0.98, -0.16] | -0.67 | -0.30 | -0.73 | -0.38 [-1.00, +0.32] |
| `A_rel_ema` | -0.48 [-0.86, +0.25] | -0.42 | -0.05 | -0.47 | -0.20 [-0.74, +0.58] |
| `sum_loss` | -0.12 [-0.73, +0.58] | -0.22 | -0.20 | +0.00 | -0.15 [-0.75, +0.48] |
| `L_fin_ema` | -0.03 [-0.68, +0.68] | -0.12 | -0.15 | +0.10 | -0.08 [-0.67, +0.58] |
| `L_fin_raw` | -0.12 [-0.73, +0.58] | -0.22 | -0.20 | +0.00 | -0.15 [-0.75, +0.48] |
| `L_start` | -0.42 [-0.88, +0.21] | -0.42 | -0.28 | -0.30 | -0.23 [-0.68, +0.53] |
| `drop_ema` | -0.90 [-1.00, -0.53] | -0.75 | -0.57 | -0.85 | -0.57 [-1.00, +0.21] |

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
| kernelsynth | 68.4 | 43.7 | 44.7 | 68.4 | 116.7 | 52.5 | 15.9 | 59.9 | 49.9 | 0.226 | 38.4 | -14.99 | -2.62 | 136.49 | 151.48 |
| ets | 26.7 | 21.6 | 21.8 | 26.7 | 172.3 | 22.6 | 4.2 | 26.0 | 25.2 | 0.104 | 494.2 | 233.76 | 236.32 | 361.75 | 128.00 |
| waveform | 70.9 | 59.1 | 56.6 | 70.9 | 99.1 | 56.3 | 14.6 | 69.7 | 61.3 | 0.282 | 146.3 | 37.69 | 43.61 | 163.43 | 125.75 |
| sde | 11.0 | 8.4 | 7.9 | 11.0 | 46.3 | 9.1 | 1.9 | 10.3 | 9.0 | 0.081 | 180.1 | 84.57 | 85.87 | 152.09 | 67.52 |
| stepfunction | 49.5 | 26.2 | 29.8 | 49.5 | 126.4 | 35.9 | 13.6 | 34.9 | 33.3 | 0.166 | 28.8 | -10.37 | 1.25 | 139.14 | 149.51 |
| tsi | 64.8 | 46.5 | 49.4 | 64.8 | 147.3 | 49.9 | 15.0 | 56.6 | 47.0 | 0.204 | -119.3 | -92.06 | -82.87 | 67.12 | 159.18 |
| arima | 24.2 | 21.0 | 19.8 | 24.2 | 145.5 | 18.7 | 5.5 | 19.1 | 18.7 | 0.110 | 171.1 | 73.47 | 75.08 | 183.30 | 109.83 |
| fbm | 9.6 | 7.7 | 6.1 | 9.6 | 105.5 | 7.9 | 1.7 | 8.6 | 5.8 | 0.060 | -197.8 | -103.73 | -102.77 | -22.89 | 80.83 |
| garch | 16.6 | 12.4 | 12.8 | 16.6 | 99.1 | 13.1 | 3.4 | 13.5 | 14.9 | 0.088 | -73.9 | -45.23 | -43.13 | 49.09 | 94.33 |
| chaotic | 126.2 | 86.8 | 86.8 | 126.2 | 201.2 | 93.6 | 32.6 | 106.7 | 87.1 | 0.310 | -30.3 | -78.27 | -58.55 | 125.17 | 203.44 |
| timesynth | 15.3 | 12.9 | 12.4 | 15.3 | 45.5 | 12.6 | 2.6 | 15.0 | 13.0 | 0.117 | 95.3 | 40.04 | 41.25 | 105.17 | 65.13 |
