# Estimator variants and Spearman correlations with forecast performance

Each cell shows Spearman's ρ between an estimate and a performance metric across generators. Lower metric values are better, so negative ρ associates a higher estimate with better models. Brackets give 95% bootstrap intervals where reported.


## All 11 (n = 11)

| Estimator | CRPS / naive (geometric mean) | CRPS / naive (median) | CRPS rank | MASE / naive (geometric mean) | GIFT-Eval NLL rank |
|---|---|---|---|---|---|
| `A_ema` | -0.18 [-0.82, +0.49] | -0.15 [-0.77, +0.63] | +0.05 [-0.71, +0.68] | -0.27 [-0.94, +0.50] | -0.08 [-0.81, +0.63] |
| `A_raw` | -0.15 [-0.75, +0.52] | -0.10 [-0.68, +0.64] | +0.09 [-0.60, +0.69] | -0.23 [-0.85, +0.53] | -0.05 [-0.76, +0.64] |
| `sum_loss` | -0.25 [-0.69, +0.37] | -0.31 [-0.75, +0.25] | -0.30 [-0.83, +0.35] | -0.09 [-0.65, +0.59] | -0.27 [-0.75, +0.35] |
| `L_fin_ema` | -0.13 [-0.70, +0.44] | -0.20 [-0.75, +0.39] | -0.22 [-0.80, +0.44] | -0.01 [-0.63, +0.68] | -0.20 [-0.77, +0.40] |
| `drop_ema` | -0.21 [-0.94, +0.47] | -0.15 [-0.81, +0.59] | -0.09 [-0.80, +0.63] | -0.33 [-0.95, +0.44] | -0.17 [-0.92, +0.58] |

## Excluding chaotic, waveform (n = 9)

| Estimator | CRPS / naive (geometric mean) | CRPS / naive (median) | CRPS rank | MASE / naive (geometric mean) | GIFT-Eval NLL rank |
|---|---|---|---|---|---|
| `A_ema` | -0.47 [-0.98, +0.38] | -0.43 [-0.95, +0.40] | -0.38 [-0.98, +0.38] | -0.67 [-1.00, +0.05] | -0.48 [-1.00, +0.32] |
| `A_raw` | -0.42 [-0.88, +0.40] | -0.37 [-0.82, +0.46] | -0.32 [-0.84, +0.38] | -0.60 [-0.98, +0.10] | -0.43 [-0.94, +0.35] |
| `sum_loss` | -0.17 [-0.73, +0.54] | -0.17 [-0.73, +0.53] | -0.28 [-0.84, +0.47] | +0.15 [-0.59, +0.81] | -0.15 [-0.72, +0.54] |
| `L_fin_ema` | -0.03 [-0.73, +0.58] | -0.05 [-0.73, +0.61] | -0.18 [-0.82, +0.58] | +0.22 [-0.53, +0.85] | -0.08 [-0.64, +0.62] |
| `drop_ema` | -0.53 [-1.00, +0.20] | -0.53 [-0.95, +0.33] | -0.47 [-0.95, +0.32] | -0.80 [-1.00, -0.28] | -0.55 [-1.00, +0.25] |

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
| kernelsynth | 68.4 | 43.7 | 38.4 | -14.99 | 151.48 |
| ets | 26.7 | 21.6 | 494.2 | 233.76 | 128.00 |
| waveform | 70.9 | 59.1 | 146.3 | 37.69 | 125.75 |
| sde | 11.0 | 8.4 | 180.1 | 84.57 | 67.52 |
| stepfunction | 49.5 | 26.2 | 28.8 | -10.37 | 149.51 |
| tsi | 64.8 | 46.5 | -119.3 | -92.06 | 159.18 |
| arima | 24.2 | 21.0 | 171.1 | 73.47 | 109.83 |
| fbm | 9.6 | 7.7 | -197.8 | -103.73 | 80.83 |
| garch | 16.6 | 12.4 | -73.9 | -45.23 | 94.33 |
| chaotic | 126.2 | 86.8 | -30.3 | -78.27 | 203.44 |
| timesynth | 15.3 | 12.9 | 95.3 | 40.04 | 65.13 |
