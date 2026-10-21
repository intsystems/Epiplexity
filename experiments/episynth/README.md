# Результаты прогонов episynth (серии 19–20 октября 2026)

*English summary.* Saved outputs of the two training series behind the time-series track
(prequential epiplexity of 11 synthetic generators vs. Moirai-Small quality on GIFT-Eval short):
per-run configs, per-step training curves, held-out losses at every checkpoint, GIFT-Eval
per-task summaries, analysis tables and figures. Per-window loss vectors (160 MB per run) are
attached to the GitHub release `results-2026-10-19`; checkpoints and per-window GIFT-Eval
metrics are kept in object storage and available on request.

Две серии одного протокола: на каждом из 11 генераторов из «Mix, Don't Pick» с нуля обучается
Moirai-Small (три компоненты смеси, патч 32, контекст до 512 по таблице длин GIFT-Eval, горизонт
64, деквантование 10⁻³), лосс каждого окна пишется до шага по нему, площадь над финальным лоссом
(prequential-эпиплексия) сопоставляется с качеством на 55 краткосрочных задачах GIFT-Eval.
Оптимизатор AdamW с группами параметров uni2ts, шаг 10⁻³ после разогрева 200 шагов, EMA весов с
постоянной времени 2 % шагов, bf16 с `torch.compile`, отложенный набор 100 тыс. окон с сидов от
10⁹, батч 1024.

| серия | каталог | окон на генератор | шагов | ряды | карты |
|---|---|---|---|---|---|
| 1 | `series1_2026-10-19/` | 1 999 872 | 1 953 | корпуса по 2 млн рядов, построенные заранее | RTX 4090 |
| 2 | `series2_2026-10-19/` | 39 999 488 | 39 062 | на лету пулом процессов; KernelSynth корпусом на A100 (float64) | RTX 4090, A100 |

Сиды окон в обеих сериях одни и те же, поэтому первые 2 млн окон серии 2 совпадают с серией 1.

## Что лежит

```
series<k>_<дата>/
  runs/<prefix><генератор>/
    run.json          конфигурация прогона целиком (данные, окна, модель, обучение, таблица контекстов)
    config.json       то же, что видит Trainer
    steps.csv         лог каждого 50-го шага: step, windows_seen, nats_per_point, lr, grad_norm, seconds
    train_curve.csv   кривая обучения по шагам: step, windows_seen, nats_per_point — среднее лосса
                      1024 окон шага, нат на точку (лосс окна / 64), окна до шага по ним
    heldout.jsonl     по чекпойнтам (доли 0.01…1): лосс на отложенном наборе для текущих (raw)
                      и EMA весов в точности обучения и в fp32; nats_per_window, nats_per_point
  eval/<prefix><генератор>/summary.csv   GIFT-Eval short, 55 задач, EMA-веса финального чекпойнта
  eval/seasonal_naive/summary.csv        сезонный наив на тех же окнах
  analysis/
    epiplexity_variants.md          корреляции Спирмена 15 вариантов оценки с качеством (все 11; без Chaotic, StepFunction)
    epiplexity_variants_subsets.md  то же для подмножеств без Chaotic, Waveform
    figures/                        рисунки серии
    timesfm3_reference.json         (серия 2) TimesFM-3 с лидерборда к нашему наиву, сверка наива по задачам
  gzip_baseline.json  (серия 1) gzip-сжимаемость рядов каждого генератора, байт на точку
scripts/              скрипты анализа, см. ниже
reference/gift_eval_leaderboard/   опубликованные all_results.csv TimesFM-3 и Seasonal_Naive из
                                   репозитория SalesforceAIResearch/gift-eval (коммит 4d5ab3f)
```

Префикс прогонов: `series_` в серии 1, `s2_` в серии 2; `s2_waveform_seed1` — Waveform со вторым
сидом инициализации (разброс по сиду). Колонки `summary.csv` описаны в
`episynth.evaluation.metrics.aggregate`: `scaled_crps` (CRPS по траекториям, масштабированный
ошибкой сезонного наива на истории), `MASE[0.5]`, `mean_weighted_sum_quantile_loss` (CRPS в
определении лидерборда), покрытия и ширины интервалов, `nll_per_point`.

Единицы: лосс в натах в исходных единицах ряда с деквантованием 10⁻³; отрицательные значения у
почти детерминированных генераторов (Chaotic, TSI, fBm) — это следствие деквантования, не ошибка.
Площадь $A = \sum_i \ell_i - N L_{\mathrm{fin}}$ считается `episynth.analysis.run_area` по
`window_loss.f32` и `heldout_ema_1.f32`.

## Большие файлы

- Релиз GitHub `results-2026-10-19`: `series1_2026-10-19_window_losses.tar` (все прогоны серии 1)
  и `series2_2026-10-19_<генератор>_window_losses.tar` (по прогону). В каждом `window_loss.f32`
  (лосс каждого окна, float32 без заголовка, по номеру окна) и `heldout_<вариант>_<доля>.f32`
  (лоссы окон отложенного набора на чекпойнтах; в серии 1 только доля 1). Распаковываются
  поверх каталога серии:
  `tar -xf series2_2026-10-19_ets_window_losses.tar -C series2_2026-10-19/`.
- Чекпойнты (`ckpt_<доля>.pt`, `last.pt`), пооконные метрики GIFT-Eval и корпус KernelSynth на
  40 млн рядов (164 ГБ) — в объектном хранилище у Данилы Черноусова, по запросу.

## Скрипты анализа

Запускаются с установленным пакетом (или `PYTHONPATH=src`), нужны `matplotlib`, `scipy`,
`pandas`. Каталог серии — тот, где лежат `runs/` и `eval/`; для площадей нужны распакованные
`window_loss.f32` и `heldout_*.f32`. Префикс прогонов задаётся переменной `SERIES_PREFIX`
(по умолчанию `series_`).

- `series_analysis.py <серия> [gzip_baseline.json]` — сводная таблица: площади, финальные лоссы,
  качество, корреляции.
- `epiplexity_variants.py <серия> <out.md>` — 15 вариантов оценки против пяти мер качества;
  `variants_subset.py` — то же с исключением генераторов из `EXCLUDE=chaotic,waveform`.
- `corr_vs_windows.py <серия> <рисунки>` — корреляция оценок по чекпойнтам с итоговым качеством.
- `series_figures2.py <серия> <рисунки>` — метрика к наиву против площади, кривые обучения с
  закрашенной площадью; `series_figures3.py` — три определения оси x;
  `series_figures4.py <серия> <рисунки> <reference/gift_eval_leaderboard>` — с прямыми
  регрессии и горизонталью TimesFM-3.
- `timesfm_reference.py <серия> <reference/gift_eval_leaderboard> [prefix]` — сверка наива с
  лидербордом по задачам и отношения TimesFM-3 к наиву.
- `gzip_baseline.py <корпуса> [n]`, `verify_corpora.py <корпуса> [n]` — по корпусам на машине
  прогонов; `bundle_results.py <серия> <prefix> <куда>` — сборка этого пакета.

## Отступления от протокола, учтённые в данных

- Серия 2, KernelSynth: прогон упал на первом чекпойнте из-за ошибки сбора отложенного набора и
  возобновлён из `last.pt` шага 390; траектория та же по построению. Запись чекпойнта 0.01 в
  `heldout.jsonl` восстановлена по сохранённому `ckpt_0.01.pt`.
- Серия 2, TimeSynth: генератор переписан пакетом по рядам (совпадение с построчной версией до
  бита закреплено тестом) после того, как построчная версия давала 1 шаг/с.
- Серия 2, KernelSynth: ряды построены корпусом на GPU (float64 Холецкий), отличие от CPU-пути
  только в округлении разложения.
- Серия 2, Waveform: геометрическое среднее CRPS к наиву 1,373 из-за одного окна bitbrains_rnd
  с взорвавшейся траекторией; у `s2_waveform_seed1` 0,972. Медиана и ранг устойчивы.
