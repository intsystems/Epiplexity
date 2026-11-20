# episynth

Инструменты эксперимента по эпиплексии синтетических временных рядов: генераторы рядов,
окна задачи прогноза, перенос модели Moirai и обучение с логом лосса каждого окна.

## Слои

- `episynth.generators` — ряды по сидам.
- `episynth.windows` — окна задачи: контекст случайной длины, горизонт, патчи, деквантование;
  поток батчей по возрастающим сидам с генерацией в фоновом потоке.
- `episynth.model` — перенос Moirai 1.x из uni2ts с загрузкой опубликованных чекпойнтов;
  смесь распределений с лог-плотностью в натах.
- `episynth.corpus` — корпуса: генерация рядов по сидам в файл процессами и чтение корпуса как
  генератора, чтобы обучение не упиралось в CPU.
- `episynth.training` — прогон обучения: лосс каждого окна до шага по нему, постоянный шаг
  после разогрева, EMA весов, чекпойнты на долях прогона, отложенный набор, возобновление.
- `episynth.train` — запуск прогона по файлу конфигурации: генератор, корпус или смесь,
  таблица длин контекста, переопределения из командной строки.
- `episynth.analysis` — площадь над финальным лоссом с ошибкой от отложенного набора, кривая
  площади по префиксам, сводки качества по задачам (ранги, отношения к наиву), корреляции
  Спирмена с бутстрепом по генераторам.
- `episynth.evaluation` — оценка на GIFT-Eval: загрузка снимка `Salesforce/GiftEval` без
  gluonts, тестовые окна как в бенчмарке, прогнозчики (сезонный наив, Moirai в нашем протоколе
  и в протоколе тетради лидерборда), метрики по окнам с точным сведением в определениях
  лидерборда и в масштабированном виде, лог-плотность правды.
- `episynth.estimators` — плагинный интерфейс оценки эпиплексии: алгоритмы регистрируются по
  имени и выбираются полем `estimator` в конфигурации `episynth.train`. Встроены `prequential`
  (площадь над кривой лосса), `v_information` (польза контекста для прогноза) и скелет `requential`; пользователь добавляет свой алгоритм одним
  модулем через `@register`.

```python
from episynth.generators import GENERATORS
from episynth.model.moirai import small
from episynth.training import TrainConfig, Trainer, build_model
from episynth.windows import WindowSpec

spec = WindowSpec(patch_size=32, context_lengths=(128, 256, 512))
model = build_model(small(patch_sizes=(32,), components=("student_t", "normal_fixed", "log_normal")), init_seed=0)
cfg = TrainConfig(n_windows=4_800_000, batch_size=256, device="cuda")
Trainer(model, GENERATORS["kernelsynth"](device="cuda"), spec, cfg, "runs/kernelsynth").run()
```

Что пишется в каталог прогона и в каких единицах — в докстринге `episynth.training`.
Обучающие и отложенные сиды проверяются на непересечение, включая компоненты смеси.
При возобновлении проверяются источник данных, форма окна, модель и настройки обучения;
для prequential можно менять `device`, `compile`, `eval_batch_size`, `stream_workers`,
`log_every` и `fused_adam`. Старые чекпойнты без описания источника нельзя безопасно
возобновить: для нового обучения нужен отдельный каталог. Их веса по-прежнему можно
использовать для оценки. Финальный чекпойнт и отложенная оценка создаются всегда;
повторный запуск восстанавливает незавершённую финальную оценку.

То же по файлу конфигурации, с корпусом вместо генерации на лету. Корпус генератора — массив
`(n, 1024)` float32 с манифестом, строка `i` есть ряд с сидом `first_seed + i`, поэтому прогон на
корпусе даёт те же окна, что прогон на лету по тем же сидам:

```bash
python -m episynth.corpus --generator kernelsynth --first-seed 0 --n 2000000 --out corpora/kernelsynth/train
python -m episynth.corpus --generator kernelsynth --first-seed 1000000000 --n 100000 --out corpora/kernelsynth/heldout
python -m episynth.train --config configs/kernelsynth.json --out runs/kernelsynth --set train.lr=3e-4
```

Файл конфигурации — разделы `data`, `spec`, `model`, `train`, путь `context_table` и имя
алгоритма оценки `estimator`; формат и правила в докстринге `episynth.train`. Равная смесь
генераторов или корпусов (`Mixture`, раздел `data.mixture`) служит для подбора шага обучения,
общего для всех генераторов.

Поле `estimator` выбирает алгоритм оценки эпиплексии из реестра `episynth.estimators` (по
умолчанию `prequential`). Прогон пишет обычные артефакты `Trainer`, а результат оценки — в
`estimator_result.json`:

```bash
python -m episynth.train --config configs/kernelsynth.json --out runs/kernelsynth \
    --set estimator=prequential --set train.lr=3e-4
```

Плагинный интерфейс описан в `episynth.estimators`: встроены `prequential` (площадь над кривой
лосса, финальный лосс берётся с отложенного набора `heldout_ema_1.f32` через
`episynth.analysis.run_area`), `v_information` и скелет `requential`. Свой алгоритм добавляется одним модулем:
класс, реализующий `EpiplexityEstimator`, помечается `@register("имя")`; выбирается по имени в
`--set estimator=имя`.

## V-information

`estimator=v_information` обучает два Moirai с одинаковыми начальными весами и бюджетом:
с контекстом и без него. Параметры конструктора передаются через `estimator_config`:
`variant` (`ema` по умолчанию или `raw`), `test_first_seed` (2 000 000 000),
`test_windows` (100 000, не меньше двух). Существующий `train.heldout_*` задаёт validation;
train, validation и test должны быть непересекающимися и покрываться источником данных,
включая исходные сиды компонентов смеси. Поддерживаются непрерывные компоненты плотности.

```bash
python -m episynth.train --config experiments/demos/v_information.json --out runs/v_information_cpu
python -m episynth.train --config experiments/demos/v_information.json --out runs/v_information_raw \
    --set estimator_config.variant=raw
python -m episynth.estimators.run --estimator v_information --generator waveform \
    --config-json experiments/demos/v_information.json --out runs/v_information_cli \
    --estimator-config '{"test_first_seed":2000,"test_windows":32}'
```

Среднее и масштаб вычисляются дополнительным проходом только по точкам горизонта train,
потоково в float64. Обе модели используют эту фиксированную нормировку; `log_prob` и NLL
остаются в исходных единицах ряда. `MoiraiConfig.fixed_scaling=(loc, scale)` требует
`scaling=True`, конечных констант и положительного масштаба. `ignore_context=True` задаёт
контекстно-независимый режим внутри модели; оба режима сохраняются с конфигурацией.

Чекпойнты выбранного `variant` сравниваются на validation в `train.eval_precision`.
Условный предиктор выбирается среди условных чекпойнтов и лучшего baseline; при равенстве
предпочитается baseline, затем более ранний шаг. Только после фиксации выбора читается test.
Обе роли оцениваются на идентичных целях, включая деквантование. Сумма NLL по горизонту
использует текущую факторизацию Moirai в произведение плотностей точек.

Результат содержит `unconditional_entropy_bits_per_window`,
`conditional_entropy_bits_per_window`, их разность `v_information_bits_per_window`,
парную `standard_error_bits_per_window`, `n_test_windows` и метаданные. Отрицательные
оценки сохраняются; ошибка отражает test-выборку при фиксированных моделях, а не разброс
обучения. Это оценка при ограниченном бюджете оптимизации, без утверждения о достижении
инфимума семейства или PAC-гарантии. Полей разложения стоимости кода у неё нет.

Каталоги `unconditional/` и `conditional/` содержат обычные артефакты Trainer.
`v_information.json` хранит нормировку, конфигурации, сплиты и выбранные чекпойнты;
`test_unconditional.f32` и `test_conditional.f32` — NLL в натах в порядке test-сидов,
`pvi_bits.f64` — их оконные разности в битах. Повторный запуск проверяет совместимость
исходных весов и настроек и возобновляет незавершённые ветви. Пересчёт готового результата
читает только сохранённые файлы и проверяет их полноту и согласованность:

```python
from episynth.estimators.v_information import VInformationEstimator
result = VInformationEstimator().from_run_dir("runs/v_information_cpu")
print(result["v_information_bits_per_window"])
```

Оценка чекпойнта прогона на краткосрочной части GIFT-Eval (снимок датасета — `huggingface_hub.snapshot_download("Salesforce/GiftEval", repo_type="dataset")`):

```bash
python -m episynth.evaluation.run --data <снимок> --out eval/kernelsynth --model moirai \
    --checkpoint runs/kernelsynth/ckpt_1.pt --patch-size 32 --device cuda
```

Конфигурация модели берётся из собственного `.pt` чекпойнта. Если передан
`--config-json`, он должен ей соответствовать. Файл `evaluation.json` фиксирует модель,
чекпойнт и настройки оценки: готовые строки `summary.csv` используются повторно только
при совпадении этих настроек. Для другого эксперимента или старой сводки без этого
файла нужен отдельный каталог вывода.

Сводка всегда содержит один и тот же набор колонок. Метрики с нулевым знаменателем
или без подходящих наблюдений записываются как `NaN`: например, MASE для постоянной
истории или метрики при полностью отсутствующих метках. Счётчики окон и точек
позволяют отличить эти случаи; неопределённая метрика не прерывает оценку.

Проверка обучения: прогон на генераторе `ou` (AR(1) с приором на коэффициент) и сравнение его
лосса с точным байесовским лоссом тех же окон, посчитанным квадратурой по приору
(`python -m episynth.checks.ou_bayes --run runs/ou`); обученная модель должна подойти к нему сверху.

Проверки конвейера: сверка переноса модели с uni2ts на одних входах
(`python -m episynth.checks.moirai --checkpoint <каталог> --uni2ts-python <интерпретатор с uni2ts 2.0.0>`),
сверка данных, окон и формул с опубликованным сезонным наивом
(`python -m episynth.checks.gift_eval_naive --data <снимок> --published <results/Seasonal_Naive/all_results.csv>`),
сравнение прогона с опубликованной таблицей модели (`python -m episynth.checks.gift_eval_published`).

## Генераторы

`episynth.generators` — одиннадцать генераторов одномерных рядов из Mix, Don't Pick
(Nagpal и др., 2026, arXiv:2606.09912): KernelSynth, ETS, Waveform, SDE, StepFunction, TSI,
ARIMA, fBm, GARCH, Chaotic, TimeSynth. Генеративный процесс каждого повторяет оригинал, а
вычисление переписано пакетом по рядам без сторонних библиотек оригинала (`fbm`, `timesynth`).

У ARIMA по умолчанию включён `coefficient_mode="legacy"`, сохраняющий опубликованные
ряды и унаследованную ошибку порядка коэффициентов полинома: все 100 кандидатов
отклоняются и выбирается запасной вектор из `U(-0.3, 0.3)`. Для нового эксперимента
исправленный отбор по корням лагового полинома включается явно:
`ARIMA(length=1024, coefficient_mode="corrected")` или
`--set data.generator_args.coefficient_mode='"corrected"'` в CLI обучения.
Это другое распределение данных; режим записывается в метаданные, а части корпуса
с разными режимами не объединяются. Основной и `exact` режимы используют один и
тот же выбранный отбор; только `legacy` воспроизводит исходный генератор.

```python
import numpy as np
from episynth.generators import GENERATORS

gen = GENERATORS["sde"](length=1024)
x = gen.generate(np.arange(1000))          # (1000, 1024) float32, по ряду на сид
y = gen.generate([5, 6], exact=True)       # ряды оригинала для тех же сидов
corpus = gen.reproduce(random_seed=42, n=100)  # корпус оригинала с сидом 42
```

Ряд зависит только от сида и длины: любой набор сидов в любом порядке даёт те же ряды.
Режим `exact` повторяет оригинал вплоть до порядка обращений к генератору случайных чисел и
нужен для проверок; у большинства генераторов основной режим с ним совпадает до бита. Что
именно и как считает каждый генератор — в докстринге его класса.

## Проверки

Против клона оригинала. Оригинал считается отдельным интерпретатором с его версиями
зависимостей (numpy 1.26.4, scipy 1.11.4, scikit-learn 1.7.1, fbm, timesynth): под numpy 2
генератор TimeSynth оригинала падает в пошаговых циклах и молча подменяет ряды запасными.

```bash
uv venv .venv-ref && uv pip install --python .venv-ref/bin/python \
    numpy==1.26.4 scipy==1.11.4 scikit-learn==1.7.1 fbm==0.3.0 tqdm git+https://github.com/TimeSynth/TimeSynth.git
python -m episynth.checks.generators --original <клон mix-dont-pick> \
    --original-python .venv-ref/bin/python --out checks_out
```

На каждый генератор пишется JSON: доля рядов, совпавших с оригиналом до бита; p-значения
Колмогорова–Смирнова по статистикам ряда между корпусом оригинала и корпусом основного
режима; скорость обоих. У KernelSynth совпадение до бита есть только при той же сборке
numpy, что у оригинала: его режим воспроизведения строит вектор собственным разложением.
Инварианты без оригинала: `pytest`.
