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

То же по файлу конфигурации, с корпусом вместо генерации на лету. Корпус генератора — массив
`(n, 1024)` float32 с манифестом, строка `i` есть ряд с сидом `first_seed + i`, поэтому прогон на
корпусе даёт те же окна, что прогон на лету по тем же сидам:

```bash
python -m episynth.corpus --generator kernelsynth --first-seed 0 --n 2000000 --out corpora/kernelsynth/train
python -m episynth.corpus --generator kernelsynth --first-seed 1000000000 --n 100000 --out corpora/kernelsynth/heldout
python -m episynth.train --config configs/kernelsynth.json --out runs/kernelsynth --set train.lr=3e-4
```

Файл конфигурации — разделы `data`, `spec`, `model`, `train` и путь `context_table`; формат и
правила в докстринге `episynth.train`. Равная смесь генераторов или корпусов (`Mixture`, раздел
`data.mixture`) служит для подбора шага обучения, общего для всех генераторов.

Оценка чекпойнта прогона на краткосрочной части GIFT-Eval (снимок датасета — `huggingface_hub.snapshot_download("Salesforce/GiftEval", repo_type="dataset")`):

```bash
python -m episynth.evaluation.run --data <снимок> --out eval/kernelsynth --model moirai \
    --checkpoint runs/kernelsynth/ckpt_1.pt --config-json runs/kernelsynth/config.json --patch-size 32 --device cuda
```

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
