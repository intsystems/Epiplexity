# episynth: Proof of Concept / Prequential

Новая презентация TM2 на русском языке: **12 основных слайдов и 3 дополнительных**,
ориентир — 12–15 минут. Существующие презентации проекта не редактируются.

- [Готовый PDF](episynth_poc_prequential.pdf).
- [LaTeX-исходник](main.tex).
- [Что говорить и показывать](speaker_notes.md).
- [Происхождение чисел и рисунков](data/provenance.json).

Основная часть: исследовательский вопрос → архитектура → блочный prequential →
площадь над EMA held-out уровнем → реализация → конфигурация CPU-демо → сохранённые
графики → расширяемость → автодок. В приложении: неопределённость, результаты
больших серий и литература.

## Сборка

Нужны XeLaTeX (TeX Live / MiKTeX), Beamer, fontspec, polyglossia, TikZ, listings
и шрифты DejaVu Sans / DejaVu Sans Mono. Из этого каталога:

```bash
mkdir -p build
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
cp build/main.pdf episynth_poc_prequential.pdf
```

В Overleaf выберите компилятор **XeLaTeX** и главный файл `main.tex`.
В архиве исходников все рисунки уже находятся в `figures/`; для компиляции
не нужны Python, PyTorch, датасеты, веса модели или Интернет.

## Рисунки и численные результаты

`figures/demo_notebook.png` извлечён из выполненного notebook output в
`experiments/demos/demo_estimators.ipynb`; код презентации не переобучает модель.
Левая заливка исходного графика выделяет положительные превышения baseline,
но оценщик считает **знаковую** сумму, что прямо указано на слайде.

`figures/area_schematic.*` — явно обозначенная схематическая иллюстрация,
а не экспериментальные данные. `large_series_context.*` построен из CSV
`publications/technical_report/data/` по 11 основным генераторам каждой серии.
Точки дополнительного повторного сида Waveform не включены.

Для регенерации рисунков из полного checkout нужны NumPy и Matplotlib:

```bash
python make_figures.py
```

Этот скрипт читает notebook и CSV из репозитория; в самостоятельном архиве
исходников готовых рисунков достаточно для сборки PDF.

Теоретическая мотивация следует Finzi et al. (2026). Формула в слайдах —
операционная prequential excess-loss оценка при фиксированных наблюдателе
и протоколе. Она не заявляется точной теоретической эпиплексией, стоимостью
файла весов или доказательством полезности генератора для переноса.
