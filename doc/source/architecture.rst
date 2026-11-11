.. _architecture:

Architecture
============

Пакет ``episynth`` — это библиотека оценки эпиплексии синтетических временных рядов. Эксперимент
устроен так: модель-наблюдатель обучается на потоке окон, сгенерированных синтетическим
генератором, а оценщик возвращает разложение кодовой длины
в битах или пользу контекста для прогноза в битах на окно (V-information).

Слои
----

Библиотека разделена на нейтральные слои и плагинный слой оценщиков. Нейтральные слои не зависят
от выбранного алгоритма оценки и используются всеми оценщиками одинаково.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Модуль
     - Назначение
   * - ``episynth.generators``
     - Одиннадцать генераторов одномерных рядов (KernelSynth, ETS, Waveform, SDE и др.) из
       Mix, Don't Pick и OU-эталон. Ряд зависит только от сида и длины; ``generate(seeds)`` даёт ``(n, L)``.
   * - ``episynth.windows``
     - Окна задачи прогноза: ``WindowSpec`` (форма задачи), ``WindowBatch`` (батч окон),
       ``WindowStream`` (поток батчей по возрастающим сидам в фоновом потоке).
   * - ``episynth.model.moirai``
     - Порт Moirai 1.x: ``MoiraiModule``, ``MoiraiConfig``, ``Mixture`` (смесь распределений
       с ``log_prob`` и ``sample``), загрузка опубликованных чекпойнтов.
   * - ``episynth.training``
     - Общий движок обучения ``Trainer``: окна, EMA весов, чекпойнты, возобновление, heldout-оценка.
       Протокол-специфичный шаг вынесен в переопределяемый ``compute_loss``.
   * - ``episynth.evaluation``
     - Оценка на GIFT-Eval: прогнозчики, метрики, ``run``.
   * - ``episynth.estimators``
     - **Плагинный слой**: оценщики эпиплексии. Выбор и добавление алгоритмов.
       Именно этот слой расширяется пользователем.

Плагинный интерфейс оценщиков
-----------------------------

Центральная абстракция — :class:`episynth.estimators.base.EpiplexityEstimator` — контракт, который
должен реализовать каждый алгоритм:

.. code-block:: python

    class EpiplexityEstimator(Protocol):
        name: str
        def run(
            self, model: MoiraiModule, generator: Generator, spec: WindowSpec,
            config: TrainConfig, out_dir: str | Path,
        ) -> Mapping[str, Any]: ...

Метод ``run`` выполняет обучение и возвращает JSON-совместимый словарь.
Prequential возвращает ``model_part_bits`` и ``data_part_bits`` (в битах);
V-information — predictive entropies и их разность в битах на окно.

Алгоритмы регистрируются в :mod:`episynth.estimators.registry` по имени и создаются фабрикой
:func:`episynth.estimators.registry.build`:

.. code-block:: python

    from pathlib import Path
    from episynth.estimators import EpiplexityEstimator, register, build
    from episynth.generators.base import Generator
    from episynth.model.moirai import MoiraiModule
    from episynth.training import TrainConfig
    from episynth.windows import WindowSpec

    @register("my_estimator")
    class MyEstimator(EpiplexityEstimator):
        def run(
            self, model: MoiraiModule, generator: Generator, spec: WindowSpec,
            config: TrainConfig, out_dir: str | Path,
        ) -> dict[str, float]:
            ...  # обучение и подсчёт кодовой длины
            return {"model_part_bits": ..., "data_part_bits": ...}

    est = build("my_estimator")          # выбор по имени
    result = est.run(model, generator, spec, config, "runs/my")

Пользователь выбирает алгоритм по имени и может добавить собственный одним модулем, не меняя
нейтральные слои.

Встроенные оценщики
-------------------

* :class:`episynth.estimators.prequential.PrequentialEstimator` — prequential-оценка: сумма лоссов
  окон до шага по ним (площадь над кривой лосса). Реализован поверх общего ``Trainer``.
* :class:`episynth.estimators.requential.RequentialEstimator` — requential-оценка (код студента,
  реконструируемого по индексам сэмплов из распределения учителя). Зарегистрирован, но его прогон
  пока скелет: бросает :class:`NotImplementedError`. См. ``publications/research_notes/.../requential_coding.md``.

* :class:`episynth.estimators.v_information.VInformationEstimator` — два обучения
  Moirai с общей нормировкой по train, выбор моделей на validation и парное
  сравнение на test. Условный предиктор может игнорировать контекст.

Взаимодействие ``Trainer`` и оценщика
-------------------------------------

Общий ``Trainer`` в :mod:`episynth.training` выполняет механику обучения. Протокол-специфичный
шаг — вычисление лосса каждого окна — вынесен в метод ``Trainer.compute_loss``. Prequential-оценщик
использует ``Trainer`` без изменений; requential-оценщик (когда будет реализован) наследует
``Trainer`` и переопределяет ``compute_loss``, чтобы возвращать KL-оценку учитель-студент.

CLI
---

Выбор оценщика встроен в конфигурационный CLI :mod:`episynth.train`, которым запускаются большие
серии экспериментов. Файл конфигурации (разделы ``data``, ``spec``, ``model``, ``train`` и поле
``estimator``) передаётся через ``--config``, любое поле переопределяется через ``--set``:

.. code-block:: bash

    python -m episynth.train --config configs/kernelsynth.json --out runs/epi \
        --set train.lr=3e-4 --set estimator=prequential

Поле ``estimator`` задаёт имя алгоритма в реестре :mod:`episynth.estimators` (по умолчанию
``prequential``). Прогон пишет обычные артефакты ``Trainer``, а результат оценки — в
``estimator_result.json`` в каталоге прогона. Поддерживается и программный вызов:
``build("prequential").run(model, generator, spec, config, out_dir)``.
Параметры алгоритма передаются через ``estimator_config``. V-information пишет
две ветви обучения и общий манифест; пример: ``experiments/demos/v_information.json``.
