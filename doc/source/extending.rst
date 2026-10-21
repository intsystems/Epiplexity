Extending: adding your own estimator
====================================

The current protocol returns a model/data code decomposition. See
:doc:`measurements` for the four-method proposal and its quantity-specific
contracts, including the planned predictive-information result and Bayesian
reference wrapper. Registering a class supplies a name; its algorithm still
needs an implementation.

Чтобы добавить новый алгоритм оценки эпиплексии, достаточно написать **один модуль** и не менять
остальные части библиотеки. Всё, что нужно, — реализовать контракт
:class:`episynth.estimators.base.EpiplexityEstimator` и зарегистрировать его.

Шаг 1. Контракт
---------------

Каждый оценщик должен иметь атрибут ``name`` и метод ``run``:

.. code-block:: python

    def run(
        self, model: MoiraiModule, generator: Generator, spec: WindowSpec,
        config: TrainConfig, out_dir: str | Path,
    ) -> dict[str, float]:
        """Выполнить прогон обучения и вернуть оценку в битах."""

Аргументы:

* ``model`` (:class:`~episynth.model.moirai.MoiraiModule`) — модель-наблюдатель;
* ``generator`` (:class:`~episynth.generators.base.Generator`) — генератор рядов;
* ``spec`` (:class:`~episynth.windows.WindowSpec`) — форма задачи прогноза;
* ``config`` (:class:`~episynth.training.TrainConfig`) — параметры прогона;
* ``out_dir`` — каталог для артефактов.

Возврат — словарь с ключами ``model_part_bits`` и ``data_part_bits`` (и любыми дополнительными).

Шаг 2. Модуль алгоритма
-----------------------

Создайте файл, например ``my_package/my_estimator.py``:

.. code-block:: python

    from pathlib import Path

    from episynth.estimators import EpiplexityEstimator, build, register
    from episynth.generators.base import Generator
    from episynth.model.moirai import MoiraiModule
    from episynth.training import TrainConfig
    from episynth.windows import WindowSpec

    @register("my_estimator")
    class MyEstimator(EpiplexityEstimator):
        name = "my_estimator"

        def __init__(self, variant: str = "ema") -> None:
            self.variant = variant

        def run(
            self, model: MoiraiModule, generator: Generator, spec: WindowSpec,
            config: TrainConfig, out_dir: str | Path,
        ) -> dict[str, float]:
            # Рабочий пример обёртки над встроенным алгоритмом.
            # Собственный алгоритм заменит тело этого метода.
            return build("prequential", variant=self.variant).run(
                model, generator, spec, config, out_dir
            )

Шаг 3. Использование
--------------------

После регистрации алгоритм доступен по имени через фабрику и CLI:

.. code-block:: python

    import my_package.my_estimator  # регистрирует внешний плагин
    from episynth.estimators import build
    est = build("my_estimator", variant="ema")
    result = est.run(model, generator, spec, config, "runs/my")

Для встроенного оценщика достаточно поля ``estimator`` в конфигурации CLI.
Внешний плагин нужно импортировать **в том же процессе** перед вызовом CLI:

.. code-block:: python

    import my_package.my_estimator
    from episynth.train import main

    main(["--config", "configs/kernelsynth.json", "--out", "runs/epi",
          "--set", "estimator=my_estimator"])

Если вы добавляете алгоритм в саму библиотеку ``episynth``, положите его в
``src/episynth/estimators/`` и убедитесь, что он импортируется (встроенные оценщики
регистрируются в :mod:`episynth.estimators.registry` автоматически при импорте).

Пример: минимальный оценщик с общим ``Trainer``
-----------------------------------------------

Проверка регистрации:

.. code-block:: python

    from episynth.estimators import EpiplexityEstimator, available, build
    assert "prequential" in available()
    assert "requential" in available()
    est = build("prequential")
    isinstance(est, EpiplexityEstimator)  # True
