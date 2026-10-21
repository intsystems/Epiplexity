Extending: adding your own estimator
====================================

Чтобы добавить новый алгоритм оценки эпиплексии, достаточно написать **один модуль** и не менять
остальные части библиотеки. Всё, что нужно, — реализовать контракт
:class:`episynth.estimators.base.EpiplexityEstimator` и зарегистрировать его.

Шаг 1. Контракт
---------------

Каждый оценщик должен иметь атрибут ``name`` и метод ``run``:

.. code-block:: python

    def run(self, model, generator, spec, config, out_dir) -> dict[str, float]:
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

    from episynth.estimators import register
    from episynth.estimators.base import EpiplexityEstimator

    @register("my_estimator")
    class MyEstimator(EpiplexityEstimator):
        name = "my_estimator"

        def __init__(self, param: float = 1.0) -> None:
            self.param = param

        def run(self, model, generator, spec, config, out_dir):
            # Любая логика: обучение, накопление кодовой длины, ...
            # При необходимости можно использовать общий движок:
            from episynth.training import Trainer
            Trainer(model, generator, spec, config, out_dir).run()
            # ... и затем посчитать model/data part по записанным артефактам.
            return {"model_part_bits": 0.0, "data_part_bits": 0.0}

Шаг 3. Использование
--------------------

После регистрации алгоритм доступен по имени через фабрику и CLI:

.. code-block:: python

    from episynth.estimators import build
    est = build("my_estimator", param=2.0)
    result = est.run(model, generator, spec, config, "runs/my")

Через CLI конвейера ``episynth.train`` оценщик выбирается полем ``estimator`` в файле
конфигурации:

.. code-block:: bash

    python -m episynth.train --config configs/kernelsynth.json --out runs/epi \
        --set estimator=my_estimator

Если вы добавляете алгоритм в саму библиотеку ``episynth``, положите его в
``src/episynth/estimators/`` и убедитесь, что он импортируется (встроенные оценщики
регистрируются в :mod:`episynth.estimators.registry` автоматически при импорте).

Пример: минимальный оценщик с общим ``Trainer``
-----------------------------------------------

Проверка регистрации:

.. code-block:: python

    from episynth.estimators import available, build
    assert "prequential" in available()
    assert "requential" in available()
    est = build("prequential")
    isinstance(est, EpiplexityEstimator)  # True