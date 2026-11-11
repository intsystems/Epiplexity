r"""Контракт оценщика эпиплексии (плагин).

Оценщик — это протокол, который из модели, генератора рядов, формы задачи и параметров прогона
получает полный прогон обучения и возвращает JSON-совместимый результат. Поля и единицы
зависят от метода: разложение кода у prequential, снижение predictive loss у V-information.
Разные алгоритмы (prequential, requential и пользовательские)
реализуют один и тот же :meth:`EpiplexityEstimator.run` и регистрируются в реестре по имени, поэтому пользователь
выбирает алгоритм по имени и добавляет новый, не трогая остальные модули библиотеки.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..generators.base import Generator
from ..model.moirai import MoiraiModule
from ..training import TrainConfig
from ..windows import WindowSpec


@runtime_checkable
class EpiplexityEstimator(Protocol):
    """Контракт алгоритма оценки эпиплексии.

    Каждый оценщик — это класс с атрибутом `name` (ключ в реестре) и методом :meth:`run`,
    который выполняет прогон обучения и пишет результат в `out_dir`.

    Args:
        name: Имя алгоритма, под которым он зарегистрирован в реестре.

    Attributes:
        name: Ключ алгоритма в реестре; используется для выбора через CLI и `build`.
    """

    name: str

    def run(
        self,
        model: MoiraiModule,
        generator: Generator,
        spec: WindowSpec,
        config: TrainConfig,
        out_dir: str | Path,
    ) -> Mapping[str, Any]:
        """Выполняет прогон обучения и возвращает оценку эпиплексии.

        Args:
            model: Модель-наблюдатель; инициализирована извне (например, через build_model).
            generator: Генератор синтетических рядов.
            spec: Форма задачи прогноза.
            config: Параметры прогона.
            out_dir: Каталог, куда пишутся артефакты прогона и результат.

        Returns:
            JSON-совместимый словарь. Prequential возвращает ``model_part_bits``,
            ``data_part_bits`` и ``total_bits``; V-information — predictive entropies,
            их разность в битах на окно и метаданные протокола.
        """
        ...
