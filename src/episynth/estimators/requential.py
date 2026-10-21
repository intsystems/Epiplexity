r"""Requential-оценщик эпиплексии (скелет).

Требует реализации (см. литературу: Qiu и др., 2026, arXiv:2607.11883; заметки в
`publications/research_notes/.../requential_coding.md`):

* **Учитель и студент** — два экземпляра :class:`~episynth.model.moirai.MoiraiModule` одинаковой
  конфигурации. Учитель обучается на реальных батчах из генератора и сглаживается EMA; студент
  обучается только на сэмплах из распределения учителя.
* **Оценка KL за шаг** — на батче, сэмплированном из EMA-учителя, несмещённая оценка
  :math:`\mathrm{KL}_i = \frac1{|B|}\sum[\log Q_i(x) - \log P_i^s(x)]` (в битах через
  :math:`\ln 2`).
* **Seedable-сэмплирование** — предложения студента должны регенерироваться по ключу
  (seed, step, index) для декодера; сейчас `Mixture.sample` использует глобальный RNG.
* **Iso-loss projection / KL-threshold** — периодический сброс учителя к студенту, чтобы KL не рос.

Кодовая длина: :math:`\mathbb{E}[L_\mathrm{req}] = C_\mathrm{setup} + \sum_i[\mathrm{KL}_i
+ 2\log_2(1+\mathrm{KL}_i) + \kappa]`, :math:`\kappa < 5.21`; на практике ведущий член — суммарный KL.
Оценщик зарегистрирован в реестре, но его прогон пока не реализован и вызывает
:class:`NotImplementedError`.
"""

from __future__ import annotations

from pathlib import Path

from ..generators.base import Generator
from ..model.moirai import MoiraiModule
from ..training import TrainConfig
from ..windows import WindowSpec
from .base import EpiplexityEstimator
from .registry import register


@register("requential")
class RequentialEstimator(EpiplexityEstimator):
    """Requential-оценка эпиплексии (код студента, реконструируемого по индексам сэмплов).

    Каркас: структура метода `run` и ожидаемые аргументы зафиксированы, но шаги requential-протокола
    (учитель-студент, KL, iso-loss projection) ещё не реализованы и бросают :class:`NotImplementedError`.
    """

    name = "requential"

    def __init__(
        self,
        teacher_ema: float = 0.1,
        student_ema: float = 0.01,
        projection_steps: tuple[int, ...] = (100, 150, 225),
        kl_scale: float = 1.0,
    ) -> None:
        """Создаёт скелет requential-оценщика.

        Args:
            teacher_ema: Постоянная времени EMA учителя (доля шага от текущего шага).
            student_ema: Постоянная времени EMA студента.
            projection_steps: Шаги, на которых делается iso-loss projection (сброс учителя к студенту).
            kl_scale: Масштабный множитель для KL-оценки (заглушка).
        """
        self.teacher_ema = teacher_ema
        self.student_ema = student_ema
        self.projection_steps = projection_steps
        self.kl_scale = kl_scale

    def run(
        self,
        model: MoiraiModule,
        generator: Generator,
        spec: WindowSpec,
        config: TrainConfig,
        out_dir: str | Path,
    ) -> dict[str, float]:
        """Заглушка: requential-прогон ещё не реализован.

        Raises:
            NotImplementedError: Всегда, пока не реализованы учитель-студент и KL.
        """
        raise NotImplementedError(
            "RequentialEstimator.run is not implemented yet. "
            "See src/episynth/estimators/requential.py and requential_coding.md for the protocol."
        )
