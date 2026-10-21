r"""Prequential-оценщик эпиплексии.

Обёртка над общим движком :class:`episynth.training.Trainer`: тот уже выполняет prequential-прогон
(пишет лосс каждого окна до шага по нему в `window_loss.f32` и лоссы отложенного набора на
последнем чекпойнте в `heldout_ema_1.f32`). После прогона оценщик пересчитывает покомпонентную
кодовую длину в битах через :func:`episynth.analysis.run_area`.

Формулы (в натах, лосс окна `l_i` в исходных единицах ряда, `L_fin` — средний лосс окна на
отложенном наборе у последнего чекпойнта для варианта весов `ema`):

* prequential-длина кода: :math:`L = \sum_i l_i`;
* оценка model part (площадь над финальным лоссом): :math:`A = \sum_i l_i - N L_\mathrm{fin}`;
* оценка data part: :math:`H = L - A`.

Площадь, в отличие от ранней версии (средний лосс хвоста прогона), использует отложенный набор,
как в основном протоколе эксперимента; это и есть финальный лосс конечной модели. Перевод в биты —
делением на :math:`\ln 2`.
"""

from __future__ import annotations

from pathlib import Path

from ..analysis import run_area
from ..generators.base import Generator
from ..model.moirai import MoiraiModule
from ..training import TrainConfig, Trainer
from ..windows import WindowSpec
from .base import EpiplexityEstimator
from .registry import register

_LN2 = 0.6931471805599453


@register("prequential")
class PrequentialEstimator(EpiplexityEstimator):
    """Prequential-оценка эпиплексии (площадь над кривой лосса).

    Использует общий :class:`~episynth.training.Trainer` в его prequential-режиме по умолчанию, затем читает
    `window_loss.f32` и `heldout_ema_1.f32` из каталога прогона через :func:`~episynth.analysis.run_area` и считает
    разложение на model/data part в битах.
    """

    name = "prequential"

    def __init__(self, variant: str = "ema") -> None:
        r"""Создаёт оценщик.

        Args:
            variant: Вариант весов финального лосса в :func:`~episynth.analysis.run_area`: `ema` или `raw`.
        """
        if variant not in ("ema", "raw"):
            raise ValueError("variant must be 'ema' or 'raw'.")
        self.variant = variant

    def run(
        self,
        model: MoiraiModule,
        generator: Generator,
        spec: WindowSpec,
        config: TrainConfig,
        out_dir: str | Path,
    ) -> dict[str, float]:
        """Выполняет prequential-прогон и возвращает оценку в битах.

        Args:
            model: Модель-наблюдатель.
            generator: Генератор рядов.
            spec: Форма задачи.
            config: Параметры прогона.
            out_dir: Каталог прогона.

        Returns:
            dict[str, float]: ``model_part_bits``, ``data_part_bits``, ``total_bits`` и
                ``n_windows``.
        """
        out = Path(out_dir)
        Trainer(model, generator, spec, config, out).run()
        return self.from_run_dir(out)

    def from_run_dir(self, out_dir: str | Path) -> dict[str, float]:
        """Считает оценку из готового каталога прогона `Trainer` через `analysis.run_area`.

        Args:
            out_dir: Каталог прогона с `window_loss.f32` и `heldout_<variant>_1.f32`.

        Returns:
            dict[str, float]: ``model_part_bits``, ``data_part_bits``, ``total_bits`` и
                ``n_windows`` (площадь в битах, оставшиеся в натах несущественны для разложения).
        """
        area = run_area(out_dir, self.variant)
        total_nats = area.sum_loss
        model_nats = area.area
        data_nats = total_nats - model_nats
        return {
            "model_part_bits": model_nats / _LN2,
            "data_part_bits": data_nats / _LN2,
            "total_bits": total_nats / _LN2,
            "n_windows": float(area.n_windows),
        }
