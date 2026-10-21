r"""Prequential-оценщик эпиплексии.

Обёртка над общим движком :class:`episynth.training.Trainer`: тот уже выполняет prequential-прогон
(пишет лосс каждого окна до шага по нему в `window_loss.f32`). После прогона оценщик пересчитывает
покомпонентную кодовую длину в битах по тем же файлам.

Формулы (в натах, лосс окна `l_i` в исходных единицах ряда):

* prequential-длина кода: :math:`L = \sum_i l_i`;
* оценка model part (площадь над финальным лоссом): :math:`S = \sum_i \max(0, l_i - l_\mathrm{final})`,
  где :math:`l_\mathrm{final}` — средний лосс последних окон (конечная модель);
* оценка data part: :math:`H = L - S`.

Перевод в биты — делением на :math:`\ln 2`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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

    Использует общий :class:`Trainer` в его prequential-режиме по умолчанию, затем читает
    `window_loss.f32` из каталога прогона и считает разложение на model/data part в битах.
    """

    name = "prequential"

    def __init__(self, final_window_fraction: float = 0.01) -> None:
        r"""Создаёт оценщик.

        Args:
            final_window_fraction: Доля последних окон прогона, чей средний лосс принимается за
                финальный лосс модели :math:`l_\mathrm{final}`. Должна быть в (0, 1].
        """
        if not 0 < final_window_fraction <= 1:
            raise ValueError("final_window_fraction must be in (0, 1].")
        self.final_window_fraction = final_window_fraction

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
        return self.from_run_dir(out, config)

    def from_run_dir(self, out_dir: Path, config: TrainConfig) -> dict[str, float]:
        """Считает оценку из готового каталога прогона `Trainer`."""
        loss = np.memmap(out_dir / "window_loss.f32", dtype=np.float32, mode="r", shape=(config.n_windows,))
        n = config.n_windows
        k = max(1, int(round(self.final_window_fraction * n)))
        final = float(loss[-k:].mean())
        total_nats = float(loss.sum())
        model_nats = float(np.maximum(loss - final, 0.0).sum())
        data_nats = total_nats - model_nats
        return {
            "model_part_bits": model_nats / _LN2,
            "data_part_bits": data_nats / _LN2,
            "total_bits": total_nats / _LN2,
            "n_windows": float(n),
        }
