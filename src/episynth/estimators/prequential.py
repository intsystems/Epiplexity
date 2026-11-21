r"""Estimate epiplexity from prequential loss.

The shared :class:`episynth.training.Trainer` writes pre-update window
losses to `window_loss.f32` and final EMA losses on held-out windows to
`heldout_ema_1.f32`. After training, :func:`episynth.analysis.run_area`
computes the area and its code-length components.

Let :math:`l_i` be the window loss in the original series units and
:math:`L_{\mathrm{fin}}` be the final checkpoint's mean held-out loss
for the selected weight variant. The quantities in nats are:

* Total prequential code length :math:`L = \sum_i l_i`.
* Model part :math:`A = \sum_i l_i - N L_{\mathrm{fin}}`.
* Data part :math:`H = L - A`.

The current protocol uses final held-out loss as the baseline. Earlier
runs used the mean loss near the end of training. Divide by :math:`\ln 2`
to convert nats to bits.
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
    """Estimate the model and data parts from the prequential training run.

    Use the shared :class:`~episynth.training.Trainer`, then read
    `window_loss.f32` and the selected final held-out losses using
    :func:`~episynth.analysis.run_area`. Return code lengths in bits.
    """

    name = "prequential"

    def __init__(self, variant: str = "ema") -> None:
        r"""Select the final weight variant.

        Args:
            variant: Use `ema` or `raw` for the held-out baseline.
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
        """Performs a prequential run and returns the score in bits.

        Args:
            model: Observer model.
            generator: Row generator.
            spec: The shape of the task.
            config: Parameters run.
            out_dir: Run directory.

        Returns:
            dict[str, float]: ``model_part_bits``, ``data_part_bits``, ``total_bits`` and
                ``n_windows``.
        """
        out = Path(out_dir)
        Trainer(model, generator, spec, config, out).run()
        return self.from_run_dir(out)

    def from_run_dir(self, out_dir: str | Path) -> dict[str, float]:
        """Compute the estimate from a completed `Trainer` run using `analysis.run_area`.

        Args:
            out_dir: Run directory with `window_loss.f32` and `heldout_<variant>_1.f32`.

        Returns:
            dict[str, float]: `model_part_bits`, `data_part_bits`, `total_bits`,
            and `n_windows`.
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
