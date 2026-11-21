r"""Requential epiplexity estimator stub.

See Qiu et al. (2026), arXiv:2607.11883, and the project notes in
`publications/research_notes/.../requential_coding.md`.

* **Teacher and student:** two copies of :class:`~episynth.model.moirai.MoiraiModule`
  with the same architecture. The teacher trains on generator batches with EMA.
  The student trains on samples from the teacher's predictive distribution.
* **Per-step KL:** estimate it on a batch sampled from the EMA teacher with
  :math:`\mathrm{KL}_i = \frac{1}{|B|}\sum[\log Q_i(x)-\log P_i^s(x)]`.
  Divide by :math:`\ln 2` to obtain bits.
* **Seeded sampling:** the decoder must regenerate student proposals from the
  seed, step, and index. `Mixture.sample` currently uses the global RNG.
* **Iso-loss projection or KL threshold:** periodically reset the teacher to the
  student to limit KL growth.

Expected code length is :math:`\mathbb{E}[L_\mathrm{req}] = C_\mathrm{setup} +
\sum_i[\mathrm{KL}_i + 2\log_2(1+\mathrm{KL}_i) + \kappa]`, with :math:`\kappa < 5.21`.
The total KL is the leading term in practice. The registered stub raises
:class:`NotImplementedError` on execution.
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
    """Requential-assessment of epiplexity (code of the student reconstructed by sample indices).

    The framework: the structure of the `run` method and the expected arguments are fixed, but the
    steps of the requential protocol (teacher-student, KL, iso-loss projection) are not yet
    implemented and throw :class:`NotImplementedError`.
    """

    name = "requential"

    def __init__(
        self,
        teacher_ema: float = 0.1,
        student_ema: float = 0.01,
        projection_steps: tuple[int, ...] = (100, 150, 225),
        kl_scale: float = 1.0,
    ) -> None:
        """Creates the skeleton of a requential estimator.

        Args:
            teacher_ema: Time constant EMA teacher (share of step from current step).
            student_ema: Time constant EMA student.
            projection_steps: Steps by which iso-loss projection is made.
            kl_scale: Scale multiplier for the KL score (stub).
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
        """Stub: The requential run has not yet been implemented.

        Raises:
            NotImplementedError: Always, until implemented teacher-student and KL.
        """
        raise NotImplementedError(
            "RequentialEstimator.run is not implemented yet. "
            "See src/episynth/estimators/requential.py and requential_coding.md for the protocol."
        )
