"""Synthetic series generators: register by name and general contract."""

from .arima import ARIMA
from .base import Generator, original_seeds
from .chaotic import Chaotic
from .ets import ETS
from .fbm import FBM
from .garch import GARCH
from .kernelsynth import KernelSynth
from .mixture import Mixture
from .ou import OU
from .sde import SDE
from .stepfunction import StepFunction
from .timesynth import TimeSynth
from .tsi import TSI
from .waveform import Waveform

GENERATORS: dict[str, type[Generator]] = {
    cls.name: cls
    for cls in (KernelSynth, ETS, Waveform, SDE, StepFunction, TSI, ARIMA, FBM, GARCH, Chaotic, TimeSynth, OU)
}
"""Generator classes by name; eleven names coincide with the register Mix, Don't Pick, `ou` - the benchmark with the well-known Bayesian forecast."""

__all__ = ["GENERATORS", "Generator", "Mixture", "original_seeds"]
