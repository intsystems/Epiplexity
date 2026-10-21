"""Генераторы синтетических рядов: реестр по именам и общий контракт."""

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
"""Классы генераторов по имени; одиннадцать имён совпадают с реестром Mix, Don't Pick, `ou` — эталон с известным байесовским прогнозом."""

__all__ = ["GENERATORS", "Generator", "Mixture", "original_seeds"]
