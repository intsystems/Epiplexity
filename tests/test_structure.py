"""Structural tests for the epimeter skeleton.

These assert that the *interfaces* exist with the right contracts. They never
call not-yet-implemented code (no fit/estimate/train/evaluate), so the suite is
always green on the skeleton.
"""

import inspect

import epimeter
from epimeter.base import BaseEstimator
from epimeter.models.base import BaseModel
from epimeter.models.registry import build
from epimeter.samplers.base import BaseSampler


def test_package_imports_and_version():
    assert isinstance(epimeter.__version__, str)
    assert epimeter.__version__.count(".") >= 1


def test_estimator_classes_are_subclasses_and_units_bits():
    for cls in (epimeter.Prequential, epimeter.Requential,
                epimeter.Bayesian, epimeter.Proxies):
        assert issubclass(cls, BaseEstimator)
        assert cls.units == "bits"
        assert hasattr(cls, "fit")
        assert hasattr(cls, "estimate")


def test_base_estimator_is_abstract():
    assert inspect.isabstract(BaseEstimator)


def test_base_model_contract():
    assert hasattr(BaseModel, "forward")
    assert hasattr(BaseModel, "save")
    assert hasattr(BaseModel, "load")
    assert callable(BaseModel.count_parameters)


def test_base_sampler_contract():
    assert hasattr(BaseSampler, "sample")


def test_registry_empty_build_raises_not_implemented():
    # No observer registered yet; building must fail loudly, not silently.
    try:
        build("mlp")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("build('mlp') should raise NotImplementedError while empty")
