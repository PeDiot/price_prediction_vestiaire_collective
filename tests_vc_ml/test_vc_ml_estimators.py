"""Description.

Test the estimators from the "vc_ml" library.
"""

import pytest 
import coverage

from vc_ml import (
    DummyEstimator,
    LREstimator, 
    RidgeEstimator, 
    TreeEstimator, 
    RFEstimator, 
    GBEstimator, 
    MLPEstimator, 
)

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import (
    LinearRegression, 
    Ridge, 
)

def test_dummy_estimator():
    est = DummyEstimator()
    assert isinstance(est, DummyEstimator)
    assert repr(est) == "DummyEstimator(strategy=['mean'])"
    init_est = est()
    assert type(init_est["estimator"]) == DummyRegressor
    assert list(init_est.keys()) == ["estimator", "strategy"]

def test_lr_estimator(): 
    est = LREstimator()
    assert isinstance(est, LREstimator)
    assert repr(est) == "LREstimator(fit_intercept=[True])"
    init_est = est()
    assert type(init_est["estimator"]) == LinearRegression
    assert list(init_est.keys()) == ["estimator", "fit_intercept"]

def test_ridge_estimator(): 
    est = RidgeEstimator()
    assert isinstance(est, RidgeEstimator)
    assert repr(est) == "RidgeEstimator(alpha=[1.0], fit_intercept=[True])"
    init_est = est()
    assert type(init_est["estimator"]) == Ridge
    assert list(init_est.keys()) == ["estimator", "alpha", "fit_intercept"]





