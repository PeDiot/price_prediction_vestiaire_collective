"""Description.

Test the estimators from the vc_ml library.
"""

import pytest

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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.neural_network import MLPRegressor

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

def test_errors_ridge_estimator(): 
    with pytest.raises(ValueError):
        est = RidgeEstimator(alpha=[.1, .8, -.02])
    with pytest.raises(ValueError):
        est = RidgeEstimator(alpha=[1, .82])

def test_tree_estimator(): 
    est = TreeEstimator(max_depth=[10, 50, None])
    assert isinstance(est, TreeEstimator)
    assert repr(est) == "TreeEstimator(max_depth=[10, 50, None], min_samples_split=[2], min_samples_leaf=[1], max_features=['auto'], ccp_alpha=[0.0])"
    init_est = est()
    assert type(init_est["estimator"]) == DecisionTreeRegressor
    assert list(init_est.keys()) == ["estimator", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "ccp_alpha"]

def test_errors_tree_estimator(): 
    with pytest.raises(ValueError):
        est = TreeEstimator(max_depth=[10, 50, 100.0])
    with pytest.raises(ValueError):
        est = TreeEstimator(min_samples_split=[-1, 1, 3, 5])
    with pytest.raises(ValueError):
        est = TreeEstimator(ccp_alpha=[-.01, .01, .1])
    with pytest.raises(ValueError):
        est = TreeEstimator(max_features=["sqrt", "log2", "exp"])

def test_rf_estimator(): 
    est = RFEstimator(n_estimators=[100, 500, 1000], max_features=["auto", "sqrt", "log2"])
    assert isinstance(est, RFEstimator)
    assert repr(est) == "RFEstimator(n_estimators=[100, 500, 1000], criterion=['squared_error'], max_depth=[None], min_samples_split=[2], min_samples_leaf=[1], max_features=['auto', 'sqrt', 'log2'], max_samples=[1.0], oob_score=[True])"
    init_est = est()
    assert type(init_est["estimator"]) == RandomForestRegressor
    assert list(init_est.keys()) == ["estimator", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "max_samples", "oob_score"]

def test_errors_rf_estimator():
    with pytest.raises(ValueError):
        est = RFEstimator(n_estimators=[10., 100, 1000])
        est = RFEstimator(n_estimators=[-100, 100])
    with pytest.raises(ValueError):
        est = RFEstimator(criterion=["gini"])
    with pytest.raises(ValueError):
        est = RFEstimator(min_samples_leaf=[2., 5, 10])
    with pytest.raises(ValueError):
        est = RFEstimator(max_samples=[.1, .5, 1.2])
    with pytest.raises(ValueError):
        est = RFEstimator(oob_score=["True"])

def test_gb_estimator(): 
    est = GBEstimator(learning_rate=[1., .1, .001])
    assert isinstance(est, GBEstimator)
    assert repr(est) == "GBEstimator(n_estimators=[100], min_samples_split=[2], min_samples_leaf=[1], max_depth=[3], loss=['squared_error'], learning_rate=[1.0, 0.1, 0.001], criterion=['friedman_mse'], tol=[0.001])"
    init_est = est()
    assert type(init_est["estimator"]) == GradientBoostingRegressor
    assert list(init_est.keys()) == ["estimator", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "loss", "learning_rate", "criterion", "tol"]

def test_errors_gb_estimator():
    with pytest.raises(ValueError):
        gb = GBEstimator(criterion=["friedman_mse", "absolute_error"])
    with pytest.raises(ValueError):
        gb = GBEstimator(loss=["mean_squared_error"])
    with pytest.raises(ValueError):
        gb = GBEstimator(learning_rate=[-.1, .1, .01])
        gb = GBEstimator(learning_rate=[.01, .1, 1])
    with pytest.raises(ValueError):
        gb = GBEstimator(tol=[-.001, .001])

def test_mlp_estimator(): 
    est = MLPEstimator(
        hidden_layer_sizes=[
            [100, 100], 
            [50, 100, 50], 
            [200,]
        ], 
        solver=["adam", "lbfgs"], 
        activation=["relu", "logistic"]
    )
    assert isinstance(est, MLPEstimator)
    assert repr(est) == "MLPEstimator(hidden_layer_sizes=[[100, 100], [50, 100, 50], [200]], max_iter=[200], activation=['relu', 'logistic'], solver=['adam', 'lbfgs'], learning_rate_init=[0.001])"
    init_est = est()
    assert type(init_est["estimator"]) == MLPRegressor
    assert list(init_est.keys()) == ["estimator", "hidden_layer_sizes", "max_iter", "activation", "solver", "learning_rate_init"]

def test_errors_mlp_estimator(): 
    with pytest.raises(ValueError):
        mlp = MLPEstimator(hidden_layer_sizes=[ 
            [100, 100,],
            [-10, 150, 60]
        ])
        mlp = MLPEstimator(hidden_layer_sizes=(
            [100, 100], 
            [200,]
        ))
        mlp = MLPEstimator(hidden_layer_sizes=[ 
            (100, 100), 
            (50, 100, 50), 
        ])
    with pytest.raises(ValueError):
        mlp = MLPEstimator(max_iter=[1000.0])
        mlp = MLPEstimator(max_iter=[-2000])
    with pytest.raises(ValueError):
        mlp = MLPEstimator(solver=["newton"])
        mlp = MLPEstimator(activation=["relu", "sigmoid"])
    with pytest.raises(ValueError): 
        mlp = MLPEstimator(learning_rate_init=[-.0001, .0001])









