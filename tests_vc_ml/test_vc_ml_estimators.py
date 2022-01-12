"""Description.

Test the estimators from the "vc_ml" library.
"""

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

def test_tree_estimator(): 
    est = TreeEstimator(max_depth=[10, 50, None])
    assert isinstance(est, TreeEstimator)
    assert repr(est) == "TreeEstimator(max_depth=[10, 50, None], min_samples_split=[2], min_samples_leaf=[1], max_features=['auto'], ccp_alpha=[0.0])"
    init_est = est()
    assert type(init_est["estimator"]) == DecisionTreeRegressor
    assert list(init_est.keys()) == ["estimator", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "ccp_alpha"]

def test_rf_estimator(): 
    est = RFEstimator(n_estimators=[100, 500, 1000], max_features=["auto", "sqrt", "log2"])
    assert isinstance(est, RFEstimator)
    assert repr(est) == "RFEstimator(n_estimators=[100, 500, 1000], criterion=['squared_error'], max_depth=[None], min_samples_split=[2], min_samples_leaf=[1], max_features=['auto', 'sqrt', 'log2'], max_samples=[1.0], oob_score=[True])"
    init_est = est()
    assert type(init_est["estimator"]) == RandomForestRegressor
    assert list(init_est.keys()) == ["estimator", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "max_samples", "oob_score"]

def test_gb_estimator(): 
    est = GBEstimator(learning_rate=[1., .1, .001])
    assert isinstance(est, GBEstimator)
    assert repr(est) == "GBEstimator(n_estimators=[100], min_samples_split=[2], min_samples_leaf=[1], max_depth=[3], loss=['squared_error'], learning_rate=[1.0, 0.1, 0.001], criterion=['friedman_mse'], tol=[0.001])"
    init_est = est()
    assert type(init_est["estimator"]) == GradientBoostingRegressor
    assert list(init_est.keys()) == ["estimator", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "loss", "learning_rate", "criterion", "tol"]

def test_mlp_estimator(): 
    est = MLPEstimator(
        hidden_layer_sizes=[
            [100, 100], 
            [50, 100, 50], 
            [200]
        ], 
        solver=["adam", "lbfgs"], 
        activation=["relu", "logistic"]
    )
    assert isinstance(est, MLPEstimator)
    assert repr(est) == "MLPEstimator(hidden_layer_sizes=[[100, 100], [50, 100, 50], [200]], max_iter=[200], activation=['relu', 'logistic'], solver=['adam', 'lbfgs'], learning_rate_init=[0.001])"
    init_est = est()
    assert type(init_est["estimator"]) == MLPRegressor
    assert list(init_est.keys()) == ["estimator", "hidden_layer_sizes", "max_iter", "activation", "solver", "learning_rate_init"]









