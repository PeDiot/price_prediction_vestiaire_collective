"""Description.

Test the config script from the vc_ml library.
"""

import pytest

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

from vc_ml.estimators import ( 
    DummyEstimator, 
    LREstimator, 
    RidgeEstimator, 
    TreeEstimator, 
    RFEstimator, 
    GBEstimator, 
    MLPEstimator
)

from vc_ml.config import (
    Config, 
    create_config, 
    load_config
)

@pytest.fixture
def config():
    return Config(
        dum=DummyEstimator(), 
        lr=LREstimator(fit_intercept=[True]),
        ridge=RidgeEstimator(alpha=[1.0], fit_intercept=[True]),
        tree=TreeEstimator(max_depth=[None],
        min_samples_split=[2], min_samples_leaf=[1], max_features=['auto'], ccp_alpha=[0.0]), 
        rf=RFEstimator(n_estimators=[100], max_depth=[None], min_samples_split=[2], min_samples_leaf=[1], max_features=['auto'], max_samples=[1.0], oob_score=[True]),
        gb=GBEstimator(n_estimators=[100], min_samples_split=[2], min_samples_leaf=[1], max_depth=[3], loss=['squared_error'], learning_rate=[0.1], criterion=['friedman_mse'], tol=[0.001]),
        mlp=MLPEstimator(hidden_layer_sizes=[[100]], max_iter=[200], activation=['relu'], solver=['adam'], learning_rate_init=[0.001])
    )

def test_config_instanciation(config):
    assert isinstance(config, Config)

def test_config_repr(config):
    assert repr(config) == "Config(dum=DummyEstimator(strategy=['mean']), lr=LREstimator(fit_intercept=[True]), ridge=RidgeEstimator(alpha=[1.0], fit_intercept=[True]), tree=TreeEstimator(max_depth=[None], min_samples_split=[2], min_samples_leaf=[1], max_features=['auto'], ccp_alpha=[0.0]), rf=RFEstimator(n_estimators=[100], criterion=['squared_error'], max_depth=[None], min_samples_split=[2], min_samples_leaf=[1], max_features=['auto'], max_samples=[1.0], oob_score=[True]), gb=GBEstimator(n_estimators=[100], min_samples_split=[2], min_samples_leaf=[1], max_depth=[3], loss=['squared_error'], learning_rate=[0.1], criterion=['friedman_mse'], tol=[0.001]), mlp=MLPEstimator(hidden_layer_sizes=[[100]], max_iter=[200], activation=['relu'], solver=['adam'], learning_rate_init=[0.001]))"

def test_config_init_models(config): 
    models = config.init_models()
    assert type(models) == tuple 
    estimators, grids = models 
    estimator_classes = [est.__class__ for est in estimators]
    assert estimator_classes == [ 
        DummyRegressor, 
        LinearRegression, 
        Ridge, 
        DecisionTreeRegressor, 
        RandomForestRegressor, 
        GradientBoostingRegressor, 
        MLPRegressor,
    ]
    assert grids == [
        {
            'strategy': ['mean']
        },
        {
            'fit_intercept': [True]
        },
        {
            'alpha': [1.0],
            'fit_intercept': [True]
        },
        {
            'max_depth': [None],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['auto'],
            'ccp_alpha': [0.0]
        },
        {
            'n_estimators': [100],
            'max_depth': [None],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'max_features': ['auto'],
            'max_samples': [1.0],
            'oob_score': [True]
        },
        {
            'n_estimators': [100],
            'max_depth': [3],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'loss': ['squared_error'],
            'learning_rate': [0.1],
            'criterion': ['friedman_mse'],
            'tol': [0.001]
        },
        {
            'hidden_layer_sizes': [[100]],
            'max_iter': [200],
            'activation': ['relu'],
            'solver': ['adam'],
            'learning_rate_init': [0.001]
        }
    ]
        

def test_error_create_config(): 
    with pytest.raises(ValueError):
        create_config(file_name="config.pkl")

def test_errors_load_config():
    not_existing_file_name = "not_existing_config.yaml"
    with pytest.raises(ValueError):
        load_config(file_name=not_existing_file_name)
    with pytest.raises(ValueError): 
        load_config(file_name="config.csv")






