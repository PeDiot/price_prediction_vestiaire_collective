"""Description.

Test ModelTraining and train_models from the vc_ml library.
"""

import pytest

from .test_data import simulated_data

from vc_ml import (
    SplitData,
    LREstimator, 
    RidgeEstimator, 
    train_models, 
    ModelTraining,
    training,
)

from sklearn.linear_model import Ridge

@pytest.fixture
def simulated_train_data(simulated_data): 
    s = SplitData(data=simulated_data)
    X, y = s.get_feature_vector(), s.get_targets()
    y = s._make_dict_of_targets(y)
    return X, y

@pytest.fixture
def ridge_estimator():
    return Ridge()

@pytest.fixture
def ridge_params():
    return {"alpha": .01, "fit_intercept": True}

@pytest.fixture
def training(
    simulated_train_data, 
    ridge_estimator, 
    ridge_params
):
    X, y = simulated_train_data
    return ModelTraining(
        X=X, 
        y=y, 
        estimator=ridge_estimator, 
        params=ridge_params, 
        n_comp=None
    )

def test_training_instanciation(training):
    assert isinstance(training, ModelTraining)




