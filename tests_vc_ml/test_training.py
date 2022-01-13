"""Description.

Test ModelTraining and train_models from the vc_ml library.
"""

from multiprocessing import Value
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

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_validate

from sklearn.linear_model import Ridge

@pytest.fixture
def simulated_train_data(simulated_data): 
    s = SplitData(data=simulated_data)
    X, y = s.get_feature_vector(), s.get_targets()
    y = s._make_dict_of_targets(y)
    return X, y["price"]

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

def test_errors_training(
    simulated_data, 
    ridge_estimator, 
    ridge_params
):
    X = simulated_data.drop(
        labels=["num_likes", "price", "we_love_tag", "lprice"], 
        axis=1
    )
    y = simulated_data["price"]
    with pytest.raises(ValueError):
        ModelTraining(
            X=X, y=y, 
            estimator=ridge_estimator, 
            params=ridge_params
        )
    X = X.values
    y = y.values
    with pytest.raises(ValueError):
        ModelTraining(
            X=X, y=y, 
            estimator=ridge_estimator, 
            params=ridge_params,
            n_comp=60.
        )
    with pytest.raises(ValueError):
        ModelTraining(
            X=X, y=y, 
            estimator=ridge_estimator, 
            params=[[.01], [True]],
            n_comp=60
        )

def test_check_backup(training):
    """Test whether training object has already been backuped."""
    assert type(int(training._key)) == int
    assert training.check_backup() == False 

def test_init_model(training): 
    """Test model initialisation."""
    init_model = training._init_model()
    params = {
        "alpha": 0.01,
        "copy_X": True,
        "fit_intercept": True,
        "max_iter": None,
        "normalize": "deprecated",
        "positive": False,
        "random_state": None,
        "solver": "auto",
        "tol": 0.001
    }
    assert init_model.get_params() == params
    assert init_model.__class__ == Ridge

def test_init_pipeline(training): 
    """Test pipeline initialisation."""
    p = training.init_pipeline()
    assert isinstance(p, Pipeline)

def test_cross_val_fit(training):
    """Test cross-validation function."""
    p = training.init_pipeline()
    cv = training.cross_val_fit(p, cv=3) 
    assert type(cv) == dict 
    assert list(cv.keys()) == ["fit_time", "score_time", "estimator", "test_score", "train_score"]