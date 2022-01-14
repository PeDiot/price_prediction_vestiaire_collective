"""Description.

Test model selection functions from the vc_ml library.
"""

import pytest

import numpy as np 
from pandas.core.frame import DataFrame

from os import remove
from os.path import exists
from pickle import dump

from typing import Dict

from sklearn.linear_model import (
    LinearRegression, 
    Ridge, 
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from vc_ml import (
    ModelDir, 
    BACKUP, 
    get_files_paths, 
    get_cv_results,
    get_best_estimator, 
    save_best_estimator,
    load_best_estimator
)

@pytest.fixture
def tree_cv_results():
    """Simulate cross-validation results for regression tree."""
    return {
        "fit_time": np.array([1.68622613, 1.67246294, 1.83367062, 1.66460729, 1.65506601]),
        "score_time": np.array([0.00790977, 0.00693893, 0.00631833, 0.00359249, 0.00694036]),
        "estimator": [
            Pipeline(steps=[("pca", PCA(n_components=60)),
                        ("model", DecisionTreeRegressor(max_features="auto"))]),
            Pipeline(steps=[("pca", PCA(n_components=60)),
                            ("model", DecisionTreeRegressor(max_features="auto"))]),
            Pipeline(steps=[("pca", PCA(n_components=60)),
                            ("model", DecisionTreeRegressor(max_features="auto"))]),
            Pipeline(steps=[("pca", PCA(n_components=60)),
                            ("model", DecisionTreeRegressor(max_features="auto"))]),
            Pipeline(steps=[("pca", PCA(n_components=60)),
                            ("model", DecisionTreeRegressor(max_features="auto"))])
        ],
        "test_score": np.array([-0.30365252, -3.64721366, -0.06795624, -1.86311601, -0.11469278]),
        "train_score": np.array([0.66043493, 0.80822139, 0.76565103, 0.13786295, 0.55582144])
    }

@pytest.fixture
def lr_cv_results():
    """Simulate cross-validation results for linear regression."""
    return {
        "fit_time": np.array([0.66210699, 0.66210437, 0.74986672, 0.67903352, 0.74384379]),
        "score_time": np.array([0.09072137, 0.09170365, 0.00792956, 0.07079434, 0.00997043]),
        "estimator": [
            Pipeline(steps=[("pca", PCA(n_components=60)), ("model", LinearRegression())]),
            Pipeline(steps=[("pca", PCA(n_components=60)), ("model", LinearRegression())]),
            Pipeline(steps=[("pca", PCA(n_components=60)), ("model", LinearRegression())]),
            Pipeline(steps=[("pca", PCA(n_components=60)), ("model", LinearRegression())]),
            Pipeline(steps=[("pca", PCA(n_components=60)), ("model", LinearRegression())])
        ],
        "test_score": np.array([0.15643771, 0.20167337, 0.06061945, 0.1559367 , 0.08229102]),
        "train_score": np.array([0.11798155, 0.11703337, 0.13262984, 0.11707063, 0.13256849])
    }

@pytest.fixture
def ridge_cv_results(): 
    """Simulate cross-validation results for ridge."""
    return {
        "fit_time": np.array([0.05073166, 0.04873657, 0.0432539 , 0.04767466, 0.04454756]),
        "score_time": np.array([0.00598431, 0.00399041, 0.01146841, 0.00946403, 0.00817943]),
        "estimator": [
            Pipeline(steps=[("model", Ridge())]),
            Pipeline(steps=[("model", Ridge())]),
            Pipeline(steps=[("model", Ridge())]),
            Pipeline(steps=[("model", Ridge())]),
            Pipeline(steps=[("model", Ridge())])
        ],
        "test_score": np.array([0.29752998, 0.39089363, 0.16111275, 0.32678599, 0.20631463]),
        "train_score": np.array([0.26180703, 0.25347416, 0.31479919, 0.25703045, 0.2878362 ])
    }

@pytest.fixture
def save_cv_results(
    tree_cv_results, 
    lr_cv_results, 
    ridge_cv_results
): 
    """Save simulated cross-validation results."""
    def _create_cv_file(
        cv_res: Dict, 
        file_name: str
    ):
        path = BACKUP+"tests/" + file_name
        if not exists(path):
            with open(path, "wb") as file: 
                dump(obj=cv_res, file=file)

    _create_cv_file(cv_res=lr_cv_results, file_name="lr_cv_results.pkl")
    _create_cv_file(cv_res=tree_cv_results, file_name="tree_cv_results.pkl")
    _create_cv_file(cv_res=ridge_cv_results, file_name="ridge_cv_results.pkl")
    
def test_model_dir():
    dir = ModelDir.GB 
    assert isinstance(dir, ModelDir)
    assert dir.value == "GradientBoostingRegressor/"

def test_error_get_files(): 
    """Create an example file which is not pickle."""
    dir = ModelDir.GB
    file_path = BACKUP + "models/" + dir.value + "demofile.txt"
    f = open(file_path, "w")
    f.write("Not pickle file.")
    f.close()
    with pytest.raises(ValueError):
        paths = get_files_paths()
    remove(path=file_path)

def test_get_cv_results(save_cv_results):
    """Test "get_cv_results" function on simulated cv results."""
    save_cv_results
    files_paths = [ 
        "tests/tree_cv_results.pkl", 
        "tests/lr_cv_results.pkl", 
        "tests/ridge_cv_results.pkl"
    ]
    cv_results = get_cv_results(files_paths)
    assert list(cv_results.columns) == [
        "estimator",
        "avg_train_score",
        "avg_test_score",
        "avg_score"
    ]
    assert cv_results.shape == (3, 4)
    assert type(cv_results) == DataFrame




    

    


