"""Description.

Methods for model selection.

Example:

In [1]: from vc_ml import (
   ...: get_files_path,
   ...: get_cv_results,
   ...: get_best_estimator
   ...: )

In [2]: paths = get_files_path()

In [3]: cv_results = get_cv_results(files_path=paths)

In [4]: get_best_estimator(cv_results=cv_results)
Out[4]: 
{'best_estimator': Pipeline(steps=[('model', GradientBoostingRegressor(tol=0.001))]),
 'train_score': 0.5537342215324665,
 'test_score': 0.29123571363147027}

In [5]: get_best_estimator(cv_results=cv_results, criterion="train")
Out[5]: 
{'best_estimator': Pipeline(('model', DecisionTreeRegressor(max_features='auto'))]),
 'train_score': 0.9744985506246093,
 'test_score': -0.14087415022162816}
"""

from enum import Enum
from typing import Dict, List

import numpy as np 
import pandas as pd 
from os import listdir, pipe 
from pickle import load

from .data import BACKUP

class ModelDir(Enum): 
    DUM = "DummyRegressor/"
    LR = "LinearRegression/"
    RIDGE = "Ridge/"
    TREE = "DecisionTreeRegressor/"
    RF = "RandomForestRegressor/"
    GB = "GradientBoostingRegressor/"
    MLP = "MLPRegressor/"

def get_files_path() -> List: 
    """Retrieve cross-validation results for each fitted pipeline."""
    files_path = list() 
    for model_dir in ModelDir:
        dir_path = BACKUP + "models/" + model_dir.value 
        model_results = list()
        for file_name in listdir(dir_path): 
            files_path.append(dir_path + file_name) 
    return files_path

def _read_file(path: str): 
    """Read pickle file."""
    with open(path, "rb") as file: 
        data = load(file)
    return data 

def get_cv_results(files_path: List[str]) -> pd.DataFrame:
    """Return a data frame with all cross-validation results."""
    d = {
        "estimator": [], 
        "avg_train_score": [], 
        "avg_test_score": []
    }
    for path in files_path:
        cv_data = _read_file(path)
        d["estimator"].append( cv_data["estimator"][0] )
        d["avg_train_score"].append( np.mean(cv_data["train_score"]) )
        d["avg_test_score"].append( np.mean(cv_data["test_score"]) )
    return pd.DataFrame.from_dict(d) 

def get_best_estimator(
    cv_results: pd.DataFrame, 
    criterion: str = "test"
) -> Dict: 
    """Return the best estimator based on test or train score."""
    best_test_score = max(cv_results["avg_test_score"])
    best_train_score = max(cv_results["avg_train_score"])
    res = dict()
    if criterion == "test":
        res["best_estimator"] = cv_results.loc[ 
            cv_results["avg_test_score"] == best_test_score, 
            "estimator"
        ].values.tolist()[0]
        res["train_score"] = cv_results.loc[ 
            cv_results["avg_test_score"] == best_test_score, 
            "avg_train_score"
        ].values.tolist()[0]
        res["test_score"] = best_test_score
    if criterion == "train": 
        res["best_estimator"] = cv_results.loc[ 
            cv_results["avg_train_score"] == best_train_score, 
            "estimator"
        ].values.tolist()[0]
        res["train_score"] = best_train_score
        res["test_score"] = cv_results.loc[ 
            cv_results["avg_train_score"] == best_train_score, 
            "avg_test_score"
        ].values.tolist()[0]
    return res 