"""Description.

Methods for model selection.

Example:

In [1]: from vc_ml import (
   ...:     ModelDir,
   ...:     get_cv_results,
   ...:     get_pipeline_scores,
   ...:     get_best_pipeline,
   ...: )

In [2]: cv_results = get_cv_results(model_dir=ModelDir.GB)

In [3]: pipeline_scores = get_pipeline_scores(cv_results)

In [4]: pipeline_scores
Out[4]: 
                                             pipeline  avg_train_score  avg_test_score
0   (OneHotEncoder(drop='first', handle_unknown='i...         0.693870        0.201618
1   (OneHotEncoder(drop='first', handle_unknown='i...         0.643984        0.037790
2   (OneHotEncoder(drop='first', handle_unknown='i...         0.692820        0.057223
3   (OneHotEncoder(drop='first', handle_unknown='i...         0.521145        0.146651
4   (OneHotEncoder(drop='first', handle_unknown='i...         0.772312        0.246166
..                                                ...              ...             ...
79  (OneHotEncoder(drop='first', handle_unknown='i...         0.545773        0.037887
80  (OneHotEncoder(drop='first', handle_unknown='i...         0.636655        0.132737
81  (OneHotEncoder(drop='first', handle_unknown='i...         0.714508        0.155943
82  (OneHotEncoder(drop='first', handle_unknown='i...         0.664312        0.170276
83  (OneHotEncoder(drop='first', handle_unknown='i...         0.587345        0.106898

[84 rows x 3 columns]

In [5]: get_best_pipeline(pipeline_scores)
Out[5]: 
(Pipeline(steps=[('enc',
                  OneHotEncoder(drop='first', handle_unknown='ignore',
                                sparse=False)),
                 ('model',
                  GradientBoostingRegressor(min_samples_leaf=15,
                                            min_samples_split=10,
                                            n_estimators=250, tol=0.001))]),
 0.31228587466370106)

In [6]: get_best_pipeline(pipeline_scores, criterion="train")
Out[6]: 
(Pipeline(steps=[('enc',
                  OneHotEncoder(drop='first', handle_unknown='ignore',
                                sparse=False)),
                 ('pca', PCA(n_components=80)),
                 ('model',
                  GradientBoostingRegressor(min_samples_leaf=2,
                                            min_samples_split=15,
                                            n_estimators=1000, tol=0.001))]),
 0.8575084837322704)
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