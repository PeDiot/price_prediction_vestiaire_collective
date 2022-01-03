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

In [5]: best_estimator, best_score = get_best_pipeline(pipeline_scores)

In [6]: best_estimator
Out[6]: 
Pipeline(steps=[('enc',
                 OneHotEncoder(drop='first', handle_unknown='ignore',
                               sparse=False)),
                ('model', GradientBoostingRegressor(tol=0.001))])
"""
from enum import Enum
from typing import Dict

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

def get_cv_results(model_dir: ModelDir) -> Dict: 
    """Retrieve cross-validation results for each fitted pipeline."""
    dir_path = BACKUP + "models/" + model_dir.value 
    cv_results = list()
    for file_name in listdir(dir_path): 
        file_path = dir_path + file_name
        with open(file_path, "rb") as file: 
            cv_results.append(load(file))
    return cv_results

def get_pipeline_scores(cv_results: Dict) -> pd.DataFrame:
    """Return a dictionnary containing the fitted pipelines, average train and test scores."""
    res = dict() 
    res["pipeline"] = [
        result["estimator"][0]
        for result in cv_results
    ]
    res["avg_train_score"] = [
        np.mean(result["train_score"])
        for result in cv_results
    ]
    res["avg_test_score"] = [
        np.mean(result["test_score"])
        for result in cv_results
    ]
    return pd.DataFrame.from_dict(res) 

def get_best_pipeline(pipeline_scores: pd.DataFrame): 
    """Return the best pipeline based on test score."""
    best_score = max(pipeline_scores["avg_test_score"])
    best_estimator = pipeline_scores.loc[ 
        pipeline_scores["avg_test_score"] == best_score, 
        "pipeline"
    ].values.tolist()[0]
    return best_estimator, best_score