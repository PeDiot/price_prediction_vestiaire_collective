"""Description.

Methods for model selection.

Example:

In [1]: from vc_ml import (
   ...: get_files_path,
   ...: get_cv_results,
   ...: get_best_estimator, 
   ...: save_best_estimator
   ...: )

In [2]: from random import shuffle

In [3]: paths = get_files_paths()

In [4] shuffle(paths)

In [4]: cv_results = get_cv_results(files_paths=paths[:10])

In [5]: cv_results
Out[5]: 
                                           estimator  avg_train_score  avg_test_score  avg_score
0  (([DecisionTreeRegressor(criterion='mse', max_...         0.904547       -0.212792   0.345877
1  (PCA(n_components=60), (DecisionTreeRegressor(...         0.243006        0.174527   0.208766
2  (PCA(n_components=60), ([DecisionTreeRegressor...         0.863105        0.169511   0.516308
3  (([DecisionTreeRegressor(criterion='friedman_m...         0.574464        0.251238   0.412851
4       (DecisionTreeRegressor(max_features='auto'))         0.974460       -0.927888   0.023286
5  (PCA(n_components=60), ([DecisionTreeRegressor...         0.385237        0.038034   0.211636
6  (PCA(n_components=40), ([DecisionTreeRegressor...         0.324534        0.206097   0.265315
7  (PCA(n_components=40), (DecisionTreeRegressor(...         0.681622        0.094684   0.388153
8  (PCA(n_components=60), (DecisionTreeRegressor(...         0.234338        0.197392   0.215865
9  (PCA(n_components=40), (DecisionTreeRegressor(...         0.271733        0.167686   0.219709

In [6]: get_best_estimator(cv_results=cv_results)
Out[6]: 
{'best_estimator': Pipeline(steps=[('model',
                  GradientBoostingRegressor(criterion='mse',
                                            loss='absolute_error', max_depth=20,
                                            min_samples_leaf=20,
                                            min_samples_split=15,
                                            n_estimators=750, tol=0.001))]),
 'train_score': 0.3558182825292002,
 'test_score': 0.3466649990038641,
 'avg_score': 0.3512416407665322}

In [7]: get_best_estimator(cv_results=cv_results, criterion="train")
Out[7]: 
{'best_estimator': Pipeline(steps=[('model', DecisionTreeRegressor(max_features='auto'))]),
 'train_score': 0.9744604159492696,
 'test_score': -0.9278878299719601,
 'avg_score': 0.023286292988654755}

In [8]: get_best_estimator(cv_results=cv_results, criterion="train_test")
Out[8]: 
{'best_estimator': Pipeline(steps=[('pca', PCA(n_components=80)),
                 ('model',
                  GradientBoostingRegressor(min_samples_split=5,
                                            n_estimators=1000, tol=0.001))]),
 'train_score': 0.9744604159492696,
 'test_score': 0.23604312714836778,
 'avg_score': 0.5532353841625166}
"""

from enum import Enum
from typing import Dict, List

import numpy as np 
import pandas as pd 
from os import listdir
from pickle import load, dump

from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline

from .data import BACKUP, read_data
from .training import CPU_COUNT

class ModelDir(Enum): 
    DUM = "DummyRegressor/"
    LR = "LinearRegression/"
    RIDGE = "Ridge/"
    TREE = "DecisionTreeRegressor/"
    RF = "RandomForestRegressor/"
    GB = "GradientBoostingRegressor/"
    MLP = "MLPRegressor/"

def get_files_paths() -> List: 
    """Retrieve cross-validation results for each fitted pipeline."""
    files_paths = list() 
    for model_dir in ModelDir:
        dir_path = "models/" + model_dir.value 
        model_results = list()
        for file_name in listdir(BACKUP+dir_path): 
            if file_name[-3:] != "pkl":
                raise ValueError("Only pickle files are accepted.")
            else:
                files_paths.append(dir_path + file_name) 
    return files_paths

def get_cv_results(files_paths: List[str]) -> pd.DataFrame:
    """Return a data frame with all cross-validation results."""
    d = {
        "estimator": [], 
        "avg_train_score": [], 
        "avg_test_score": [], 
        "avg_score": [] 
    }

    def _process(path: str): 
        cv_data = read_data(file_path=path)
        est = cv_data["estimator"][0]
        train_score = np.mean(cv_data["train_score"])
        test_score = np.mean(cv_data["test_score"]) 
        score = np.mean([train_score, test_score])
        d["estimator"].append(est)
        d["avg_train_score"].append(train_score)
        d["avg_test_score"].append(test_score)
        d["avg_score"].append(score)
        return pd.DataFrame.from_dict(d)
    
    cv_res_list = Parallel(
            n_jobs=CPU_COUNT-2
        )(
            delayed(_process)(path)
            for path in files_paths
        ) 
    
    return pd.concat(
        objs=cv_res_list, 
        axis=0
    ).reset_index(drop=True)

def get_best_estimator(
    cv_results: pd.DataFrame, 
    criterion: str = "test"
) -> Dict: 
    """Return the best estimator based on test or train score."""
    best_test_score = max(cv_results["avg_test_score"])
    best_train_score = max(cv_results["avg_train_score"])
    best_avg_score = max(cv_results["avg_score"])

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
        res["avg_score"] = cv_results.loc[ 
            cv_results["avg_test_score"] == best_test_score, 
            "avg_score"
        ].values.tolist()[0]
    elif criterion == "train": 
        res["best_estimator"] = cv_results.loc[ 
            cv_results["avg_train_score"] == best_train_score, 
            "estimator"
        ].values.tolist()[0]
        res["train_score"] = best_train_score
        res["test_score"] = cv_results.loc[ 
            cv_results["avg_train_score"] == best_train_score, 
            "avg_test_score"
        ].values.tolist()[0]
        res["avg_score"] = cv_results.loc[ 
            cv_results["avg_train_score"] == best_train_score, 
            "avg_score"
        ].values.tolist()[0]
    elif criterion == "train_test": 
        res["best_estimator"] = cv_results.loc[ 
            cv_results["avg_score"] == best_avg_score, 
            "estimator"
        ].values.tolist()[0]
        res["train_score"] = cv_results.loc[ 
            cv_results["avg_train_score"] == best_train_score, 
            "avg_train_score"
        ].values.tolist()[0]
        res["test_score"] = cv_results.loc[ 
            cv_results["avg_score"] == best_avg_score, 
            "avg_test_score"
        ].values.tolist()[0]
        res["avg_score"] = best_avg_score
    else: 
        raise ValueError("Criterion takes the following values: 'test', 'train', 'train_test'.")
    return res 

def save_best_estimator(best_estimator: Pipeline): 
    """Save best estimator in a backup directory."""
    path = BACKUP + "best_estimator.pkl"
    with open(path, "wb") as file: 
        dump(obj=best_estimator, file=file)

def load_best_estimator(): 
    """Load best estimator from backup directory."""
    path = BACKUP + "best_estimator.pkl"
    with open(path, "rb") as file: 
        est = load(file=file)
    return est 

