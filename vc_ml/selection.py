"""Description.

Methods for model selection.

Example:

In [1]: from vc_ml import (
   ...: get_files_path,
   ...: get_cv_results,
   ...: get_best_estimator, 
   ...: save_best_estimator
   ...: )

In [2]: paths = get_files_path()

In [3]: cv_results = get_cv_results(files_path=paths)

In [4]: get_best_estimator(cv_results=cv_results)
Out[4]: 
{'best_estimator': Pipeline(steps=[('model',
                  GradientBoostingRegressor(criterion='mse',
                                            loss='absolute_error', max_depth=20,
                                            min_samples_leaf=20,
                                            min_samples_split=15,
                                            n_estimators=750, tol=0.001))]),
 'train_score': 0.3558182825292002,
 'test_score': 0.3466649990038641,
 'avg_score': 0.3512416407665322}

In [5]: get_best_estimator(cv_results=cv_results, criterion="train")
Out[5]: 
{'best_estimator': Pipeline(steps=[('model', DecisionTreeRegressor(max_features='auto'))]),
 'train_score': 0.9744604159492696,
 'test_score': -0.9278878299719601,
 'avg_score': 0.023286292988654755}

In [6]: get_best_estimator(cv_results=cv_results, criterion="train_test")
Out[6]: 
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

from .data import BACKUP
from .training import CPU_COUNT

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
        "avg_test_score": [], 
        "avg_score": [] 
    }

    def _process(path: str): 
        cv_data = _read_file(path)
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
            n_jobs=CPU_COUNT-1
        )(
            delayed(_process)(path)
            for path in files_path
        ) 
    
    return pd.concat(
        objs=cv_res_list, 
        axis=0
    ) 

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

