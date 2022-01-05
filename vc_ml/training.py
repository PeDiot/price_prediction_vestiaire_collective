"""Description.

Automate training process using pipelines, grid search and cross-validation.
"""

import numpy as np 
import pandas as pd
from pickle import dump 
import os

from rich.table import Table 
from rich import print

from typing import (
    List, 
    Dict, 
    Optional, 
    Tuple, 
)

from sklearn.decomposition import PCA

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import (
    ParameterGrid, 
    cross_validate, 
)

from .data import (
    Target, 
    BACKUP
)

from .config import Config

CPU_COUNT = os.cpu_count()
         
class ModelTraining: 
    """Fit model using pipeline and cross-validation after PCA.
    
    Example: 

    In [1]: from vc_ml import (
    ...: load_feature_vector,
    ...: load_target,
    ...: ModelTraining,
    ...: Target
    ...: )

    In [2]: from sklearn.linear_model import Ridge

    In [3]: X_tr = load_feature_vector(file_name="train.pkl")

    In [4]: y_tr = load_target(file_name="train.pkl")

    In [5]: est = Ridge()

    In [6]: params = {"alpha": .01, "fit_intercept": True}

    In [7]: training = ModelTraining(
    ...: X=X_tr,
    ...: y=y_tr,
    ...: estimator=est,
    ...: params=params,
    ...: n_comp=60
    ...: )

    In [8]: training
    Out[8]: 
    Training(X=[[1 0 0 ... 0 1 0]
    [1 0 0 ... 0 1 0]
    [1 0 0 ... 0 1 0]
    ...
    [1 0 0 ... 0 1 0]
    [1 0 0 ... 0 0 1]
    [1 0 0 ... 0 1 0]], y=[ 126.    450.    470.   ...  170.    346.17 1800.  ], estimator=Ridge(), params={'alpha': 0.01, 'fit_intercept': True}, n_comp=60)

    In [9]: training._key
    Out[9]: '-5041755674150669196'

    In [10]: training.check_backup()
    Out[10]: False

    In [11]: pipe = training.init_pipeline()

    In [12]: pipe
    Out[12]: Pipeline(steps=[('pca', PCA(n_components=60)), ('model', Ridge(alpha=0.01))])

    In [13]: cv_results = training.cross_val_fit(p=pipe, cv=5)
    [Parallel(n_jobs=7)]: Using backend LokyBackend with 7 concurrent workers.
    [CV] END ..................., score=(train=0.114, test=0.157) total time=   0.2s
    [CV] END ..................., score=(train=0.111, test=0.189) total time=   0.2s
    [Parallel(n_jobs=7)]: Done   2 out of   5 | elapsed:    2.2s remaining:    3.4s
    [CV] END ..................., score=(train=0.116, test=0.153) total time=   0.2s
    [CV] END ..................., score=(train=0.130, test=0.080) total time=   0.2s
    [CV] END ..................., score=(train=0.138, test=0.069) total time=   0.2s
    [Parallel(n_jobs=7)]: Done   5 out of   5 | elapsed:    2.2s finished

    In [14]: cv_results
    Out[14]: 
    {'fit_time': array([0.2692802 , 0.25930619, 0.26628733, 0.26429367, 0.25332499]),
    'score_time': array([0.00598407, 0.00698256, 0.00498819, 0.00598168, 0.00597906]),
    'estimator': [Pipeline(steps=[('pca', PCA(n_components=60)), ('model', Ridge(alpha=0.01))]),
    Pipeline(steps=[('pca', PCA(n_components=60)), ('model', Ridge(alpha=0.01))]),
    Pipeline(steps=[('pca', PCA(n_components=60)), ('model', Ridge(alpha=0.01))]),
    Pipeline(steps=[('pca', PCA(n_components=60)), ('model', Ridge(alpha=0.01))]),
    Pipeline(steps=[('pca', PCA(n_components=60)), ('model', Ridge(alpha=0.01))])],
    'test_score': array([0.15260963, 0.18946449, 0.06892803, 0.15654169, 0.07997544]),
    'train_score': array([0.11551795, 0.11138411, 0.13817636, 0.11410483, 0.13049429])}

    In [15]: training.save_cv_results(cv_results)
    """

    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        estimator, 
        params: Dict,
        n_comp: Optional[int] = None 
    ):
        self.X, self.y = X, y 
        self.estimator = estimator
        self.params = params 
        self.n_comp = n_comp
        self._backup_dir = BACKUP + "models/" + str(self.estimator).split("(")[0] + "/"
        self._key = self._get_hash_key() 

    def __repr__(self) -> str:
        return f"Training(X={self.X}, y={self.y}, estimator={repr(self.estimator)}, params={self.params}, n_comp={self.n_comp})"
    
    def _get_hash_key(self) -> str:
        """Return a unique key to identify the trained model.""" 
        return str(
            hash(
                f"{self.estimator},{self.params},{self.n_comp}"
            )
        )

    def _init_model(self): 
        """Initialize model from estimator and params grid."""
        return self.estimator.set_params(**self.params)

    def init_pipeline(self) -> Pipeline: 
        """Return a pipeline with PCA as first step."""
        if self.n_comp is not None:
            steps = [
                (
                    "pca",
                    PCA(n_components=self.n_comp)
                ),
                (
                    "model",
                    self._init_model()
                ),
            ]
        else:
            steps = [
                (
                    "model", 
                    self._init_model()
                ),
            ]
        return Pipeline(steps=steps)

    def cross_val_fit(self,
        p: Pipeline, 
        cv: int
    ) -> Dict: 
        """Fit cross-validation with pipeline as estimator."""
        return cross_validate(
            estimator=p, 
            X=self.X, 
            y=self.y, 
            cv=cv, 
            return_estimator=True,
            return_train_score=True, 
            verbose=3, 
            n_jobs=CPU_COUNT-1
        )

    def check_backup(self) -> bool:
        """Check whether object has already been tested and backuped."""
        if os.path.isfile(self._backup_dir + self._key + ".pkl"): 
            return True 
        return False  

    def save_cv_results(
        self, 
        cv_results: Dict, 
    ):
        with open(self._backup_dir + self._key + ".pkl", "wb") as file:
            dump(obj=cv_results, file=file)

def train_models(
    X_tr: np.ndarray, 
    y_tr: np.ndarray,
    config: Config, 
    cv: float = 5, 
    comp_grid: Optional[list[int]] = None 
): 
    """Train and save multiple models using pipeline, grid search and cross validation."""
    results = {
        "pipeline": [], 
        "avg_train_score": [], 
        "avg_test_score": []
    }

    def _fill_results_dict(cv_results: Dict) -> Dict: 
        results["pipeline"].append(cv_results["estimator"][0])
        results["avg_train_score"].append( np.mean(cv_results["train_score"]) )
        results["avg_test_score"].append( np.mean(cv_results["test_score"]) )

    def _display_num_combinations(
        estimator, 
        param_grids: List, 
        comp_grid: Optional[List] = None
    ) -> str: 
        """Return a message with the number of combinations to test."""
        if comp_grid is not None: 
            msg = f"{len(comp_grid)*len(param_grids)} combinations to test for {estimator}."
        else: 
             msg = f"{len(param_grids)} combinations to test for {estimator}."
        return msg

    estimators, grids = config.init_models()
    for est, g in zip(estimators, grids):
        g = list(ParameterGrid(g))
        _display_num_combinations(
            estimator=est, 
            param_grids=g, 
            comp_grid=comp_grid
        )
        for params in g: 
            if comp_grid is not None: 
                for n_comp in comp_grid:
                    training = ModelTraining(
                        X=X_tr, 
                        y=y_tr, 
                        estimator=est,
                        params=params, 
                        n_comp=n_comp
                    )
                    if not training.check_backup():
                        p = training.init_pipeline()
                        print(f"pipeline: {p}")
                        cv_results = training.cross_val_fit(p=p, cv=cv)
                        training.save_cv_results(cv_results)
                        _fill_results_dict(cv_results)
            else:
                training = ModelTraining(
                        X=X_tr, 
                        y=y_tr, 
                        estimator=est,
                        params=params
                    )
                if not training.check_backup():
                    p = training.init_pipeline()
                    print(f"pipeline: {p}")
                    cv_results = training.cross_val_fit(p=p, cv=cv)
                    training.save_cv_results(cv_results)
                    _fill_results_dict(cv_results)
                        
    return results 


        
