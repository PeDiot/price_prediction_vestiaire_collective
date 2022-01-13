"""Description.

Automate training process using pipelines, grid search and cross-validation.
"""

from multiprocessing import Value
import numpy as np 
import pandas as pd
from pickle import dump 
import os
from random import shuffle

from rich.table import Table 
from rich import print

from typing import (
    List, 
    Dict, 
    Optional, 
    Tuple, 
)

from joblib import Parallel, delayed

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

    In [14]: training.display_results(cv_results)
    Out[14]: {'avg_train_score': 0.12040312046404515, 'avg_test_score': 0.1269623978424663}

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

        if type(self.X) != np.ndarray or type(self.y) != np.ndarray:
            raise ValueError("X and y must be numpy arrays.")
        
        if type(self.params) != dict:
            raise ValueError("'params' must be a dictionnary.")

        if self.n_comp is not None: 
            if self.n_comp <= 0 or type(self.n_comp) != int:
                raise ValueError("The number of principal components must be a striclty positive integer.")

        self._estimator_name = self.estimator.__class__.__name__
        self._backup_dir = BACKUP + "models/" + str(self._estimator_name) + "/"
        self._key = self._get_hash_key() 

    def __repr__(self) -> str:
        return f"Training(estimator={repr(self.estimator)}, params={self.params}, n_comp={self.n_comp})"
    
    def _get_hash_key(self) -> str:
        """Return a unique key to identify the trained model.""" 
        return str(
            hash(
                f"{self._estimator_name},{self.params},{self.n_comp}"
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
        if cv <= 0 or type(cv) != int:
            raise ValueError("The number of folds must be a strictly positive integer.")
        return cross_validate(
            estimator=p, 
            X=self.X, 
            y=self.y, 
            cv=cv, 
            return_estimator=True,
            return_train_score=True, 
            verbose=3, 
            n_jobs=max(5, cv)
        )

    def check_backup(self) -> bool:
        """Check whether object has already been tested and backuped."""
        if os.path.isfile(self._backup_dir + self._key + ".pkl"): 
            return True 
        return False  

    def display_results(
        self, 
        cv_results: Dict, 
    ):
        """Show average train and test scores from cross-validation."""
        return {
            "avg_train_score": np.mean(cv_results["train_score"]), 
            "avg_test_score": np.mean(cv_results["test_score"])
        }

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

    def _process(est, params: List): 
        """Cross-validation process to parallelize."""
        if comp_grid is not None: 
            for n_comp in comp_grid:
                training = ModelTraining(
                    X=X_tr, 
                    y=y_tr, 
                    estimator=est,
                    params=params, 
                    n_comp=n_comp
                )
                print(training)
                if not training.check_backup():
                    p = training.init_pipeline()
                    cv_results = training.cross_val_fit(p, cv)
                    training.save_cv_results(cv_results)
                    print( training.display_results(cv_results) )
                else: 
                    print("Model already trained.")
        else:
            training = ModelTraining(
                    X=X_tr, 
                    y=y_tr, 
                    estimator=est,
                    params=params
                )
            print(training)
            if not training.check_backup():
                p = training.init_pipeline()
                cv_results = training.cross_val_fit(p, cv)
                training.save_cv_results(cv_results)
                print( training.display_results(cv_results) )
            else: 
                print("Model already trained.")

    n_jobs = CPU_COUNT - max(cv, 5) - 1
    estimators, grids = config.init_models()
    for estimator, grid in zip(estimators, grids):
        estimator = [ estimator for _ in range(len(grid)) ]
        grid = list(ParameterGrid(grid))
        shuffle(grid)
        Parallel(
            n_jobs=n_jobs
        )(
            delayed(_process)(est, params)
            for (est, params) in zip(estimator, grid)
        ) 


        
