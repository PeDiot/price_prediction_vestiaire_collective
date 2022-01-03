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

    In [1]: est = LinearRegression()
    In [2]: param_grid = {'fit_intercept': True}
    In [3]: training = ModelTraining(
   ...: X=X_tr,
   ...: y=y_tr,
   ...: estimator=est,
   ...: param_grid=param_grid,
   ...: n_comp=50
   ...: )
   In [4]: p = training.init_pipeline()
   In [5]: p
   Out[5]: 
   Pipeline(steps=[
       (
           'enc',
            OneHotEncoder(
                drop='first', 
                handle_unknown='ignore',
                sparse=False
            )
        ),
        (
            'pca', 
            PCA(n_components=50)
        ), 
        (
            'model',
            LinearRegression()
        )
    ])
    In [6]: training.cross_val_fit(p=p, cv=5)
    [Parallel(n_jobs=7)]: Using backend LokyBackend with 7 concurrent workers...
    Out[16]: 
{'fit_time': array([0.23427629, 0.21670818, 0.21571112, 0.21584296, 0.24417901]),
 'score_time': array([0.00982404, 0.01001382, 0.01001382, 0.00982404, 0.01001287]),
 'estimator': [Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('pca', PCA(n_components=50)), ('model', LinearRegression())]),
  Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('pca', PCA(n_components=50)), ('model', LinearRegression())]),
  Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('pca', PCA(n_components=50)), ('model', LinearRegression())]),
  Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('pca', PCA(n_components=50)), ('model', LinearRegression())]),
  Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('pca', PCA(n_components=50)), ('model', LinearRegression())])],
 'test_score': array([0.11987983, 0.17250853, 0.05587166, 0.1421948 , 0.08326009]),
 'train_score': array([0.10265922, 0.09539385, 0.12307284, 0.10482568, 0.11259356])}
    """

    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        estimator, 
        param_grid: Dict,
        n_comp: Optional[int] = None 
    ):
        self.X, self.y = X, y 
        self.estimator = estimator
        self.param_grid = param_grid 
        self.n_comp = n_comp
        self._backup_dir = BACKUP + "models/" + str(self.estimator).split("(")[0] + "/"
        self._key = hash(str(self)) 

    def __repr__(self) -> str:
        return f"Training(X={self.X}, y={self.y}, estimator={repr(self.estimator)}, param_grid={self.param_grid}, n_comp={self.n_comp})"

    def _init_model(self): 
        """Initialize model from estimator and params grid."""
        return self.estimator.set_params(**self.param_grid)

    def init_pipeline(self): 
        """Return a pipeline with PCA as first step."""
        if self.n_comp is not None:
            steps = [
                (
                    "enc", 
                    OneHotEncoder(
                        drop="first",
                        handle_unknown="ignore", 
                        sparse=False
                    )
                ), 
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
                    "enc", 
                    OneHotEncoder(
                        drop="first", 
                        handle_unknown="ignore", 
                        sparse=False
                    )
                ), 
                (
                    "model", 
                    self._init_model()
                ),
            ]
        return Pipeline(steps=steps)

    def cross_val_fit(self,
        p: Pipeline, 
        cv: int
    ): 
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

    def check_pipe_backup(self, p: Pipeline) -> bool:
        """Check whether parameters grid has already been tested and backuped."""
        if os.path.isfile(self._backup_dir + str(self._key) + ".pkl"): 
            return True 
        return False  

    def save_pipe_results(
        self, 
        p: Pipeline,
        cv_results: Dict, 
    ):
        with open(self._backup_dir + str(self._key) + ".pkl", "wb") as file:
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
        "train_score": [], 
        "test_score": []
    }

    def fill_results_dict(
        p: Pipeline, 
        cv_results: Dict
    ): 
        results["pipeline"].append(p)
        results["train_score"].append( np.mean(cv_results["train_score"]) )
        results["test_score"].append( np.mean(cv_results["test_score"]) )

    estimators, params = config.init_models()
    for est, p in zip(estimators, params):
        grids = list(ParameterGrid(p))
        for param_grid in grids: 
            if comp_grid is not None: 
                for n_comp in comp_grid:
                    training = ModelTraining(
                        X=X_tr, 
                        y=y_tr, 
                        estimator=est,
                        param_grid=param_grid, 
                        n_comp=n_comp
                    )
                    p = training.init_pipeline()
                    if not training.check_pipe_backup(p):
                        cv_results = training.cross_val_fit(p=p, cv=cv)
                        training.save_pipe_results(p=p, cv_results=cv_results)
                        fill_results_dict(p, cv_results)
            else:
                training = ModelTraining(
                        X=X_tr, 
                        y=y_tr, 
                        estimator=est,
                        param_grid=param_grid
                    )
                p = training.init_pipeline()
                if not training.check_pipe_backup(p):
                    cv_results = training.cross_val_fit(p=p, cv=cv)
                    training.save_pipe_results(p=p, cv_results=cv_results)
                    fill_results_dict(p, cv_results)
                        
    return results 


        
