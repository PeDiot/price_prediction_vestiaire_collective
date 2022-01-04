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

    In [4]: y_tr = load_target(
    ...: file_name="train.pkl",
    ...: target=Target.NUM_LIKES
    ...: )

    In [5]: est = Ridge()

    In [6]: params = {"alpha": .01, "fit_intercept": True}

    In [7]: training = ModelTraining(
    ...: X=X_tr,
    ...: y=y_tr,
    ...: estimator=est,
    ...: params=params,
    ...: n_comp=50
    ...: )

    In [8]: pipe = training.init_pipeline()

    In [9]: pipe
    Out[9]: Pipeline(steps=[('pca', PCA(n_components=50)), ('model', Ridge(alpha=0.01))])

    In [10]: cv_results = training.cross_val_fit(p=pipe, cv=5)
    [Parallel(n_jobs=7)]: Using backend LokyBackend with 7 concurrent workers.
    [CV] END ..................., score=(train=0.106, test=0.091) total time=   0.1s
    [CV] END ..................., score=(train=0.098, test=0.078) total time=   0.1s
    [Parallel(n_jobs=7)]: Done   2 out of   5 | elapsed:    2.4s remaining:    3.6s
    [CV] END ..................., score=(train=0.100, test=0.069) total time=   0.1s
    [CV] END ..................., score=(train=0.090, test=0.097) total time=   0.1s
    [CV] END ..................., score=(train=0.103, test=0.090) total time=   0.1s
    [Parallel(n_jobs=7)]: Done   5 out of   5 | elapsed:    2.4s finished

    In [11]: cv_results
    Out[11]:
    {'fit_time': array([0.19979906, 0.18805981, 0.17964172, 0.16840768, 0.20743465]),
    'score_time': array([0.01279092, 0.01179409, 0.0090611 , 0.01211762, 0.00498939]),
    'estimator': [Pipeline(steps=[('pca', PCA(n_components=50)), ('model', Ridge(alpha=0.01))]),
    Pipeline(steps=[('pca', PCA(n_components=50)), ('model', Ridge(alpha=0.01))]),
    Pipeline(steps=[('pca', PCA(n_components=50)), ('model', Ridge(alpha=0.01))]),
    Pipeline(steps=[('pca', PCA(n_components=50)), ('model', Ridge(alpha=0.01))]),
    Pipeline(steps=[('pca', PCA(n_components=50)), ('model', Ridge(alpha=0.01))])],
    'test_score': array([0.06944281, 0.09702687, 0.09063966, 0.07798346, 0.09012322]),
    'train_score': array([0.09994824, 0.09032111, 0.10575738, 0.09777393, 0.1027564 ])}

    In [12]: training.check_pipe_backup(p=pipe)
    Out[12]: False
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
        self._key = hash(str(self)) 

    def __repr__(self) -> str:
        return f"Training(X={self.X}, y={self.y}, estimator={repr(self.estimator)}, params={self.params}, n_comp={self.n_comp})"

    def _init_model(self): 
        """Initialize model from estimator and params grid."""
        return self.estimator.set_params(**self.params)

    def init_pipeline(self): 
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

    estimators, grids = config.init_models()
    for est, g in zip(estimators, grids):
        g = list(ParameterGrid(g))
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
                    print(f"estimator: {training.estimator} - params: {training.params} - n_comp: {training.n_comp}")
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
                        param_grid=params
                    )
                print(f"Estimator: {training.estimator} - params: {training.params}")
                p = training.init_pipeline()
                if not training.check_pipe_backup(p):
                    cv_results = training.cross_val_fit(p=p, cv=cv)
                    training.save_pipe_results(p=p, cv_results=cv_results)
                    fill_results_dict(p, cv_results)
                        
    return results 


        
