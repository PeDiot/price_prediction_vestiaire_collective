"""Description.

Automate the training step to obtain the most performant model.

Example: 
>>> from vc_ml import (
... load_data, 
... load_config, 
... Training, 
... Target, 
)   
>>> config = load_config()
>>> X_tr, y_tr = load_data(file_name="train.pkl", target=Target.PRICE)
>>> training = Training(
... X=X_tr, 
... y=y_tr, 
... config=config,
... cv=3 
... )
>>> training
Training(X=[[0 1 0 ... 0 1 0]
 [0 1 0 ... 0 0 0]
 [0 1 1 ... 0 1 0]
 ...
 [0 1 0 ... 0 1 0]
 [0 1 0 ... 0 1 0]
 [1 0 0 ... 0 1 0]], y=[300.    50.35 125.   ...  85.   325.   249.  ], config=Config(lr=LREstimator(fit_intercept=[True, False]), ridge=RidgeEstimator(alpha=[1.0, 0.1, 0.01], fit_intercept=[True, False]), rf=RFEstimator(n_estimators=[100, 250, 500], max_depth=[10, 50, 100, None], oob_score=[True])), cv=3, n_pcs=40)
>>> p = training.make_pipeline()
>>> p 
Pipeline(steps=[('pca', PCA(n_components=40)), ('model', EstimatorSwitcher())])
>>> g = training.build_grid_search(p) 
>>> g
GridSearchCV(cv=3,
             estimator=Pipeline(steps=[('pca', PCA(n_components=40)),
                                       ('model', EstimatorSwitcher())]),
             n_jobs=7,
             param_grid=[{'model__estimator': [LinearRegression()],
                          'model__estimator__fit_intercept': [True, False]},
                         {'model__estimator': [Ridge()],
                          'model__estimator__alpha': [1.0, 0.1, 0.01],
                          'model__estimator__fit_intercept': [True, False]},
                         {'model__estimator': [RandomForestRegressor()],
                          'model__estimator__max_depth': [10, 50, 100, None],
                          'model__estimator__n_estimators': [100, 250, 500],
                          'model__estimator__oob_score': [True]}],
             return_train_score=True, verbose=3)
>>> training.fit(g)    
Fitting 3 folds for each of 20 candidates, totalling 60 fits
[CV 1/3] END model__estimator=LinearRegression(), model__estimator__fit_intercept=True;, score=(train=0.241, test=0.148) total time=   0.0s
...
         
"""

import numpy as np 
import pandas as pd
from pickle import load, dump 
import os
from rich import print

from serde import serialize, deserialize
from serde.yaml import to_yaml, from_yaml

from typing import (
    List, 
    Dict, 
    Optional, 
)
from dataclasses import dataclass

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LinearRegression, 
    Ridge, 
)
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
)

from .data import (
    Target, 
    BACKUP
)

CPU_COUNT = os.cpu_count()

@serialize
@deserialize
@dataclass
class LREstimator: 
    fit_intercept: Optional[list[bool]] = None  

    def __call__(self) -> Dict:
        return {
            "model__estimator": [LinearRegression()], 
            "model__estimator__fit_intercept": self.fit_intercept
        }

@serialize
@deserialize
@dataclass
class RidgeEstimator: 
    alpha: Optional[list[float]] = None
    fit_intercept: Optional[list[bool]] = None 

    def __call__(self) -> Dict:
        return {
            "model__estimator": [Ridge()], 
            "model__estimator__alpha": self.alpha, 
            "model__estimator__fit_intercept": self.fit_intercept
        }

@serialize
@deserialize
@dataclass
class RFEstimator: 
    n_estimators: Optional[list[int]] = None
    max_depth: Optional[list[int]] = None
    min_samples_split: Optional[list[int]] = None 
    min_samples_leaf: Optional[list[int]] = None 
    max_features: Optional[list[str]] = None 
    max_samples: Optional[list[float]] = None 
    oob_score: Optional[list[bool]] = None 

    def __call__(self) -> Dict:
        return {
            "model__estimator": [RandomForestRegressor()], 
            "model__estimator__n_estimators": self.n_estimators, 
            "model__estimator__max_depth": self.max_depth, 
            "model__estimator__min_samples_split": self.min_samples_split, 
            "model__estimator__min_samples_leaf": self.min_samples_leaf,  
            "model__estimator__max_features": self.max_features, 
            "model__estimator__max_samples": self.max_samples, 
            "model__estimator__oob_score": self.oob_score
        }
        
@serialize
@deserialize
@dataclass
class Config: 
    lr: Optional[LREstimator] = None 
    ridge: Optional[RidgeEstimator] = None 
    rf: Optional[RFEstimator] = None 

    def __iter__(self):
        """Iterate across models."""
        return iter(self.__dict__.values())

    def __repr__(self) -> str:
        return f"Config(lr={self.lr}, ridge={self.ridge}, rf={self.rf})"

def create_config(file_name: str = "config.yaml"): 
    config = Config(
        lr=LREstimator(fit_intercept=[True, False]), 
        ridge=RidgeEstimator(
            fit_intercept=[True, False], 
            alpha=[1., .1, .01]
        ), 
        rf=RFEstimator(
            n_estimators=[100, 250, 500], 
            max_depth=[10, 50, 100, None], 
            oob_score=[True]
        )
    )
    path = BACKUP + "models/" + file_name
    with open(path, "w") as file:
        file.write(to_yaml(config))

def load_config(file_name: str= "config.yaml"): 
    """Load models configuration file."""
    path = BACKUP + "models/" + file_name
    with open(path, "r") as file:
        config = file.read()
    return from_yaml(Config, config)

class EstimatorSwitcher(BaseEstimator):
    """A custom BaseEstimator that can switch between estimators."""
    def __init__(
        self, 
        estimator=LinearRegression(),
    ):
        self.estimator = estimator

    def __repr__(self):
        return "EstimatorSwitcher()"

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

class Training: 
    """Fit a specific model to the train data for a given feature in pipeline."""

    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        config: Config, 
        cv: int, 
        n_pcs: int = 40
        
    ):
        self.X, self.y = X, y 
        self._config = config
        self._cv = cv 
        self._n_pcs = n_pcs 

    def __repr__(self) -> str:
        return f"Training(X={self.X}, y={self.y}, config={repr(self._config)}, cv={self._cv}, n_pcs={self._n_pcs})"

    def make_pipeline(self): 
        """Return a pipeline with PCA as first step."""
        return Pipeline([
            ("pca", PCA(n_components=self._n_pcs)),
            ("model", EstimatorSwitcher()),
        ])

    def build_grid_search(
        self, 
        p: Pipeline, 
    ): 
        """Build the grid search CV for parameters tuning."""
        param_grid = [
            mod()
            for mod in self._config  
            if mod is not None 
        ]
        return GridSearchCV(
            estimator=p, 
            param_grid=param_grid, 
            cv=self._cv, 
            n_jobs=CPU_COUNT-1, 
            return_train_score=True, 
            verbose=3
        )
    
    def fit(self, g:GridSearchCV) -> GridSearchCV: 
        """Fit grid search to the data."""
        return g.fit(self.X, self.y)
    
    def save(
        self, 
        g_fitted:GridSearchCV, 
        file_name: str
    ): 
        """Save fitted grid search."""
        path = BACKUP + "models/" + file_name
        with open(path , "wb") as file:
            dump(obj=g_fitted, file=file)  