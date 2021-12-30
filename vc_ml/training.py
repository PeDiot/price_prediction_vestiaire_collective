"""Description.

Automate models training process using pipelines and grid search.
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
)

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from .data import (
    Target, 
    BACKUP
)

from .config import Config

from .estimators import EstimatorSwitcher

CPU_COUNT = os.cpu_count()
         
class Training: 
    """Fit a specific model to the train data for a given feature in pipeline."""

    def __init__(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        config: Config, 
        cv: int, 
        n_pcs: List[int] = [40]
        
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
            ("pca", PCA()),
            ("model", EstimatorSwitcher()),
        ])

    def build_grid_search(
        self, 
        p: Pipeline, 
    ): 
        """Build the grid search CV for parameters tuning."""
        param_grid = [
            mod(n_pcs=self._n_pcs)
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

def save_grid_search(
    g_fitted:GridSearchCV, 
    file_name: str
): 
    """Save fitted grid search."""
    path = BACKUP + "models/" + file_name
    with open(path , "wb") as file:
        dump(obj=g_fitted, file=file)  

def display_results(g:GridSearchCV) -> Table: 
    """Display best model's results."""
    tab = Table()
    best_params = g.best_params_
    best_score = g.best_score_
    for key in best_params.keys(): 
        tab.add_column(key)
    tab.add_column("Score")
    tab.add_row( *[str(val) for val in best_params.values()], str(best_score) )
    print(tab)
