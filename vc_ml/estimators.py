"""Description.

Define estimators to train as dataclasses and create class EstimatorSwitcher to train multiple estimators in GridSearchCV. 
"""

from serde import serialize, deserialize

from dataclasses import dataclass, field

from typing import (
    List, 
    Dict, 
    Optional, 
    Tuple, 
)

from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LinearRegression, 
    Ridge, 
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
)
from sklearn.neural_network import MLPRegressor

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


@serialize
@deserialize
@dataclass
class LREstimator: 
    fit_intercept: Optional[list[bool]] = field(default_factory=lambda: [True])  

    def __call__(
        self, 
        n_pcs: List[int] 
    ) -> Dict:
        return {
            "pca__n_components": n_pcs, 
            "model__estimator": [LinearRegression()], 
            "model__estimator__fit_intercept": self.fit_intercept
        }

@serialize
@deserialize
@dataclass
class RidgeEstimator: 
    alpha: Optional[list[float]] = field(default_factory=lambda: [1.0])  
    fit_intercept: Optional[list[bool]] = field(default_factory=lambda: [True])  

    def __call__(
        self,
        n_pcs: List[int] 
    ) -> Dict:
        return {
            "pca__n_components": n_pcs, 
            "model__estimator": [Ridge()], 
            "model__estimator__alpha": self.alpha, 
            "model__estimator__fit_intercept": self.fit_intercept
        }

@serialize
@deserialize
@dataclass
class TreeEstimator: 
    max_depth: Optional[list[str]] = field(default_factory=lambda: [None])  
    min_samples_split: Optional[list[int]] = field(default_factory=lambda: [2])  
    min_samples_leaf: Optional[list[int]] = field(default_factory=lambda: [1])  
    max_features: Optional[list[str]] = field(default_factory=lambda: ["auto"])  
    ccp_alpha: Optional[list[float]] = field(default_factory=lambda: [0.0])  

    def __call__(
        self,
        n_pcs: List[int] 
    ) -> Dict:
        return {
            "pca__n_components": n_pcs, 
            "model__estimator": [DecisionTreeRegressor()], 
            "model__estimator__max_depth": self.max_depth, 
            "model__estimator__min_samples_split": self.min_samples_split, 
            "model__estimator__min_samples_leaf": self.min_samples_leaf,  
            "model__estimator__max_features": self.max_features, 
            "model__estimator__ccp_alpha": self.ccp_alpha
        }


@serialize
@deserialize
@dataclass
class RFEstimator: 
    n_estimators: Optional[list[int]] = field(default_factory=lambda: [100])  
    max_depth: Optional[list[int]] = field(default_factory=lambda: [None])  
    min_samples_split: Optional[list[int]] = field(default_factory=lambda: [2])  
    min_samples_leaf: Optional[list[int]] = field(default_factory=lambda: [1])  
    max_features: Optional[list[str]] = field(default_factory=lambda: ["auto"])  
    max_samples: Optional[list[float]] = field(default_factory=lambda: [1.0])  
    oob_score: Optional[list[bool]] = field(default_factory=lambda: [True])   

    def __call__(
        self, 
        n_pcs: List[int]
    ) -> Dict:
        return {
            "pca__n_components": n_pcs, 
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
class GBEstimator: 
    n_estimators: Optional[list[int]] = field(default_factory=lambda: [100])   
    min_samples_split: Optional[list[int]] = field(default_factory=lambda: [2])  
    min_samples_leaf: Optional[list[int]] = field(default_factory=lambda: [1])  
    max_depth: Optional[list[int]] = field(default_factory=lambda: [3])
    loss: Optional[list[str]] = field(default_factory=lambda: ["squared_error"])  
    learning_rate: Optional[list[float]] = field(default_factory=lambda: [.1])  
    criterion: Optional[list[str]] = field(default_factory=lambda: ["friedman_mse"])  
    tol: Optional[list[float]] = field(default_factory=lambda: [.001])  

    def __call__(
        self, 
        n_pcs: List[int]
    ) -> Dict:
        return {
            "pca__n_components": n_pcs, 
            "model__estimator": [GradientBoostingRegressor()], 
            "model__estimator__n_estimators": self.n_estimators, 
            "model__estimator__max_depth": self.max_depth, 
            "model__estimator__min_samples_split": self.min_samples_split, 
            "model__estimator__min_samples_leaf": self.min_samples_leaf,  
            "model__estimator__loss": self.loss, 
            "model__estimator__learning_rate": self.learning_rate, 
            "model__estimator__criterion": self.criterion, 
            "model__estimator__tol": self.tol 
        }

@serialize
@deserialize
@dataclass
class MLPEstimator:
    hidden_layer_sizes: Optional[list[list[int]]] = field(default_factory=lambda: [[100,]])  
    max_iter: Optional[list[float]] = field(default_factory=lambda: [200])  
    activation: Optional[list[str]] = field(default_factory=lambda: ["relu"])  
    solver: Optional[list[str]] = field(default_factory=lambda: ["adam"])  
    learning_rate_init: Optional[list[float]] = field(default_factory=lambda: [.001])  

    def __call__(
        self, 
        n_pcs: List[int]
    ) -> Dict:
        return {
            "pca__n_components": n_pcs, 
            "model__estimator": [MLPRegressor()], 
            "model__estimator__hidden_layer_sizes": self.hidden_layer_sizes, 
            "model__estimator__max_iter": self.max_iter, 
            "model__estimator__activation": self.activation, 
            "model__estimator__solver": self.solver, 
            "model__estimator__learning_rate_init": self.learning_rate_init, 
        }

