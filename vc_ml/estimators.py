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

from sklearn.dummy import DummyRegressor

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

@serialize
@deserialize
@dataclass
class DummyEstimator:
    strategy: Optional[list[str]] = field(default_factory=lambda: ["mean"]) 

    def __call__(self) -> Dict:
        return {
            "estimator": DummyRegressor(), 
            "strategy": self.strategy
        }

@serialize
@deserialize
@dataclass
class LREstimator: 
    fit_intercept: Optional[list[bool]] = field(default_factory=lambda: [True])  

    def __call__(self) -> Dict:
        return {
            "estimator": LinearRegression(), 
            "fit_intercept": self.fit_intercept
        }

@serialize
@deserialize
@dataclass
class RidgeEstimator: 
    alpha: Optional[list[float]] = field(default_factory=lambda: [1.0])  
    fit_intercept: Optional[list[bool]] = field(default_factory=lambda: [True])  

    def __call__(self) -> Dict:
        return {
            "estimator": Ridge(), 
            "alpha": self.alpha, 
            "fit_intercept": self.fit_intercept
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

    def __call__(self) -> Dict:
        return  {
            "estimator": DecisionTreeRegressor(), 
            "max_depth": self.max_depth, 
            "min_samples_split": self.min_samples_split, 
            "min_samples_leaf": self.min_samples_leaf,  
            "max_features": self.max_features, 
            "ccp_alpha": self.ccp_alpha
        }
        
@serialize
@deserialize
@dataclass
class RFEstimator: 
    n_estimators: Optional[list[int]] = field(default_factory=lambda: [100])  
    criterion: Optional[list[str]] = field(default_factory=lambda: ["squared_error"])  
    max_depth: Optional[list[int]] = field(default_factory=lambda: [None])  
    min_samples_split: Optional[list[int]] = field(default_factory=lambda: [2])  
    min_samples_leaf: Optional[list[int]] = field(default_factory=lambda: [1])  
    max_features: Optional[list[str]] = field(default_factory=lambda: ["auto"])  
    max_samples: Optional[list[float]] = field(default_factory=lambda: [1.0])  
    oob_score: Optional[list[bool]] = field(default_factory=lambda: [True])   

    def __call__(self) -> Dict:
        return {
            "estimator": RandomForestRegressor(), 
            "n_estimators": self.n_estimators, 
            "max_depth": self.max_depth, 
            "min_samples_split": self.min_samples_split, 
            "min_samples_leaf": self.min_samples_leaf,  
            "max_features": self.max_features, 
            "max_samples": self.max_samples, 
            "oob_score": self.oob_score
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

    def __call__(self) -> Dict:
        return {
            "estimator": GradientBoostingRegressor(), 
            "n_estimators": self.n_estimators, 
            "max_depth": self.max_depth, 
            "min_samples_split": self.min_samples_split, 
            "min_samples_leaf": self.min_samples_leaf,  
            "loss": self.loss, 
            "learning_rate": self.learning_rate, 
            "criterion": self.criterion, 
            "tol": self.tol 
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

    def __call__(self) -> Dict:
        return {
            "estimator": MLPRegressor(), 
            "hidden_layer_sizes": self.hidden_layer_sizes, 
            "max_iter": self.max_iter, 
            "activation": self.activation, 
            "solver": self.solver, 
            "learning_rate_init": self.learning_rate_init, 
        }

