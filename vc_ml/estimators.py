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

    def __post_init__(self):
        """Check whether alpha is a list of positive floats."""
        for alpha in self.alpha:
            if alpha < 0 or type(alpha) != float:
                raise ValueError("alpha must be a list of positive floats.")

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

    def __post_init__(self):
        """Check parameters."""
        for max_depth in self.max_depth:
            if type(max_depth) != type(None):
                if type(max_depth) != int or max_depth <= 0:
                    raise ValueError("'max_depth' must be a list of strictly positive intergers.")
        for min_samples_split, min_samples_leaf in zip(self.min_samples_split, self.min_samples_leaf):
            if type(min_samples_split) != int or min_samples_split <= 0:
                raise ValueError("min_samples_split' must be a list of strictly positive intergers.")
            if type(min_samples_leaf) != int or min_samples_leaf <= 0:
                raise ValueError("min_samples_leaf' must be a list of strictly positive intergers.")
        for ccp in self.ccp_alpha: 
            if type(ccp) != float or ccp < 0.0:
                raise ValueError("'ccp_alpha' must be a list of positive floats.")
        for max_features in self.max_features:
            if max_features not in ("auto", "sqrt", "log2", None):
                raise ValueError("'max_features' must take the following values: 'auto', 'sqrt', 'log2' or None.")

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

    def __post_init__(self):
        """Check parameters."""
        for n_estimators in self.n_estimators:
            if type(n_estimators) != int or n_estimators <= 0:
                raise ValueError("'n_estimators' must be a list of strictly positive intergers.")
        for criterion in self.criterion:
            if criterion not in ("squared_error", "absolute_error", "poisson"):
                raise ValueError("'criterion' must take the following values: 'squared_error', 'absolute_error' or 'poisson'.")
        for max_depth in self.max_depth:
            if type(max_depth) != type(None):
                if type(max_depth) != int or max_depth <= 0:
                    raise ValueError("'max_depth' must be a list of strictly positive intergers.")
        for min_samples_split, min_samples_leaf in zip(self.min_samples_split, self.min_samples_leaf):
            if type(min_samples_split) != int or min_samples_split <= 0:
                raise ValueError("min_samples_split' must be a list of strictly positive intergers.")
            if type(min_samples_leaf) != int or min_samples_leaf <= 0:
                raise ValueError("min_samples_leaf' must be a list of strictly positive intergers.")
        for max_features in self.max_features:
            if max_features not in ("auto", "sqrt", "log2", None):
                raise ValueError("'max_features' must take the following values: 'auto', 'sqrt', 'log2' or None.")
        for max_samples in self.max_samples:
            if type(max_samples) != float or max_samples < 0.0 or max_samples > 1.:
                raise ValueError("'max_samples' must be a list of floats between 0 and 1.")
        for oob_score in self.oob_score:
            if type(oob_score) != bool:
                raise ValueError("'oob_score' must be a list of boolean.")

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

    def __post_init__(self):
        """Check parameters."""
        for n_estimators in self.n_estimators:
            if type(n_estimators) != int or n_estimators <= 0:
                raise ValueError("'n_estimators' must be a list of strictly positive intergers.")
        for criterion in self.criterion:
            if criterion not in ("squared_error", "friedman_mse", "mse", "mae"):
                raise ValueError("'criterion' must take the following values: 'squared_error', 'friedman_mse', 'mse or 'mae'.")
        for max_depth in self.max_depth:
            if type(max_depth) != type(None):
                if type(max_depth) != int or max_depth <= 0:
                    raise ValueError("'max_depth' must be a list of strictly positive intergers.")
        for min_samples_split, min_samples_leaf in zip(self.min_samples_split, self.min_samples_leaf):
            if type(min_samples_split) != int or min_samples_split <= 0:
                raise ValueError("min_samples_split' must be a list of strictly positive intergers.")
            if type(min_samples_leaf) != int or min_samples_leaf <= 0:
                raise ValueError("min_samples_leaf' must be a list of strictly positive intergers.")
        for loss in self.loss: 
            if loss not in ("squared_error", "absolute_error", "huber", "quantile"):
                raise ValueError("'loss' must take the following values: 'squared_error', 'absolute_error', 'huber' or 'quantile'.")
        for learning_rate in self.learning_rate:
            if learning_rate <= 0 or type(learning_rate) != float: 
                raise ValueError("'learning_rate' must be a list of strictly positive floats.")
        for tol in self.tol:
            if tol <= 0 or type(tol) != float:
                raise ValueError("'tol' must be a list of strictly positive float.")        

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

    def __post_init__(self): 
        "Check parameters."
        for layers in self.hidden_layer_sizes:
            if type(layers) != list:
                raise ValueError("'The neural network architecture must be a list.")
            for size in layers:
                if type(size) != int or size <= 0:
                    raise ValueError("The number of neurons in each layer must be a strictly positive integer.")
        for max_iter in self.max_iter:
            if type(max_iter) != int or max_iter <= 0:
                raise ValueError("'max_iter' must be a list of strictly positive intergers.")
        for activation in self.activation: 
            if activation not in ("identity", "logistic", "tanh", "relu"):
                raise ValueError("'activation' must take the following values: 'identity', 'logistic', 'tanh' or 'relu'.")
        for solver in self.solver: 
            if solver not in ("lbfgs", "sgd", "adam"):
                raise ValueError("'solver' must take the following values: 'lbfgs', 'sgd' or'adam'.")
        for learning_rate_init in self.learning_rate_init:
            if learning_rate_init <= 0 or type(learning_rate_init) != float:
                raise ValueError("'learning_rate_init' must be a strictly positive floats.")

    def __call__(self) -> Dict:
        return {
            "estimator": MLPRegressor(), 
            "hidden_layer_sizes": self.hidden_layer_sizes, 
            "max_iter": self.max_iter, 
            "activation": self.activation, 
            "solver": self.solver, 
            "learning_rate_init": self.learning_rate_init, 
        }

