"""Description.

Classes and functions for models configuration.

Example: 

In [1]: from vc_ml import (create_config, load_config)

In [2]: create_config(file_name="config_init.yaml")

In [3]: config = load_config(file_name="config_init.yaml")

In [4]: config
Out[4]: Config(lr=LREstimator(fit_intercept=[True]), ridge=RidgeEstimator(alpha=[1.0], fit_intercept=[True]), tree=TreeEstimator(max_depth=[None], min_samples_split=[2], min_samples_leaf=[1], max_features=['auto'], ccp_alpha=[0.0]), rf=RFEstimator(n_estimators=[100], max_depth=[None], min_samples_split=[2], min_samples_leaf=[1], max_features=['auto'], max_samples=[1.0], oob_score=[True]), gb=GBEstimator(n_estimators=[100], min_samples_split=[2], min_samples_leaf=[1], max_depth=[3], loss=['squared_error'], learning_rate=[0.1], criterion=['friedman_mse'], tol=[0.001]), mlp=MLPEstimator(hidden_layer_sizes=[[100]], max_iter=[200], activation=['relu'], solver=['adam'], learning_rate_init=[0.001]))

In [5]: config.init_models()
Out[5]: 
([DummyRegressor(),
  LinearRegression(),
  Ridge(),
  DecisionTreeRegressor(),
  RandomForestRegressor(),
  GradientBoostingRegressor(),
  MLPRegressor()],
 [{'strategy': ['mean']},
  {'fit_intercept': [True]},
  {'alpha': [1.0], 'fit_intercept': [True]},
  {'max_depth': [None],
   'min_samples_split': [2],
   'min_samples_leaf': [1],
   'max_features': ['auto'],
   'ccp_alpha': [0.0]},
  {'n_estimators': [100],
   'max_depth': [None],
   'min_samples_split': [2],
   'min_samples_leaf': [1],
   'max_features': ['auto'],
   'max_samples': [1.0],
   'oob_score': [True]},
  {'n_estimators': [100],
   'max_depth': [3],
   'min_samples_split': [2],
   'min_samples_leaf': [1],
   'loss': ['squared_error'],
   'learning_rate': [0.1],
   'criterion': ['friedman_mse'],
   'tol': [0.001]},
  {'hidden_layer_sizes': [[100]],
   'max_iter': [200],
   'activation': ['relu'],
   'solver': ['adam'],
   'learning_rate_init': [0.001]}])
"""

from dataclasses import dataclass

from serde import serialize, deserialize
from serde.yaml import to_yaml, from_yaml

from typing import (
    List, 
    Dict, 
    Optional, 
    Tuple
)

from .data import BACKUP 

from .estimators import ( 
    DummyEstimator, 
    LREstimator, 
    RidgeEstimator, 
    TreeEstimator, 
    RFEstimator, 
    GBEstimator,
    MLPEstimator,
)

@serialize
@deserialize
@dataclass
class Config: 
    dum: Optional[DummyEstimator] = None
    lr: Optional[LREstimator] = None 
    ridge: Optional[RidgeEstimator] = None 
    tree: Optional[TreeEstimator] = None 
    rf: Optional[RFEstimator] = None 
    gb: Optional[GBEstimator] = None
    mlp: Optional[MLPEstimator] = None  

    def __iter__(self):
        """Iterate across models."""
        return iter(self.__dict__.values())

    def __repr__(self) -> str:
        return f"Config(dum={self.dum}, lr={self.lr}, ridge={self.ridge}, tree={self.tree}, rf={self.rf}, gb={self.gb}, mlp={self.mlp})"
    
    def init_models(self) -> Tuple[List]:
        """Identify estimator and param grid."""
        estimators, params = list(), list() 
        for mod in self.__dict__.values():
            if mod is not None:
                model_dict = mod()
                estimators.append(model_dict["estimator"])
                params.append(
                    {
                        key: model_dict[key] 
                        for key in model_dict.keys()
                        if key != "estimator"
                    }
                )
        return estimators, params 

def create_config(file_name: str = "config_init.yaml"): 
    """Initialize a configuration file."""
    config = Config(
        dum=DummyEstimator(), 
        lr=LREstimator(), 
        ridge=RidgeEstimator(), 
        tree=TreeEstimator(), 
        rf=RFEstimator(),
        gb=GBEstimator(), 
        mlp=MLPEstimator() 
    )
    path = BACKUP + "config/" + file_name
    with open(path, "w") as file:
        file.write(to_yaml(config))

def load_config(file_name: str= "config.yaml"): 
    """Load models configuration file."""
    path = BACKUP + "config/" + file_name
    with open(path, "r") as file:
        config = file.read()
    return from_yaml(c=Config, s=config)
