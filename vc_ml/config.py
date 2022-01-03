"""Description.

Classes and functions for models configuration.
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
        return f"Config(lr={self.lr}, ridge={self.ridge}, tree={self.tree}, rf={self.rf}, gb={self.gb}, mlp={self.mlp})"
    
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
    config = Config(
        dum=DummyEstimator(), 
        lr=LREstimator(), 
        ridge=RidgeEstimator(), 
        tree=TreeEstimator(), 
        rf=RFEstimator(),
        gb=GBEstimator(), 
        mlp=MLPEstimator(hidden_layer_sizes=[[100, 100,], [200,], [50, 100, 50,]]) 
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
