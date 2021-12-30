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
)

from .data import BACKUP 

from .estimators import ( 
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

def create_config(file_name: str = "config.yaml"): 
    config = Config(
        lr=LREstimator(), 
        ridge=RidgeEstimator(), 
        tree=TreeEstimator(), 
        rf=RFEstimator(),
        gb=GBEstimator(), 
        mlp=MLPEstimator() 
    )
    path = BACKUP + "models/" + file_name
    with open(path, "w") as file:
        file.write(to_yaml(config))

def load_config(file_name: str= "config.yaml"): 
    """Load models configuration file."""
    path = BACKUP + "models/" + file_name
    with open(path, "r") as file:
        config = file.read()
    return from_yaml(c=Config, s=config)
