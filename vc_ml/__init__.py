"""Description. 

Machine Learning library to fit models on Vestiaire Collective data."""

from .data import (
    BACKUP, 
    Target, 
    SplitData, 
    load_data,
) 

from .training import(
    CPU_COUNT,
    LREstimator, 
    RidgeEstimator, 
    RFEstimator,  
    Config, 
    create_config, 
    load_config, 
    EstimatorSwitcher, 
    Training, 
)

