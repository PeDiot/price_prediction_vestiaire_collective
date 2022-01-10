"""Description.

Usage example of the vc_ml library.
"""

from .data import (
    BACKUP, 
    Target, 
    read_data, 
    to_dummies, 
    SplitData, 
    load_feature_vector, 
    load_target,
) 

from .estimators import ( 
    DummyEstimator, 
    LREstimator, 
    RidgeEstimator, 
    TreeEstimator, 
    RFEstimator, 
    GBEstimator, 
    MLPEstimator,
)

from .config import (
    Config, 
    create_config, 
    load_config, 
)

from .training import(
    CPU_COUNT,
    ModelTraining, 
    train_models, 
)

from .selection import (
    ModelDir, 
    get_files_path, 
    get_cv_results,
    get_best_estimator, 
)

from rich import print

X_tr, y_tr = load_feature_vector(file_name="train.pkl"), load_target(file_name="train.pkl")
print(f"Feature vector: {X_tr}")
print(f"Target: {y_tr}")

config = load_config(file_name="config_init.yaml")
print(config)

train_models(
    X_tr=X_tr,
    y_tr=y_tr,
    config=config,
    cv=5,
    comp_grid=[60]
)