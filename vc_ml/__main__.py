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

print(" "*100)
print("Usage example of the 'vc_ml' library which aims to automate model training.")
print(" "*100)
print("-"*100)
print(" "*100)
print("Data used in models")
print(" "*100)

X_tr, y_tr = load_feature_vector(file_name="train.pkl"), load_target(file_name="train.pkl")
print(f"Feature vector: {X_tr}")
print(f"Target: {y_tr}")
print(" "*100)
print("-"*100)

print(" "*100)
print("Model configuration")
print(" "*100)

config = load_config(file_name="config_init.yaml")
print(config)
print(" "*100)
print("-"*100)

print(" "*100)
print("Model training")
print(" "*100)

train_models(
    X_tr=X_tr,
    y_tr=y_tr,
    config=config,
    cv=5,
    comp_grid=[60]
)
print(" "*100)
print("-"*100)

print(" "*100)
print("Model selection")

cv_results = get_cv_results(files_path=get_files_path())
print(f"""Cross-validation results\n {cv_results}""")
print(f"Best estimator: {get_best_estimator(cv_results)}")