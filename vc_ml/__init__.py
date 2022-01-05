"""Description. 

Machine Learning library to fit models on Vestiaire Collective data.

Example: 
In [1]: from vc_ml import (
   ...: read_data,
   ...: to_dummies,
   ...: SplitData,
   ...: Target
   ...: )
In [2]: data = read_data(file_name="vc_data_cleaned.pkl")
In [3]: new_data = to_dummies(data)
In [4]: s = SplitData(data=new_data)
In [5]: X = s.get_feature_vector()
In [6]: y = s.get_targets()
In [7]: X_train, X_test, y_train, y_test = s.split(X, y)
In [8]: s.save(
    ...: X=X_train,
    ...: y=y_train,
    ...: file_name="train.pkl"
    ...: )
In [9]: s.save(
    ...: X=X_test,
    ...: y=y_test,
    ...: file_name='test.pkl'
    ...: )

In [2]: from vc_ml import (
   ...: load_feature_vector,
   ...: load_target,
   ...: load_config,
   ...: train_models
   ...: )

In [3]: X_tr = load_feature_vector(file_name="train.pkl")

In [4]: y_tr = load_target(file_name="train.pkl")

In [5]: config = load_config()

In [6]: config
Out[6]: Config(lr=None, ridge=None, tree=None, rf=None, gb=GBEstimator(n_estimators=[250, 500, 750, 1000], min_samples_split=[2, 5, 10, 15, 20], min_samples_leaf=[1, 5, 10, 15, 20], max_depth=[3, 5, 10, 15, 20, 100], loss=['squared_error', 'absolute_error', 'huber'], learning_rate=[0.1, 0.01, 0.001], criterion=['friedman_mse', 'squared_error', 'mse', 'mae'], tol=[0.001]), mlp=None)

In [7]: train_models(
   ...: X_tr=X_tr, y_tr=y_tr,
   ...: config=config,
   ...: comp_grid=[40, 60, 80, None]
   ...: )
21600 parameter combinations to test for GradientBoostingRegressor().
estimator: GradientBoostingRegressor() - params: {'criterion': 'friedman_mse', 'learning_rate': 0.1, 'loss': 'squared_error', 'max_depth': 3,
'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 250, 'tol': 0.001} - n_comp: 40
[Parallel(n_jobs=7)]: Using backend LokyBackend with 7 concurrent workers...
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

