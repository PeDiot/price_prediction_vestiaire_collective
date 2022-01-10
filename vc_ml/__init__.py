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

set PYTHONHASHSEED=0

In [1]: from vc_ml import (
   ...: load_feature_vector,
   ...: load_target,
   ...: load_config,
   ...: train_models,
   ...: )

In [2]: X_tr, y_tr = load_feature_vector(file_name="train.pkl"), load_target(file_name="train.pkl")

In [4]: config = load_config(file_name="config_gb.yaml")

In [5]: train_models(
   ...: X_tr=X_tr,
   ...: y_tr=y_tr,
   ...: config=config,
   ...: cv=5,
   ...: comp_grid=[40, 60, 80, None]
   ...: )
Training(estimator=GradientBoostingRegressor(), params={'criterion': 'squared_error', 'learning_rate': 0.01, 'loss': 'huber', 'max_depth': 15,        
'min_samples_leaf': 5, 'min_samples_split': 10, 'n_estimators': 250, 'tol': 0.001}, n_comp=40)
Training(estimator=GradientBoostingRegressor(), params={'criterion': 'friedman_mse', 'learning_rate': 0.001, 'loss': 'absolute_error', 'max_depth':   
20, 'min_samples_leaf': 10, 'min_samples_split': 5, 'n_estimators': 1000, 'tol': 0.001}, n_comp=40)
[Parallel(n_jobs=5)]: Using backend ThreadingBackend with 5 concurrent workers.
[Parallel(n_jobs=5)]: Using backend ThreadingBackend with 5 concurrent workers...

In [1]: from vc_ml import (
   ...: get_files_path,
   ...: get_cv_results,
   ...: get_best_estimator
   ...: )

In [2]: paths = get_files_path()

In [3]: cv_results = get_cv_results(files_path=paths)

In [4]: get_best_estimator(cv_results=cv_results)
Out[4]: 
{'best_estimator': Pipeline(steps=[('model',
                  GradientBoostingRegressor(criterion='mse',
                                            loss='absolute_error', max_depth=20,
                                            min_samples_leaf=20,
                                            min_samples_split=15,
                                            n_estimators=750, tol=0.001))]),
 'train_score': 0.3558182825292002,
 'test_score': 0.3466649990038641,
 'avg_score': 0.3512416407665322}
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
    save_best_estimator,
    load_best_estimator,
)

