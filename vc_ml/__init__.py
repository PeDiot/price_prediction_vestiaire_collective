"""Description. 

Machine Learning library to fit models on Vestiaire Collective data.

Example: 
In [1]: from vc_ml import SplitData
In [2]: s = SplitData(file_name='vc_data_cleaned.pkl')
In [3]: X = s.get_feature_vector()
In [4]: y = s.get_targets()
In [5]: X_train, X_test, y_train, y_test = s.split(X, y)
In [6]: s.save(
   ...: X=X_train,
   ...: y=y_train,
   ...: file_name='train.pkl'
   ...: )
In [7]: s.save(
    ...: X=X_test,
    ...: y=y_test,
    ...: file_name='test.pkl'
    ...: )

In [2]: from vc_ml import (
   ...:    ...: load_config,
   ...:    ...: load_data,
   ...:    ...: train_models
   ...:    ...: )

In [3]: config = load_config(file_name='config_init.yaml')

In [4]:

In [4]: X_tr, y_tr = load_data(file_name='train.pkl')

In [5]: X_tr
Out[5]: 
array([['women', 'shoes', 'heels', ..., 'pink', 'size_40', 'eu'],
       ['women', 'shoes', 'sandals', ..., 'red', 'size_37', 'eu'],
       ['women', 'shoes', 'sandals', ..., 'pink', 'size_39', 'eu'],
       ...,
       ['women', 'shoes', 'flats', ..., 'silver', 'size_40', 'eu'],
       ['women', 'shoes', 'heels', ..., 'silver', 'size_38', 'uk'],
       ['women', 'clothing', 'coats', ..., 'navy', 'size_xxl', 'eu']],
      dtype=object)

In [6]: y_tr
Out[6]: array([ 126.  ,  450.  ,  470.  , ...,  170.  ,  346.17, 1800.  ])

In [7]: train_models(
   ...:    ...: X_tr=X_tr,
   ...:    ...: y_tr=y_tr,
   ...:    ...: config=config
   ...:    ...: )
estimator: DummyRegressor() - param_grid: {'strategy': 'mean'}
[Parallel(n_jobs=7)]: Using backend LokyBackend with 7 concurrent workers...
[Parallel(n_jobs=7)]: Done   5 out of   5 | elapsed:   11.1s finished
Out[7]: 
{'pipeline': [Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('model', DummyRegressor())]),
  Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('model', LinearRegression())]),
  Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('model', Ridge())]),
  Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('model', DecisionTreeRegressor(max_features='auto'))]),
  Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('model',
                   RandomForestRegressor(max_samples=1.0, oob_score=True))]),
  Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('model', GradientBoostingRegressor(tol=0.001))]),
  Pipeline(steps=[('enc',
                   OneHotEncoder(drop='first', handle_unknown='ignore',
                                 sparse=False)),
                  ('model', MLPRegressor(hidden_layer_sizes=[100]))])],
 'train_score': [0.0,
  0.27561146372517936,
  0.2754115644408693,
  0.9744985506246093,
  0.8788006464491552,
  0.5537342215324665,
  0.27253550156933865],
 'test_score': [-0.0002750862547341804,
  0.2735945152729668,
  0.275715773342255,
  -0.14087415022162816,
  0.2513063008716697,
  0.29123571363147027,
  0.29035041766619446]}
"""

from .data import (
    BACKUP, 
    Target, 
    SplitData, 
    load_data,
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

