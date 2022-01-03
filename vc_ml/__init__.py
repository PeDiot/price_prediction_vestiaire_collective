"""Description. 

Machine Learning library to fit models on Vestiaire Collective data.

Example: 
In [1]: from vc_ml import SplitData
In [2]: s = SplitData(file_name='vc_data_cleaned.pkl')
In [3]: X = s.get_feature_vector()
In [4]: y = s.get_targets()
In [5]: X_train, X_test, y_train, y_test = s.split(X, y)
In [6]: s.save(
    ...: ... X=X_train,
    ...: ... y=y_train,
    ...: ... file_name='train.pkl'
    ...: ... )
In [7]: s.save(
    ...: X=X_test,
    ...: y=y_test,
    ...: file_name='test.pkl'
    ...: )

In [1]: from vc_ml import (
   ...: load_config,
   ...: load_data,
   ...: train_models
   ...: )

In [2]: config = load_config(file_name='config_init.yaml')

In [3]: X_tr, y_tr = load_data(file_name='train.pkl')

In [4]: train_models(
   ...: X_tr=X_tr,
   ...: y_tr=y_tr,
   ...: config=config
   ...: )
[Parallel(n_jobs=7)]: Using backend LokyBackend with 7 concurrent workers.
...
[Parallel(n_jobs=7)]: Done   5 out of   5 | elapsed:  1.3min finished
Out[4]: 
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
                  ('model',
                   MLPRegressor(hidden_layer_sizes=[100], max_iter=1000))])],
 'train_score': [0.0,
  0.27561146372517936,
  0.2754115644408693,
  0.9744985506246093,
  0.8767106900430065,
  0.5537342215324665,
  0.4269886698222124],
 'test_score': [-0.0002750862547341804,
  0.2735945152729668,
  0.275715773342255,
  -0.49164590153689236,
  0.23452220869275164,
  0.2912611455559938,
  0.38442793820220944]}
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
    get_cv_results,
    get_pipeline_scores, 
    get_best_pipeline, 
)

