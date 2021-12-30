"""Description. 

Machine Learning library to fit models on Vestiaire Collective data.

Example: 
In [1]: from vc_ml import (
   ...: load_data,
   ...: load_config,
   ...: Training,
   ...: Target,
   ...: )
In [2]: config = load_config()
In [3]: X_tr, y_tr = load_data(file_name="train.pkl", target=Target.PRICE)
In [4]: training = Training(
   ...: X=X_tr,
   ...: y=y_tr,
   ...: config=config,
   ...: cv=3
   ...: )
In [5]: training
Out[5]: 
Training(X=[[0 1 0 ... 0 1 0]
 [0 1 0 ... 0 0 0]
 [0 1 1 ... 0 1 0]
 ...
 [0 1 0 ... 0 1 0]
 [0 1 0 ... 0 1 0]
 [1 0 0 ... 0 1 0]], y=[300.    50.35 125.   ...  85.   325.   249.  ], config=Config(lr=LREstimator(fit_intercept=[True]), ridge=RidgeEstimator(alpha=[1.0], fit_intercept=[True]), tree=TreeEstimator(max_depth=[None], min_samples_split=[2], min_samples_leaf=[1], max_features=['auto'], ccp_alpha=[0.0]), rf=RFEstimator(n_estimators=[100], max_depth=[None], min_samples_split=[2], min_samples_leaf=[1], max_features=['auto'], max_samples=[1.0], oob_score=[True]), gb=GBEstimator(n_estimators=[100], min_samples_split=[2], min_samples_leaf=[1], max_depth=[3], loss=['squared_error'], learning_rate=[0.1], criterion=['friedman_mse'], tol=[0.001]), mlp=MLPEstimator(hidden_layer_sizes=[[100]], max_iter=[200], activation=['relu'], solver=['adam'], learning_rate_init=[0.001])), cv=3, n_pcs=[40])
In [6]: p = training.make_pipeline()

In [7]: g = training.build_grid_search(p)

In [8]: g
Out[8]: 
GridSearchCV(cv=3,
             estimator=Pipeline(steps=[('pca', PCA()),
                                       ('model', EstimatorSwitcher())]),
             n_jobs=7,
             param_grid=[{'model__estimator': [LinearRegression()],
                          'model__estimator__fit_intercept': [True],
                          'pca__n_components': [40]},
                         {'model__estimator': [Ridge()],
                          'model__estimator__alpha': [1.0],
                          'model__estimator__fit_intercept': [True],
                          'pca__n_components': [40]},
                         {'...
                          'model__estimator__tol': [0.001],

In [9]: training.fit(g)
Fitting 3 folds for each of 6 candidates, totalling 18 fits
...         
In [10]: from vc_ml import display_results

In [11]: display_results(g)
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ model__estimator       ┃ model__estimator__act… ┃ model__estimator__hid… ┃ model__estimator__lea… ┃ model__estimator__max… ┃ model__estimator__sol… ┃ pca__n_components ┃ Score               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ MLPRegressor(hidden_l… │ relu                   │ [100]                  │ 0.001                  │ 200                    │ adam                   │ 40                │ 0.26610131582485597 │
└────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────┴────────────────────────┴───────────────────┴─────────────────────┘
In [12]: from vc_ml import save_grid_search

In [13]: save_grid_search(g_fitted=g, file_name="grid_search_init.pkl")
"""

from .data import (
    BACKUP, 
    Target, 
    SplitData, 
    load_data,
) 

from .estimators import (
    EstimatorSwitcher, 
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
    Training, 
    save_grid_search, 
    display_results, 
)

