"""Decription. 

Automate the train/test split process. 

Example: 
>>> from vc_ml import SplitData
>>> s = SplitData(file_name="vc_data_dummies.pkl")
>>> s.data
        num_likes   price  we_love_tag    lprice  men  women  bags  clothing  shoes  belts  boots  ...  size_37  size_38  size_39  size_40  size_41  size_42  size_43  size_44  size_>=45  eu  uk
id                                                                                                   ...                                                                                         

19126896          7  180.00            1  5.198497    0      1     0         0      1      0      1  ...        0        0        1        0        0        0        0        0          0   1   0
19181389          1   40.55            1  3.726898    0      1     0         1      0      0      0  ...        0        0        0        0        0        0        0        0          0   1   0
19182029          6  332.50            1  5.809643    1      0     0         1      0      0      0  ...        0        0        0        0        0        0        0        0          0   1   0
19132670          3   45.00            0  3.828641    1      0     0         1      0      0      0  ...        0        0        0        0        0        0        0        0          0   1   0
19118182          9  105.00            0  4.663439    0      1     0         1      0      0      0  ...        0        0        0        0        0        0        0        0          0   1   0
...             ...     ...          ...       ...  ...    ...   ...       ...    ...    ...    ...  ...      ...      ...      ...      ...      ...      ...      ...      ...        ...  ..  ..
19201767          1   95.00            0  4.564348    0      1     1         0      0      0      0  ...        0        0        0        0        0        0        0        0          0   1   0
19062770          4   44.00            1  3.806662    0      1     0         0      0      0      0  ...        0        0        0        0        0        0        0        0          0   1   0
19210693         15   80.00            0  4.394449    0      1     0         0      0      1      0  ...        0        0        0        0        0        0        0        0          0   1   0
18970201         46  162.00            1  5.093750    0      1     1         0      0      0      0  ...        0        0        0        0        0        0        0        0          0   1   0
19216508          0  173.80            0  5.163642    0      1     0         1      0      0      0  ...        0        0        0        0        0        0        0        0          0   1   0

[9694 rows x 91 columns]
>>> s
SplitData(file_name==vc_data_dummies.pkl, targets=['price', 'lprice', 'we_love_tag', 'num_likes'], test_prop=0.3)
>>> X = s.get_feature_vector()
>>> X
array([[0, 1, 0, ..., 0, 1, 0],
    [0, 1, 0, ..., 0, 1, 0],
    [1, 0, 0, ..., 0, 1, 0],
    ...,
    [0, 1, 0, ..., 0, 1, 0],
    [0, 1, 1, ..., 0, 1, 0],
    [0, 1, 0, ..., 0, 1, 0]], dtype=uint8)
>>> y = s.get_targets()
>>> y
array([[180.        ,   5.19849703,   1.        ,   7.        ],
    [ 40.55      ,   3.72689752,   1.        ,   1.        ],
    [332.5       ,   5.80964287,   1.        ,   6.        ],
    ...,
    [ 80.        ,   4.39444915,   0.        ,  15.        ],
    [162.        ,   5.0937502 ,   1.        ,  46.        ],
    [173.8       ,   5.16364246,   0.        ,   0.        ]])
>>> X_train, X_test, y_train, y_test = s.split(X, y)
>>> s._make_dict_of_targets(y_train)
{'price': array([180.  ,  40.55, 332.5 , ...,  80.  , 162.  , 173.8 ]), 'lprice': array([5.19849703, 3.72689752, 5.80964287, ..., 4.39444915, 5.0937502 ,
    5.16364246]), 'we_love_tag': array([1., 1., 1., ..., 0., 1., 0.]), 'num_likes': array([ 7.,  1.,  6., ..., 15., 46.,  0.])}
>>> s.save(
... X=X_train,
... y=y_train,
... file_name="train.pkl"
... )
>>> s.save(
... X=X_test, 
... y=y_test, 
... file_name="test.pkl"
... )
"""

import pandas as pd
import numpy as np
from pickle import load, dump 

from enum import Enum 
from typing import Dict, List

from sklearn.model_selection import train_test_split

BACKUP = "C:/Users/pemma/OneDrive - Université de Tours/Mécen/M2/S1/02 - Machine Learning/05 - Projet/ML_Vestiaire_Collective/backup/"

class Target(Enum): 
    PRICE = "price"
    LPRICE = "lprice"
    WE_LOVE_TAG = "we_love_tag"
    NUM_LIKES = "num_likes"

class SplitData: 
    """Build and save training and testing sets."""
    def __init__(self,
     file_name: str,
     test_prop: float = .3
     ):
        self._file_name = file_name
        self._targets = [
            target.value 
            for target in (
                Target.PRICE, 
                Target.LPRICE, 
                Target.WE_LOVE_TAG, 
                Target.NUM_LIKES
            )
        ]
        self.data = self._read_data()
        self._test_prop = test_prop 
        if self._test_prop <= 0 or self._test_prop >= 1: 
            raise ValueError("Test proportion needs to be between 0 and 1.")

    def _read_data(self) -> pd.DataFrame: 
        """Read the data file."""
        with open(BACKUP+self._file_name, "rb") as file: 
            data = load(file)
        return data 

    def __repr__(self) -> str:
        return f"SplitData(file_name=={self._file_name}, targets={self._targets}, test_prop={self._test_prop})"

    def get_feature_vector(self): 
        """Return an array of features."""
        return self.data.drop(
            labels=self._targets, 
            axis=1
        ).values

    def get_targets(self): 
        """Return an array of target variables."""
        return self.data.loc[
            :, 
            self._targets
        ].values

    def split(self, 
        X: np.ndarray,
        y: np.ndarray,
        random_state: float = 42
    ): 
        """Train/test split."""
        return train_test_split(
            X, y,  
            test_size=self._test_prop, 
            random_state=random_state
        )
    
    def _make_dict_of_targets(self, y: np.ndarray) -> Dict: 
        """Return a dict whith targets as keys and arrays as values."""
        return {
            target: y[:, ix]
            for ix, target in enumerate(self._targets)
        }

    def save(
        self, 
        X: np.ndarray,
        y: np.ndarray, 
        file_name: str
    ):
        """Save split dataset.""" 
        data =  {
            "X": X, 
            "y": self._make_dict_of_targets(y)
        }
        with open(BACKUP+"data"+file_name , "wb") as file:
            dump(obj=data, file=file) 

def load_data(file_name: str, target: Target): 
    """Return X and y arrays."""
    file_path = BACKUP + "data/" + file_name 
    with open(BACKUP+file_name , "rb") as file:
        data = load(file) 
    return data["X"], data["y"][target.value] 
