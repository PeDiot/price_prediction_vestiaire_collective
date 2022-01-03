"""Decription. 

Automate the train/test split process. 

Example: 
In [1]: from vc_ml import SplitData

In [2]: s = SplitData(file_name="vc_data_cleaned.pkl")

In [3]: s
Out[3]: SplitData(file_name==vc_data_cleaned.pkl, targets=['price', 'lprice', 'we_love_tag', 'num_likes'], test_prop=0.3)

In [4]: s.data
Out[4]: 
          num_likes   price  we_love_tag gender     category sub_category            designer            condition        material  color      size location    lprice
id
19126896          7  180.00            1  women        shoes        boots        acne studios  very_good_condition         leather  black   size_39       eu  5.198497
19181389          1   40.55            1  women     clothing        jeans        acne studios       good_condition     denim_jeans   navy    size_m       eu  3.726898
19182029          6  332.50            1    men     clothing        coats        acne studios       good_condition            wool  black    size_l       eu  5.809643
19132670          3   45.00            0    men     clothing        jeans        acne studios           never_worn          cotton   grey    size_m       eu  3.828641
19118182          9  105.00            0  women     clothing      dresses        acne studios  very_good_condition  other_material  black    size_s       eu  4.663439
...             ...     ...          ...    ...          ...          ...                 ...                  ...             ...    ...       ...      ...       ...
19201767          1   95.00            0  women         bags   small_bags  yves saint laurent           never_worn           cloth  black   no_size       eu  4.564348
19062770          4   44.00            1  women  accessories      scarves  yves saint laurent  very_good_condition       polyester   navy   no_size       eu  3.806662
19210693         15   80.00            0  women  accessories        belts  yves saint laurent  very_good_condition         leather   blue   size_xs       eu  4.394449
18970201         46  162.00            1  women         bags     handbags  yves saint laurent  very_good_condition       synthetic   pink   no_size       eu  5.093750
19216508          0  173.80            0  women     clothing      jackets  yves saint laurent  very_good_condition          cotton  camel  size_xxl       eu  5.163642

[9572 rows x 13 columns]

In [5]: X = s.get_feature_vector()

In [6]: y = s.get_targets()

In [7]: X_train, X_test, y_train, y_test = s.split(X, y)

In [8]: s._make_dict_of_targets(y_train)
Out[8]:
{'price': array([ 126.  ,  450.  ,  470.  , ...,  170.  ,  346.17, 1800.  ]),
 'lprice': array([4.84418709, 6.11146734, 6.15485809, ..., 5.14166356, 5.84981457,
        7.49609735]),
 'we_love_tag': array([0., 0., 0., ..., 1., 0., 0.]),
 'num_likes': array([ 6., 16.,  2., ..., 16.,  1.,  6.])}

In [9]: s.save(
   ...: X=X_train,
   ...: y=y_train,
   ...: file_name="train.pkl"
   ...: )

In [10]: s.save(
    ...: X=X_test,
    ...: y=y_test,
    ...: file_name="test.pkl"
    ...: )
"""

import pandas as pd
import numpy as np
from pickle import load, dump 

from enum import Enum 
from typing import Dict, List

from sklearn.model_selection import train_test_split

BACKUP = "./backup/"

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
        file_path = BACKUP + "data/" + self._file_name
        with open(file_path, "rb") as file: 
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
        file_path = BACKUP + "data/" + file_name
        with open(file_path , "wb") as file:
            dump(obj=data, file=file) 

def load_data(file_name: str, target: Target = Target.PRICE): 
    """Return X and y arrays."""
    file_path = BACKUP + "data/" + file_name 
    with open(file_path , "rb") as file:
        data = load(file) 
    return data["X"], data["y"][target.value] 
