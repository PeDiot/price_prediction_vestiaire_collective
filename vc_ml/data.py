"""Decription. 

Automate the train/test split process. 

Example: 
In [1]: from vc_ml import (
   ...: read_data,
   ...: to_dummies,
   ...: SplitData,
   ...: Target
   ...: )

In [2]: data = read_data(file_path="./data/vc_data_cleaned.pkl")

In [3]: data.head()
Out[3]: 
          num_likes   price  we_love_tag gender  category sub_category  ...            condition        material  color     size location    lprice   
id                                                                      ...
19126896          7  180.00            1  women     shoes        boots  ...  very_good_condition         leather  black  size_39       eu  5.198497   
19181389          1   40.55            1  women  clothing        jeans  ...       good_condition     denim_jeans   navy   size_m       eu  3.726898   
19182029          6  332.50            1    men  clothing        coats  ...       good_condition            wool  black   size_l       eu  5.809643   
19132670          3   45.00            0    men  clothing        jeans  ...           never_worn          cotton   grey   size_m       eu  3.828641   
19118182          9  105.00            0  women  clothing      dresses  ...  very_good_condition  other_material  black   size_s       eu  4.663439   

[5 rows x 13 columns]

In [4]: data = data.drop(
   ...: labels=["category"],
   ...: axis=1
   ...: )

In [5]: data.columns
Out[5]: 
Index(['num_likes', 'price', 'we_love_tag', 'gender', 'sub_category',
       'designer', 'condition', 'material', 'color', 'size', 'location',
       'lprice'],
      dtype='object')

In [6]: new_data = to_dummies(data)

In [7]: new_data.head()
Out[7]:
          num_likes   price  we_love_tag    lprice  women  belts  boots  coats  ...  size_m  size_s  size_xl  size_xs  size_xxl  size_xxs  eu  uk     
id                                                                              ...
19126896          7  180.00            1  5.198497      1      0      1      0  ...       0       0        0        0         0         0   1   0     
19181389          1   40.55            1  3.726898      1      0      0      0  ...       1       0        0        0         0         0   1   0     
19182029          6  332.50            1  5.809643      0      0      0      1  ...       0       0        0        0         0         0   1   0     
19132670          3   45.00            0  3.828641      0      0      0      0  ...       1       0        0        0         0         0   1   0     
19118182          9  105.00            0  4.663439      1      0      0      0  ...       0       1        0        0         0         0   1   0     

[5 rows x 117 columns]

In [8]: s = SplitData(data=new_data)

In [9]: s
Out[9]:
SplitData(data==          num_likes   price  we_love_tag    lprice  women  belts  boots  coats  ...  size_m  size_s  size_xl  size_xs  size_xxl  size_xxs  eu  uk
id                                                                              ...
19126896          7  180.00            1  5.198497      1      0      1      0  ...       0       0        0        0         0         0   1   0     
19181389          1   40.55            1  3.726898      1      0      0      0  ...       1       0        0        0         0         0   1   0     
19182029          6  332.50            1  5.809643      0      0      0      1  ...       0       0        0        0         0         0   1   0     
19132670          3   45.00            0  3.828641      0      0      0      0  ...       1       0        0        0         0         0   1   0     
19118182          9  105.00            0  4.663439      1      0      0      0  ...       0       1        0        0         0         0   1   0     
...             ...     ...          ...       ...    ...    ...    ...    ...  ...     ...     ...      ...      ...       ...       ...  ..  ..     
19201767          1   95.00            0  4.564348      1      0      0      0  ...       0       0        0        0         0         0   1   0     
19062770          4   44.00            1  3.806662      1      0      0      0  ...       0       0        0        0         0         0   1   0     
19210693         15   80.00            0  4.394449      1      1      0      0  ...       0       0        0        1         0         0   1   0     
18970201         46  162.00            1  5.093750      1      0      0      0  ...       0       0        0        0         0         0   1   0     
19216508          0  173.80            0  5.163642      1      0      0      0  ...       0       0        0        0         1         0   1   0     

[9572 rows x 117 columns], targets=['price', 'lprice', 'we_love_tag', 'num_likes'], test_prop=0.3)

In [10]: X = s.get_feature_vector()

In [11]: X
Out[11]:
array([[1, 0, 1, ..., 0, 1, 0],
       [1, 0, 0, ..., 0, 1, 0],
       [0, 0, 0, ..., 0, 1, 0],
       ...,
       [1, 1, 0, ..., 0, 1, 0],
       [1, 0, 0, ..., 0, 1, 0],
       [1, 0, 0, ..., 0, 1, 0]], dtype=uint8)

In [12]: y = s.get_targets()

In [13]: X_train, X_test, y_train, y_test = s.split(X, y)

In [14]: s.save(
    ...: X=X_train,
    ...: y=y_train,
    ...: file_path="./data/train.pkl"
    ...: )

In [15]: s.save(
    ...: X=X_test,
    ...: y=y_test,
    ...: file_name="./data/test.pkl"
    ...: )
"""

from _pytest.python_api import raises
import pandas as pd
import numpy as np
from pickle import load, dump 

from enum import Enum 
from typing import (
    Dict, 
    Tuple, 
    List
)

from sklearn.model_selection import train_test_split

BACKUP = "./backup/"

class Target(Enum): 
    PRICE = "price"
    LPRICE = "lprice"
    WE_LOVE_TAG = "we_love_tag"
    NUM_LIKES = "num_likes"

def read_data(file_path: str) -> pd.DataFrame: 
    """Read the data file."""
    if file_path[-3:] != "pkl": 
        raise ValueError("Only .pkl files can be read.")
    file_path = BACKUP + file_path
    with open(file_path, "rb") as file: 
        data = load(file)
    return data 

def to_dummies(data: pd.DataFrame) -> pd.DataFrame: 
    """Convert categorical variables into dummies."""
    return pd.get_dummies(
        data=data, 
        columns=[
            "gender",
            "sub_category",
            "designer",
            "condition",
            "material",
            "color",
            "size",
            "location"
        ], 
        drop_first=True, 
        prefix="", 
        prefix_sep=""
    )

class SplitData: 
    """Build and save training and testing sets."""
    def __init__(
        self,
        data: pd.DataFrame, 
        test_prop: float = .3
     ):
        self.data = data 
        self._test_prop = test_prop 
        self._targets = [
            target.value 
            for target in (
                Target.PRICE, 
                Target.LPRICE, 
                Target.WE_LOVE_TAG, 
                Target.NUM_LIKES
            )
        ]
        if self._test_prop <= 0 or self._test_prop >= 1: 
            raise ValueError("Test proportion needs to be between 0 and 1.")

    def __repr__(self) -> str:
        return f"SplitData(data=={self.data}, targets={self._targets}, test_prop={self._test_prop})"

    def get_feature_vector(self) -> np.ndarray: 
        """Return an array of features."""
        return np.array(
            self.data.drop(
                labels=self._targets, 
                axis=1
            ).values
        )

    def get_targets(self) -> np.ndarray: 
        """Return an array of target variables."""
        return np.array(
            self.data.loc[
                :, 
                self._targets
            ].values
        ) 

    def split(
        self, 
        X: np.ndarray,
        y: np.ndarray,
        random_state: float = 42
    ) -> Tuple[np.ndarray]: 
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
        if file_name[-3:] != "pkl": 
            raise ValueError("Only .pkl files can be saved.")
        data =  {
            "X": X, 
            "y": self._make_dict_of_targets(y)
        }
        file_path = BACKUP + "data/" + file_name
        with open(file_path , "wb") as file:
            dump(obj=data, file=file) 

def load_feature_vector(file_path: str) -> np.ndarray: 
    """Return X array."""
    data = read_data(file_path)
    return data["X"]

def load_target(
    file_path: str, 
    target: Target = Target.PRICE
) -> np.ndarray: 
    """Return y array."""
    if not isinstance(target, Target):
        raise ValueError("'target' must be of class 'Target'.")
    data = read_data(file_path)
    return data["y"][target.value]

