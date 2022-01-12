"""Description.

Test the data script from the "vc_ml" library.
"""

import pytest 

import numpy as np
import pandas as pd 

from vc_ml import (
    to_dummies, 
    Target, 
    SplitData, 
)

@pytest.fixture
def simulated_data():
    """Generate a fictional data set."""
    d = {
        "num_likes": [10, 5, 89, 12, 6], 
        "price": [121.8, 500.0, 16.3, 348.0, 1670.9], 
        "we_love_tag": [0, 0, 0, 0, 1], 
        "gender": ["Women", "Women", "Men", "Women", "Men"], 
        "category": ["accessories", "bags", "clothing", "clothing", "shoes"], 
        "sub_category": ["belts", "handbags", "coats", "trousers", "trainers"], 
        "designer": ["gucci", "hermès", "givenchy", "fendi", "balenciaga"], 
        "condition": ["fair_condition", "good_condition", "fair_condition", "never_worn", "very_good_condition"], 
        "material": ["leather", "patent_leather", "cotton", "viscose", "plastic"], 
        "color": ["red", "brown", "beige", "purple", "beige"], 
        "size": ["no_size", "no_size", "size_m", "size_xxs", "size_44"], 
        "location": ["eu", "eu", "eu", "uk", "other_country"], 
        "lprice": [np.log(price+1) for price in [121.8, 500.0, 16.3, 348.0, 1670.9]]
    }
    data = pd.DataFrame.from_dict(d)
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = pd.Categorical(data[col])
    return data

def test_to_dummies(simulated_data): 
    """Test the categorical variables encoding."""
    data_enc = to_dummies(simulated_data)
    columns = [
        "num_likes",
        "price",
        "we_love_tag",
        "category",
        "lprice",
        "Women",
        "coats",
        "handbags",
        "trainers",
        "trousers",
        "fendi",
        "givenchy",
        "gucci",
        "hermès",
        "good_condition",
        "never_worn",
        "very_good_condition",
        "leather",
        "patent_leather",
        "plastic",
        "viscose",
        "brown",
        "purple",
        "red",
        "size_44",
        "size_m",
        "size_xxs",
        "other_country",
        "uk"
    ]
    shape = (simulated_data.shape[0], 29)
    assert data_enc.columns.tolist() == columns
    assert data_enc.shape == shape 

def test_split_data_instanciation(simulated_data):
    """Instanciation testing for SplitData."""
    s = SplitData(data=simulated_data)
    assert isinstance(s, SplitData)

def test_split_data_test_prop(simulated_data):
    """Test SplitData for negative testing proportion."""
    with pytest.raises(ValueError):
        SplitData(data=simulated_data, test_prop=-.1)
    with pytest.raises(ValueError):
        SplitData(data=simulated_data, test_prop=2.)

def test_get_feature_vector(simulated_data):
    """Test the function which outputs the feature vector."""
    s = SplitData(data=simulated_data)
    X = s.get_feature_vector()
    assert type(X) == np.ndarray
    assert X.shape == (5, 9)

def test_get_targets(simulated_data):
    """Test the function which outputs the targets vector."""
    s = SplitData(data=simulated_data)
    y = s.get_targets()
    assert type(y) == np.ndarray
    assert y.shape == (5, 4)

def test_split_function(simulated_data): 
    """Test the split function in SplitData."""
    s = SplitData(data=simulated_data)
    X, y = s.get_feature_vector(), s.get_targets()
    X_tr, X_te, y_tr, y_te = s.split(X, y)
    assert X_tr.shape[0] + X_te.shape[0] == X.shape[0]
    assert X_tr.shape[0] == y_tr.shape[0]
    assert X_te.shape[0] == y_te.shape[0] 

def test_make_dict_of_targets(simulated_data): 
    """Test target variables storage in dict."""
    s = SplitData(data=simulated_data)
    y = s.get_targets() 
    s = SplitData(data=simulated_data)
    targets_dict = s._make_dict_of_targets(y)
    assert list(targets_dict.keys()) == [target.value for target in Target] 


