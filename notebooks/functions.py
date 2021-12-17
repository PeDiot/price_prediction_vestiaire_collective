"""Description.

Functions used for data exploration. 
"""

import pandas as pd
import numpy as np
from typing import Optional
import plotly.express as px

def feature_prop_table(feature: str, data:pd.DataFrame): 
    """Return a propotion table for a given feature in the dataset."""
    keys = data[feature].value_counts().keys().tolist()
    counts = data[feature].value_counts().values.tolist()
    frequencies = data[feature].value_counts(normalize=True).values.tolist()
    cum_frequencies = np.cumsum(frequencies)
    d = {
            feature: keys, 
            "count": counts, 
            "freq": frequencies, 
            "cumul_freq": cum_frequencies
    }
    return pd.DataFrame.from_dict(d)

def get_quantiles_by_group(
    feature: str, 
    data: pd.DataFrame, 
    target: str="lprice"
): 
    """Return target's quantiles in the categories of a given features."""
    def rename(newname):
        def decorator(f):
            f.__name__ = newname
            return f
        return decorator

    def q_at(y):
        @rename(f'q{y:0.2f}')
        def q(x):
            return x.quantile(y)
        return q

    f = {target: [q_at(0.25), "median", q_at(0.75)]}

    return data.groupby(feature).agg(f).sort_values(
        by=(target, "median"), 
        ascending=False
    )

def target_boxplot(
    feature: str, 
    data:pd.DataFrame, 
    target: str = "lprice", 
    title:Optional[str]=None
): 
    """Plot the target's repartition depending on the feature's category."""
    if title is None: 
        title = f"Price distribution by {feature}"
    if target == "lprice": 
        yaxis_label = "Log Price"
    else: 
        yaxis_label = "Price"
    fig = px.box(
        data, 
        x=feature, 
        y=target, 
        title=title, 
        labels={
            feature: "",
            target: yaxis_label
        }
    )
    fig.show()
