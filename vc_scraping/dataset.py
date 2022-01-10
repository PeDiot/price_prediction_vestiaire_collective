"""Description.

Sub-library to create the final dataset.

Example:

>>> import vc_scraping as vcs
>>> final_data = vcs.make_dataset()          
Final dataset saved at ./backup/vc_data.pkl.
>>> final_data.head()
         id                                                url  num_likes   price  we_love_tag online_date  ...      designer             condition       material  color size location
0  19126896  https://fr.vestiairecollective.com//women-shoe...          7  180.00         True  09/11/2021  ...  acne studios   very good condition        leather  black   39    italy
1  19181389  https://fr.vestiairecollective.com//women-clot...          1   40.55         True  12/11/2021  ...  acne studios        good condition  denim - jeans   navy   30   poland
2  19182029  https://fr.vestiairecollective.com//men-clothi...          6  332.50         True  12/11/2021  ...  acne studios        good condition           wool  black    l  denmark
3  19132670  https://fr.vestiairecollective.com//men-clothi...          3   45.00        False  09/11/2021  ...  acne studios  never worn, with tag         cotton   grey   28  germany
4  19118182  https://fr.vestiairecollective.com//women-clot...          9  105.00        False  09/11/2021  ...  acne studios   very good condition          linen  black    s  germany

[5 rows x 15 columns]
>>> final_data.shape
(10409, 15)
"""

import pandas as pd 
from.methods import BRANDS_DIR
from .home_page import get_brands

SAVE_PATH = "./backup/data/vc_data.pkl"

def make_dataset(num_pages: int = 6): 
    """Create the final dataset and save to pickle.
    Input: number of pages where each page number is related to a web page on Vestiaire Collective for a given brand.
    Output: dataset without duplicates containing N articles and P features.
    """
    d = pd.DataFrame() 
    for no in range(num_pages+1):
        for brand in get_brands(): 
            basic_items_df = pd.read_json(f"{BRANDS_DIR}{brand}/items/basic_items_p{no}.json")
            items_attrs_df = pd.read_json(f"{BRANDS_DIR}{brand}/items/items_attrs_p{no}.json")
            if items_attrs_df.shape[1] != 0 and basic_items_df.shape[1] != 0: 
                temp = pd.merge(
                    left=basic_items_df, 
                    right=items_attrs_df,
                    on="id"
                )
                d = pd.concat([d, temp])
    d = d.drop_duplicates(subset="id")
    d.to_pickle(path=SAVE_PATH)
    print(f"Final dataset saved at {SAVE_PATH}.")
    return d