"""Description.

Sub-library to create ItemAttrs objects with all items' characteristics.

Example: 

>>> import VCScraping as vcs
>>> brand = "gucci"
>>> page_no = 2
>>> parser = vcs.DescriptionParser(brand, page_no)
>>> parser
DescriptionParser(brand=gucci, page_no=2)
>>> items = parser.to_ItemAttrs()
>>> items
[
    ItemAttrs(
        online_date='11/11/2021', 
        gender='men', 
        category='bags',
        sub_category='belt bags',
        designer='gucci',
        condition='very good condition',
        material='cloth',
        color='black',
        size='no size',
        location='france', id=19176729),
    ..., 
    ItemAttrs(
        online_date='07/11/2021', 
        gender='women',
        category='bags',
        sub_category='handbags',
        designer='gucci',
        condition='good condition',
        material='leather',
        color='black',
        size='no size'
        location='italy',
        id=19053366
    )
]
>>> parser.save(items)
File saved at ./backup/brands/gucci/items/items_attrs_p2.json.

"""

from .methods import (
    BRANDS_DIR, 
    load_json,
    save_json, 
)
from .brand_page import BasicItem
from .item_page import ItemDesc

from bs4 import BeautifulSoup as bs 

from typing import Dict, List, Tuple 
from serde import serialize, deserialize
from serde.json import from_json, to_json
from dataclasses import dataclass

import os 
from datetime import datetime


@serialize
@deserialize
@dataclass
class ItemAttrs: 
    """Object storing each feature describing items."""
    online_date: str 
    gender: str 
    category: str 
    sub_category: str 
    designer: str 
    condition: str 
    material: str
    color: str 
    size: str 
    location: str 
    id: int 

class DescriptionParser: 
    """Build ItemAttrs objects from item's description page code."""

    def __init__(self, brand: str, page_no: int) -> None:
        self._brand = brand 
        self._page_no = page_no 
        self._path = f"{BRANDS_DIR}/{self._brand}/items/items_desc_p{self._page_no}.json"
        self.data = load_json(
            file_name=self._path, 
            data_type=List[ItemDesc]
        )

    def __repr__(self) -> str:
        return f"DescriptionParser(brand={self._brand}, page_no={self._page_no})"

    @staticmethod
    def _get_features_and_values(item: ItemDesc) -> Dict: 
        """Return a dictionnary of description features and values."""
        if item.description != "None": 
            d = dict() 
            soup = bs(item.description, "lxml")
            for attr in soup.find_all(name="li"):
                feature = attr.text.split(":")[0].lower().strip()
                val = attr.text.split(":")[1].lower().strip()
                if feature == "bracelet material": 
                    d["material"] = val 
                else:
                    d[feature] = val
            return d

    @staticmethod
    def _has_size_desc(features: List[str]) -> bool: 
        """Verify if the item's description contains a size description."""
        if "size" in features: 
            return True
        return False

    @staticmethod
    def _parse_condition_desc(condition_desc: str) -> str: 
        """Remove the substring 'conditionMoreinfo' from the product's condition value."""
        return condition_desc.replace("more info", "").strip()

    @staticmethod
    def _parse_size_desc(size_desc: str) -> int: 
        """Identify the correct size in the product's size value."""
        return size_desc.split(" ")[0]

    @staticmethod
    def _parse_location_desc(loc_desc: str) -> str: 
        """Identify the country in the location description."""
        return loc_desc.split(",")[0]

    def to_ItemAttrs(self) -> ItemAttrs: 
        """Convert the lists of description page code to lists of ItemAttrs object."""
        items = list()
        for item in self.data:
            description_dict = self._get_features_and_values(item=item)
            if description_dict != None: 
                if self._has_size_desc( list(description_dict.keys()) ): 
                    size = self._parse_size_desc(description_dict["size"])
                else: 
                    size = "no size"
                items.append(
                    ItemAttrs(
                        online_date=description_dict["online since"], 
                        gender=description_dict["categories"], 
                        category=description_dict["category"],
                        sub_category=description_dict["sub-category"], 
                        designer=description_dict["designer"], 
                        condition=self._parse_condition_desc(description_dict["condition"]), 
                        material=description_dict["material"], 
                        color=description_dict["color"], 
                        size=size, 
                        location=self._parse_location_desc(description_dict["location"]), 
                        id=int(description_dict["reference"]) 
                    )
                )
            else: 
                pass
        return items 

    def save(self, items: List[ItemAttrs]): 
        """Save a list of ItemAttrs objects."""
        path = f"{BRANDS_DIR}/{self._brand}/items/items_attrs_p{self._page_no}.json"
        save_json(
            data=items, 
            file_name=path 
        )
        print(f"File saved at {path}.")


    

