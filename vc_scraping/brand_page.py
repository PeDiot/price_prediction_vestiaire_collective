"""Description.

Sub-library to collect data on items from the selected pages for each brand.

"""

from .methods import (
    load_json, 
    save_json, 
    BRANDS_DIR
)
from .home_page import get_brands
from .brands import BrandPage

import os

from typing import Dict, List
from serde import serialize, deserialize
from serde.json import from_json, to_json
from dataclasses import dataclass

from bs4 import BeautifulSoup as bs
import re

INT_MOTIF = re.compile(pattern="[0-9]+")

def find_paths(brand_name: str) -> str:
    """Return the paths to the file containing data on the first 10 pages for each brand.
    
    Example: 
    >>> vcs.find_paths(brand_name="balenciaga")
    ['./backup/brands/balenciaga/p1.json', './backup/brands/balenciaga/p10.json', ..., './backup/brands/balenciaga/p9.json'] 
    """ 
    paths = []
    sub_dir = BRANDS_DIR + "/" + brand_name
    pages = os.listdir(sub_dir)
    for page in pages: 
        if page != "items":
            path = sub_dir + "/" + page
            paths.append(path)
    return paths

def get_brands_files_paths() -> Dict[str, List[str]]: 
    """Return a dictionnary matching each brand with its corresponding files' paths.
    
    Example: 
    >>> brands_files_paths = vcs.get_brands_files_paths()
    >>> brands_files_paths["balenciaga"]
    ['./backup/brands/balenciaga/p1.json', './backup/brands/balenciaga/p10.json', ..., './backup/brands/balenciaga/p9.json']
    """
    d = dict()
    for brand_name in get_brands(): 
        d[brand_name] = find_paths(brand_name)
    return d

def create_items_dir(brand_name: str): 
    """Create an 'items' sub-directory in the brand's directory.
    
    Example: 
    >>> vcs.create_items_dir(brand_name="balenciaga")
    "./backup/brands/balenciaga/items already exists.
    """
    path = f"./backup/brands/{brand_name}/items"
    if os.path.exists(path): 
        print(f"{path} already exists.")
    else: 
        os.mkdir(path)
        print(f"{path} has been created.")

@serialize
@deserialize
@dataclass
class BasicItem: 
    """Dataclass to store basic information on an item located on a brand's page ."""
    id: int 
    url: str 
    num_likes: int 
    price: float
    we_love_tag: bool 

class PageParser:
    """Methods to retrieve basic information (url, number of likes, price) on items located on a specific page for a selected brand.""" 

    def __init__(self, page_dir: str):
        self._page_dir = page_dir
        self._page = load_json(file_name=page_dir, data_type=BrandPage)
        self.page_items = self._get_items_from_page()

    def _get_items_from_page(self): 
       """Return each item's source code from the page."""
       soup = bs(self._page.catalog, "lxml")
       return soup.find_all(name="li")
    
    def _get_item_url(self, item: bs) -> str: 
        """Get the item's page url."""
        href = item.find(
            name="a", 
            attrs={
                "itemprop": "url", 
                "rel": "noopener",
                "target": "_self"
            }
        ).get("href")
        return "https://fr.vestiairecollective.com/" + href

    @staticmethod
    def _find_unique_id(url: str) -> int: 
        """Find the item's unique ID in the item's url."""
        return int( INT_MOTIF.findall(url)[0] )

    @staticmethod
    def _get_num_likes(item: bs) -> int: 
        """Return the item's number of likes."""
        return int(
            item.find(
                name="div",
                attrs={"class": "likeSnippet__count"}
            ).text.strip()
        )

    @staticmethod
    def _get_price(item: bs) -> float: 
        """Return the item's price."""
        prices = item.find(
            name="span",
            attrs={"class": ["productSnippet__price", "productSnippet__price--discount"]}
        ).text.split(sep=" ")
        return float( prices[-1].replace("€", "") )

    @staticmethod
    def _has_we_love_tag(item: bs) -> bool: 
        """Check whether the item has a 'we love' tag."""
        if item.find(name="span", attrs="productSnippet__tags__item") == None: 
            return False
        return True

    def to_basic_items(self): 
        """Return a list of items' basic information."""
        basic_items = list()
        for item in self.page_items: 
            url=self._get_item_url(item)
            basic_items.append(
                BasicItem(
                    id=self._find_unique_id(url), 
                    url=url, 
                    num_likes=self._get_num_likes(item), 
                    price = self._get_price(item), 
                    we_love_tag=self._has_we_love_tag(item)
                )
            )
        return basic_items

    def get_backup_file_name(self, page_no: int): 
        """Return the correct file name to store the list of basic items."""
        brand_dir = os.path.dirname(self._page_dir)
        return f"{brand_dir}/items/basic_items_p{page_no}.json"

def save_all_basic_items(brands_files_paths: Dict[str, List[str]]): 
    """Save list of basic items for each page, for each brand."""
    for brand, pages_paths in brands_files_paths.items(): 
        print(f"Processing {brand}...")
        create_items_dir(brand_name=brand)
        for page_no, page_dir in enumerate(pages_paths): 
            parser = PageParser(page_dir)
            basic_item = parser.to_basic_items()
            file_name = parser.get_backup_file_name(page_no)
            save_json(
                data=basic_item, 
                file_name=file_name
            )
        print(f"{brand} processed.")
        print(f"*"*30)


