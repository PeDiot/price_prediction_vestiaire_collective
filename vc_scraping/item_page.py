"""Description.

Sub-library to collect each item's description for all brands given a specific page.

"""

from os import listdir
from .methods import (
    BRANDS_DIR,
    flatten_list, 
    load_json,
    accept_cookies,
    save_json
)
from .home_page import get_brands
from .brand_page import BasicItem

import os
from typing import List, Dict

from serde import serialize, deserialize
from serde.json import from_json, to_json
import json

from dataclasses import dataclass

from bs4 import BeautifulSoup as bs
from selenium import webdriver

@serialize
@deserialize
@dataclass
class ItemDesc: 
    """Store item's url, brand, page number and description."""
    id: int 
    url: str
    brand: str
    page_no: int
    description: str

class DescriptionScraper: 
    """Methods to collect product description for a list of items, given a brand and a page number."""

    def __init__(self, 
        driver: webdriver, 
        brand: str, 
        page_no: int
    ):
        self._brand = brand 
        self._page_no = page_no
        self.driver = driver

    def _get_items_url_and_id(self)-> Dict[str, List[str]]: 
        """Return the url to collect and the id related to each basic item.."""
        file_path = f"{BRANDS_DIR}{self._brand}/items/basic_items_p{self._page_no}.json"
        basic_items_file = load_json(
            file_name=file_path,
            data_type=List[BasicItem]
        )
        return [(basic_item.id, basic_item.url) for basic_item in basic_items_file]

    def get_items_desc(self) -> List[ItemDesc]: 
        """Return the url, brand and description for each item."""
        items = list()
        for id, url in self._get_items_url_and_id(): 
            self.driver.delete_all_cookies()
            self.driver.get(url) 
            soup = bs(self.driver.page_source, "lxml")
            desc = soup.find(
                name="ul", 
                attrs={"class": ["descriptionList__block__list", "descriptionList__block__list--detail"]}
            )
            items.append(
                ItemDesc(
                    id=id, 
                    url=url, 
                    brand=self._brand, 
                    page_no=self._page_no, 
                    description=str(desc)
                )
            )
        return items
    
    def get_backup_file_name(self): 
        """Return the proper backup file name to store the ItemDesc objects."""
        return f"{BRANDS_DIR}{self._brand}/items/items_desc_p{self._page_no}.json"
           
def save_items_desc_for_all_brands(
    brands: List[str], 
    page_no: int
): 
    """Build a DescriptionScraper for all brands for a given page number."""
    driver = webdriver.Chrome()
    driver.maximize_window()
    for brand in brands: 
        print(f"Collecting items' description for {brand}...")
        scraper = DescriptionScraper(
            driver, 
            brand, 
            page_no
        )
        items_desc = scraper.get_items_desc()
        file_name = scraper.get_backup_file_name()
        save_json(
            data=items_desc, 
            file_name=file_name
        )
        print(f"Collected data saved at {file_name}.")
        print("*"*100)
    driver.quit()


    