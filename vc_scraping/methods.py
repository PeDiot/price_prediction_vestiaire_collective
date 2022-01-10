"""Description.

Sub-library with useful functions for Vestiaire collective scraping and data storage.
"""

from typing import List
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from serde.json import from_json, to_json

import os

BACKUP_DIR = "./backup/scraping/"
BRANDS_DIR = BACKUP_DIR + "brands/"

def flatten_list(big_list: List[List]) -> List: 
    """Flatten a list of lists."""
    return [ sublist for list_ in big_list for sublist in list_ ]

def load_json(file_name: str, data_type): 
    """Load a json file in the correct format.
    
    Example: 
    >>> path = "./backup/brands_links.json"
    >>> vcs.load_json(file_name=path, data_type=List[vcs.BrandLink])
    [BrandLink(name='acne-studios', url='https://fr.vestiairecollective.com//acne-studios/'),..., BrandLink(name='yves-saint-laurent', url='https://fr.vestiairecollective.com//yves-saint-laurent/')]
    """
    with open(file_name, "r") as file: 
        data = file.read()
    return from_json(data_type, data)

def save_json(data, file_name: str): 
    """Save a python object to json."""
    with open(file_name, "w") as file: 
        file.write(to_json(data))

def accept_cookies(driver: webdriver.chrome.webdriver.WebDriver): 
    """Accept cookies on Vestiaire Collective pages."""
    button = WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((
            By.ID,
            "popin_tc_privacy_button_2"
        ))
    )
    button.click()
