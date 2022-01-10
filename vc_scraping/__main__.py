"""Description.

Usage example of the vc_scraping library. 
"""

from .methods import (
    BACKUP_DIR, 
    BRANDS_DIR, 
    flatten_list, 
    load_json, 
    save_json, 
    accept_cookies
)

from .home_page import (
    HomePage, 
    BrandLink, 
    HomePageScraper, 
    get_brands
)

from .brands import (
    BrandPage, 
    BrandPageScraper, 
    save_all_brands_pages
)

from .brand_page import (
    find_paths, 
    get_brands_files_paths, 
    create_items_dir, 
    BasicItem, 
    PageParser, 
    save_all_basic_items
)

from .item_page import (
    ItemDesc, 
    DescriptionScraper, 
    save_items_desc_for_all_brands, 
)

from .item import (
    ItemAttrs, 
    DescriptionParser, 
)

from .dataset import make_dataset

from selenium import webdriver
from bs4 import BeautifulSoup as bs

from typing import List

from rich import print

print(" "*100)
print("Usage example of the 'vs_scraping' library which aims to collect data from the Vestiaire Collective website.")
print(" "*100)
print("-"*100)

print("Collecting brands from Vestiaire Collective website...")
scraper = HomePageScraper()
brands_links = scraper.get_brands()
print(brands_links)
scraper.driver.quit()
print(f"{len(brands_links)} brands collected.")

print("-"*100)

brand = "gucci"
page_no = 2

print(" "*100)

print(f"Parsing page n°{page_no} for {brand}")

print("Parsing items' prices, number of likes and links...")
print(" "*100)
brands_files_paths = get_brands_files_paths()
page_dir = brands_files_paths[brand][page_no]
parser = PageParser(page_dir)
print(parser)
print(" "*100)
basic_items = parser.to_basic_items()
print(f"{len(basic_items)} items have been parsed.")
print(f"First item's basic features : {basic_items[0]}")

print("-"*100)

print("Parsing items' specific features...")
print(" "*100)
parser = DescriptionParser(brand, page_no)
print(parser)
print(" "*100)
items = parser.to_ItemAttrs()
print(f"{len(items)} items have been parsed.")
print(f"First item's features: {items[0]}")
parser.save(items)