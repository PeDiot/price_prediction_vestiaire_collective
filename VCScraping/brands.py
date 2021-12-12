"""Description.

Sub-library to collect the catalog of items each brand's first ten pages.

Example: 

>>> import VcScraping as vcs
>>> from selenium import webdriver
>>> from typing import List
>>> brands_links = vcs.load_json(
...     file_name="./backup/brands_links.json", 
...     data_type=List[vcs.BrandLink]
... )
>>> acne = brands_links[0] 
>>> acne
BrandLink(name='acne-studios', url='https://fr.vestiairecollective.com//acne-studios/')
>>> driver = webdriver.Chrome()

DevTools listening on ws://127.0.0.1:49675/devtools/browser/7a919794-fac6-42f4-a2b2-6b14d4c63514
>>> driver.maximize_window()
>>> scraper = vcs.BrandPageScraper(brand=acne, driver=driver)
>>> scraper.create_directory()
./backup/acne-studios has been created.
>>> from bs4 import BeautifulSoup as bs
>>> soup = bs(scraper.driver.page_source, "lxml")
>>> scraper._get_page_number(soup)
1
>>> scraper._get_catalog(soup)
'<ul _ngcontent-vc-app-c140="" class="catalog__gridContainer">...</ul>'
>>> scraper.to_brand_page()
BrandPage(name='acne-studios', no=1, catalog='<ul _ngcontent-vc-app-c140="" class="catalog__gridContainer">...</ul>')
"""

from .methods import accept_cookies, save_json
from .home_page import BrandLink

from bs4 import BeautifulSoup as bs

from selenium import webdriver
from selenium.webdriver.common.by import By

from serde import serialize, deserialize
from serde.json import from_json, to_json

from dataclasses import dataclass

from typing import List
import os

@serialize
@deserialize
@dataclass
class BrandPage: 
    name: str
    no: int
    catalog: str

class BrandPageScraper: 
    """Collect the source code representing the catalog of items for each brand."""

    def __init__(
        self, 
        brand: BrandLink, 
        driver: webdriver.chrome.webdriver.WebDriver
    ):
        self.brand = brand
        self.driver = driver
        self._get_page()

    def _get_page(self): 
        """Get the web page related to the brand."""
        self.driver.delete_all_cookies()
        self.driver.get(self.brand.url)
        accept_cookies(self.driver)

    def create_directory(self): 
        """Create a folder for each brand in the 'backup' directory."""
        parent_dir = "./backup/brands"
        path = parent_dir + "/" + self.brand.name
        if os.path.exists(path): 
            print(f"{path} already exists.")
        else:
            os.mkdir(path)
            print(f"{path} has been created.")

    def _get_page_number(self, soup: bs) -> int: 
        """Return page number."""
        button = soup.find(
            name="button", 
            attrs={"class": "catalogPagination__pageLink--active"}
        )
        return int(button.text.strip()) 

    def _get_catalog(self, soup: bs) -> str: 
        """Return the catalog of items."""
        catalog = soup.find(
            name="ul", 
            attrs={'class': 'catalog__gridContainer'}
        )
        return str(catalog)

    def to_brand_page(self) -> BrandPage: 
        """Return the brand's name, page number and catalog of items."""
        soup = bs(self.driver.page_source, "lxml")
        active_page_no = self._get_page_number(soup)
        catalog = self._get_catalog(soup)
        return BrandPage(
            name=self.brand.name, 
            no=active_page_no, 
            catalog=catalog
        )
    
    def click_next_page(self): 
        """Click on the arrow to go to the next page."""
        next_page = self.driver.find_elements(
            By.CLASS_NAME, 
            "catalogPagination__prevNextButton"
        )[1]
        next_page.click()

def save_all_brands_pages(brands: List[BrandLink], n_pages=10): 
    """Get and save a BrandPage object for the first 10 pages for each collected brand."""
    driver = webdriver.Chrome() 
    driver.maximize_window()
    for brand in brands: 
        print(f"Data collection for {brand.name}.")
        scraper = BrandPageScraper(brand=brand, driver=driver)
        scraper.create_directory()
        no = 0
        while no <= n_pages-1:
            brand_page = scraper.to_brand_page()
            path = f"./backup/brands/{brand.name}/"
            save_json(
                data=brand_page, 
                file_name=f"{path}p{no}.json"
            )
            scraper.click_next_page()
            no += 1
        print(f"Data has been added to {path}.")
        print("*"*50)
    driver.quit()