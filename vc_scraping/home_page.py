"""Description.

Sub-library to collect data from Vestiaire Collective home page.

>>> import vc_scraping as vcs
>>> scraper = vcs.HomePageScraper()

DevTools listening on ws://127.0.0.1:50148/devtools/browser/037aa521-2542-4fcf-8189-7766ee8fdf8b
>>> vcs.save_json(
...     data=scraper.home_page,
...     file_name="./backup/home_page.json"
... )
>>> brands_links = scraper.get_brands()
>>> vcs.save_json(
...     data=brands_links,  
...     file_name="./backup/brands_links.json"
... )
>>> scraper.driver.quit()
"""

from .methods import DIR, accept_cookies, load_json

from bs4 import BeautifulSoup as bs

from selenium import webdriver

from serde import serialize, deserialize
from serde.json import from_json, to_json

from dataclasses import dataclass
from typing import List

HOME_PAGE_URL = "https://fr.vestiairecollective.com/"

@serialize
@deserialize
@dataclass
class HomePage: 
    page_code: str
    url: str = HOME_PAGE_URL

@serialize
@deserialize
@dataclass
class BrandLink: 
    name: str
    url: str

class HomePageScraper: 
    """Vestiaire Collective's home page scraper."""

    def __init__(self): 
        """Launch the webdriver."""
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()
        self.home_page = self._get_code()

    def _get_code(self) -> HomePage: 
        """Get Vestiaire Collective's home page code."""
        self.driver.get(HOME_PAGE_URL)
        accept_cookies(self.driver)
        return HomePage(page_code=self.driver.page_source)

    def get_brands(self) -> List[BrandLink]: 
        """Get brands located in 'Editor's pick'."""
        soup = bs(self.home_page.page_code, "lxml")
        brands = soup.find(
            name="li", 
            attrs={"class": "mainNav__level3--brands"}
        ).find_all(
            name="a", 
            attrs={"class" : "link--block"}
        )
        return [
            BrandLink(
                name=brand.get("href")[1:-1], 
                url="https://fr.vestiairecollective.com/" + brand.get("href")
            ) for brand in brands
        ]

def get_brands(): 
    """Return the name of all collected brands."""
    brands_links = load_json(
        file_name=DIR+"/backup/brands_links.json", 
        data_type=List[BrandLink]
    )
    return [brand.name for brand in brands_links]