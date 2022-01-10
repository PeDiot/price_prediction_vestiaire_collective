"""Description.

Library to scrap the Vestiaire Collective website. 
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
