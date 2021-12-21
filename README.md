# Decision Support for Pricing of Second-Hand Items


## Data source


[Vestiaire Collective](https://fr.vestiairecollective.com/) is a platform on which second-hand fashion items are traded. Its objective is to connect sellers and buyers while certifying the authenticity of the products offered by the sellers. This platform allows to remove the problem of asymetric information related to the sale of luxury products.


## Goal 

This project aims to help sellers set the right price when they want to put an item on sale on Vestiaire Collective. 

Stages of the sales process:

1. The seller fills in the characteristics of the product.
2. The seller sets a price for his product.
3. The *machine learning* algorithm predicts the price based on the product characteristics. This prediction serves as a *benchmark*.
4. The *benchmark* price is sent back to the user to help him choose the right price. 
5. Extension: 
    - Ads similar to the product the seller wants to sell are also returned as a comparison.
    - Prediction of the number of likes received by the article. 
    - Prediction of whether the article will be a favorite of the platform. 


## Project challenges


### For the seller 

The algorithm's prediction allows him to avoid wasting time in setting the price. This decision support maximizes his profits :
- by preventing him from selling at too low a price, 
- by preventing him from not finding buyers because his price is too high.


### For potential buyers

The algorithm's prediction of the market price is also an approximation of the *willingness to pay of buyers for a given type of item. By helping sellers set the "right" price, the algorithm allows buyers to find items sold at a price close to their willingness to pay.

### For Vestiaire Collective 

**Profit maximization**: help in setting prices ensures the volume and speed of transactions. The more transactions there are on the platform, the more commissions Vestiaire Collective receives.

**User satisfaction**: Buyers find items that are priced to generate a surplus and sellers save time selling their items.


## Collected data  


| Variable   |      Type      | Description    |
|:---------- |:------------- |:------------- |
| `id`         |  int        | item id               | 
| `url`        |   string    | link of the item's ad             | 
| `num_likes`   | int  | number of *likes* received by the article               | 
| `price` | float | item's price |
| `we_love_tag` | bool | indicates if the item is a favorite of Vestiaire Collective |
| `online_date` | datetime | date of online publication | 
| `gender` | bool | item's type | 
| `category` | category | item's category (shoes, clothing, bags, etc.) | 
|  `sub_category` | category | item's sub-category (shirt, jeans, trainers, etc.) | 
| `designer` | category | item's brand | 
| `condition` | category | item's condition |
| `material` | category | materials used to make the item | 
| `color` | category | color of the item | 
| `size` | category | item's size (M, L, 40, etc.) | 
| `location` | category ou string | seller's location | 


## Implementation 

### Predict the price

Explanatory variables: product characteristics excluding `id`, `url`, `online_date`

Target: `price`

Optimization of several regression models to determine the best-fitting relationship between the target and the explanatory variables

Choice of the best model, training on the data set and saving

### Predict `we_love_tag` and `num_likes`

When a new article is put online, we have no information neither on the `we_love_tag` nor the `num_likes` variables: 

- Classification model would consist in predicting whether the article will be a favorite of Vestiaire Collective.
- Regression model to forecast the number of likes that can be received by a new item. 

Targets: `num_likes` and `we_love_tag` 

## User Interface

GUI (Graphical User Interface) or CLI (Command Line Interface)



