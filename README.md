<h1>Decision Support for Pricing of Second-Hand Luxury Fashion Items <img src="./imgs/vestiaire_logo.jpg" width=40 height=30> </h1>

## Introduction

### Vestiaire Collective 

Vestiaire Collective is a platform on which second-hand fashion items are traded. Its objective is to connect sellers and buyers while certifying the authenticity of the products offered by the sellers. This platform allows to remove the problem of asymetric information related to the sale of luxury products. 

### Purpose

Implementing a machine learning algorithm to **estimate the price of a second-hand luxury item** in order to help users set the "right" price when they put an item on sale on the platform.

### Motivations

The decision-support tool could minimize the time spent by sellers to set prices by providing them with an estimated price for their item. Besides, it could help sellers maximize their profits by minimizing the risks of selling at too low a price or not finding buyers because their a too high price.

The tool we want to build could help buyers to find items sold at a price close to their willingness to pay by encouraging sellers to set the "right" price.

For Vestiaire Collective, the price prediction tool could ensure the volume and speed of transactions. The more transactions, the more commissions the platform earns. It could also enhance user satisfaction: buyers would find items that are priced to generate a surplus and sellers would save time selling their items. 

### Method 

1. Data collection from Vestiaire Collective [website](https://fr.vestiairecollective.com/)

2. Data cleaning and feature engineering to build a reliable data set

3. Model training and parameter tuning

4. Model selection and decision-making

## Data collection 

### Web scraping

Selection of the **35 first brands** from the [Notre Sélection](https://fr.vestiairecollective.com/marques/) section on the website

Implementation of a python library `vc_scraping` to collect and parse web data from [Vestiaire Collective](https://fr.vestiairecollective.com/):
- Saving the **10 first pages** of articles for each brand and the web pages of each article in serializable dataclasses
- Identification and storage of each item's characteristics (price, brand, number of *likes*, etc.) in serializable dataclasses

Example with `python -m vc_scraping`

![width:1150px height:640px](imgs/vc_scraping.gif)

**Code summary**

```
C:\Users\pemma\OneDrive - Université de Tours\Mécen\M2\S1\02 - Machine Learning\05 - Projet\price_prediction_vestiaire_collective>pygount --format=summary ./vc_scraping
Language   Files    %     Code    %     Comment    %   
---------  -----  ------  ----  ------  -------  ------
Python         9  100.00   557  100.00      211  100.00
---------  -----  ------  ----  ------  -------  ------
Sum total      9           557              211
```

### Data description

Data set obtained after the scraping and parsing stages via the `make_dataset` function of the `vc_scraping` library

| Variable   |      Type      | Description    |
|:-|:-|:-|
| `id`         |  int        | Item's ID number               | 
| `url`        |   string    | Item web page link        | 
| `num_likes`   | int  | Number of likes received
| `price` | float | Price |
| `we_love_tag` | bool | Indicate whether the item is a favorite of Vestiaire Collective |
| `online_date` | datetime | Date of online publication|
| `gender` | bool | Men, Women, Kids | 
| `category` | category | Shoes, Clothing, Bags, etc. | 
|  `sub_category` | category | Jeans, Trainers, Coats, etc. | 
| `designer` | category | Item's brand | 
| `condition` | category | Item's condition (fair, good, nerver worn, etc.) |
| `material` | category | Materials  | 
| `color` | category | Color | 
| `size` | category | Size (M, L, 40, etc.) | 
| `location` | category | Seller's location | 


## Data analysis and cleaning

### Target variable visualization

![center](imgs/price_distribution.png)

One can note a right skewed density curve related to the `price` target variable. The average price ($\approx$ 421€) is indeed higher than the median price (250€).

### Data preprocessing

- Removal of less represented categories in each categorical variable
- Treatment of the `size` variable: 

  - Sizes are encoded as "S, M, L" for clothing
  - Sizes are encoded from 35 to 45 for shoes

- Encoding of categorical variables into dummies with `pd.get_dummies`

Nb: the data cleaning process can be found in the `notebooks` folder, it has not been automated.

## Regression models with `sklearn`

### General information

**Purpose** : identify a model that best explains the relationship between the price of an item and its characteristics

**Method** : implementation of a machine learning library `vc_ml` to almost automate models' training and select the best performing model

### `vc_ml`

It is a library which consists in the following scripts: 

- `data.py` : create training and testing sets
- `estimators.py` : define estimators as serializable dataclasses
- `config.py` : configuration of the models and parameter grids to train
- `training.py` 
  - Model training for a given combination of parameters and cross-validation
  - Function to train multiple models
- `selection.py` : functions to select the best performing estimator and its parameters

**Code summary**

```
C:\Users\pemma\OneDrive - Université de Tours\Mécen\M2\S1\02 - Machine Learning\05 - Projet\price_prediction_vestiaire_collective>pygount --format=summary ./vc_ml
Language   Files    %     Code    %     Comment    %   
---------  -----  ------  ----  ------  -------  ------
Python         7  100.00   708  100.00      398  100.00
---------  -----  ------  ----  ------  -------  ------
Sum total      7           708              398
```

**Tests**: the library is partly tested, see `tests_vc_ml`.

### Automated training

Each estimator is a serializable dataclass whose arguments are the parameters associated to the `sklearn` estimator.

The estimators are stored in a configuration file in `yaml` format via the `Config` dataclass.

Minimal example for the configuration of a `GradientBoostingRegressor` estimator:

```
In [1]: from vc_ml import load_config
In [2]: config = load_config(file_name="config_gb.yaml")
In [3]: config
Out[3]: Config(lr=None, ridge=None, tree=None, rf=None, gb=GBEstimator(n_estimators=[250, 500, 750, 1000], ...), mlp=None)
```

`ModelTraining` class's tools: 

- Create a pipeline with possibility to reduce the feature vector's dimensionality with PCA before fitting the model
- Train an estimator with given parameters using cross-validation
- Save fitted model using a unique and identifiable name

The `train_models` function: 

- Train multiple estimators with a parameter grid for each of them
- Indicate a list of values for the `n_components` argument of `PCA`

Example in command line via `python -m vc_ml` 

![width:1150px height:640px](imgs/vc_ml.gif)

## Model selection

### Method

- Retrieve models and cross-validation scores stored in a backup folder using the `get_cv_results`
- Identify the best performing model with 3 possible criteria using `get_best_estimator`: 

  - Cross-validation average train score $\text{R}^2_{\text{tr}}$ 
  - Cross-validation average test score $\text{R}^2_{\text{te}}$ 
  - Cross-validation average score $\text{R}^2_{\text{avg}} = \frac{\text{R}^2_{\text{tr}}}{\text{R}^2_{\text{te}}}$

$\rightarrow$ `GradientBoostingRegressor` without PCA

### `GradientBoostingRegressor` parameters

Gradient Boosting is an ensemble method which combines several weak predictors like regression trees into a strong predictor.The idea of most boosting methods is to train predictors sequentially, each trying to correct its predecessor.

- `n_estimators` : 250
- `max_depth` : 10
- `min_samples_split` : 20
- `min_samples_leaf` : 5
- `learning_rate` : 0.1
- `loss` : "huber"
- `criterion` : "squared_error"

### `GradientBoostingRegressor` visualization

![center](imgs/prediction_plot.png)

Even though train and test scores are not that high, they are quite close which indicate that the selected model manages to scale on unseen data. However, it fails in predicting new prices accurately. 

### Is it worth the cost to use machine learning?

- Significant increase in score between the `GradientBoostingRegressor` (43%) estimator and a baseline model such as the `DummyRegressor` estimator (0%) (prediction by the mean) 
- However, **only 43% of the price variability is explained by the explanatory variables used in the model**
- The model has difficulty in predicting expensive items

## Concluding remarks

### Coding 

- Semi automation data collection method to retrieve information on luxury fashion items from Vestiaire Collective
- Methods to train regression models and model selection in an almost automated fashion

### Business value

- Model selection that can help Vestiaire Collective users in pricing their items
- However, the selected model is limited in terms of predictive power and its results need to be qualified

### Limits & further improvements

Through this project, it can be concluded that predicting the price of second-hand luxury fashion items is a complicated task. The data can be subject to both bias and noise as this kind of products have a psychological value which varies accross individuals. To adress this issue, several strategies could be adopted such as:
- Collecting much more data to train models on a greater number of samples
- Implementing more advanced deep learning models to deal with noise and bias in the data

Furthermore, if a better perfoming predictor can be identified it could be a good idea to build a dashboard which would take the item's features as inputs and render the predicted price. It could be a useful tool for Vestiaire Collective users.

Finally, the tests related to the `vc_ml` and `vc_scraping` library could be enriched. 