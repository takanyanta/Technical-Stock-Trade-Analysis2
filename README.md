# Technical-Stock-Trade-Analysis2
Using Logistic Regression, Gradient Boost Classification, GRU Classification

## Purpose
* To predict that the stock price will whether go up or down.
* Compare some machine learning techniques about classification.

## Process

### 1. Prepair Data

* Choose randomly 200 tickers

#### 1-1. Extract stock price data by using pandas-datareader

#### 1-2. Preprocess both the input and the target data

| Label |Explanation| Definition |
---|---|---
| 0 |Up|   <img src="https://latex.codecogs.com/gif.latex?\frac{p_{t&plus;30}}{p_t}&space;\geq&space;5%" /> |
| 1 |Stay| <img src="https://latex.codecogs.com/gif.latex?-5%&space;\leq&space;\frac{p_{t&plus;30}}{p_t}&space;\leq&space;5%" /> |
| 2 |Down| <img src="\frac{p_{t+30}}{p_t} \leq -5%" /> |

### 2. Train and Test

#### 2-1. Logistic Regression

#### 2-2. Gradient Boost Classification

#### 2-3. GRU Classification

### 3. 
