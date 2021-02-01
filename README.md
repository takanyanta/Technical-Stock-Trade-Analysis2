# Technical-Stock-Trade-Analysis2
Using Logistic Regression, Gradient Boost Classification, GRU Classification

## Purpose
* To predict that the stock price will whether go up or down.
* Compare some machine learning techniques about classification.

## Process

### 1. Prepair Data

* Choose randomly 50 tickers

#### 1-1. Extract stock price data by using pandas-datareader

#### 1-2. Preprocess both the input and the target data

* Define price at time point *t* as  <img src="https://latex.codecogs.com/gif.latex?p_{t}" />

**Input Data**

* For prediction, use time series data(<img src="https://latex.codecogs.com/gif.latex?p_{t-60}" /> ~ <img src="https://latex.codecogs.com/gif.latex?p_{t-1}" />)
* Each data are transformed to [0, 1] by MinMaxScaler

**Target Data**

| Label |Explanation| Definition |
---|---|---
| 0 |Up|   <img src="https://latex.codecogs.com/gif.latex?\frac{p_{t&plus;30}}{p_t}&space;\geq&space;5%" /> |
| 1 |Stay| <img src="https://latex.codecogs.com/gif.latex?-5%&space;\leq&space;\frac{p_{t&plus;30}}{p_t}&space;\leq&space;5%" /> |
| 2 |Down| <img src="https://latex.codecogs.com/gif.latex?\frac{p_{t&plus;30}}{p_t}&space;\leq&space;-5%" /> |

### 2. Training and Testing

* Use hold-out method simply(test_size=20%)
* Data distribution is as below;

** Train Data **

|Label|Explanation|Count|
---|---|---
|0|up|39,577|
|1|stay|44,379|
|2|down|37,895|

** Test Data **

|Label|Explanation|Count|
---|---|---
|0|up|13,036|
|1|stay|14,650|
|2|down|12,931|

#### 2-1. Logistic Regression

#### 2-2. Gradient Boost Classification

#### 2-3. GRU Classification

### Verification in actual data

* At 2021/2/1, the prediction for each tikers is as below;

|Ticker|Company Name|Prediction(2020/2/1)|Verification(2021/3/17)|
---|---|---|---
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
|||||
