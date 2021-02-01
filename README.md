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

**Train Data**

|Label|Explanation|Count|Composition ratio|
---|---|---|---
|0|up|39,577|36.4%|
|1|stay|44,379|32.4%|
|2|down|37,895|31.0%|

**Test Data**

|Label|Explanation|Count|Composition ratio|
---|---|---|---
|0|up|13,036|36.0%|
|1|stay|14,650|32.0%|
|2|down|12,931|31.8%|

#### 2-1. Logistic Regression

|Class|Accuracy-Score|
---|---
|**Train**|0.3902635185595522|
|**Test**|0.3884826550459167|

|-|Predict:0(up) | Predict:1(stay) | Predict:2(down) |
---|---|---|---
|**Actual:0(up)**        |     3632       |      8680       |       724|
|**Actual:1(stay)**      |     2927       |     10910       |       813|
|**Actual:2(down)**       |    2811       |      8883       |      1237|

#### 2-2. Gradient Boost Classification

* Only change *max_depth* parameter

|Class|Accuracy-Score|
---|---
|**Train**|0.3902635185595522|
|**Test**|0.3884826550459167|

|-|Predict:0(up) | Predict:1(stay) | Predict:2(down) |
---|---|---|---
|**Actual:0(up)**        |     3632       |      8680       |       724|
|**Actual:1(stay)**      |     2927       |     10910       |       813|
|**Actual:2(down)**       |    2811       |      8883       |      1237|

#### 2-3. GRU Classification
* Define *batch_size* as 128

**Model Structure**

|Class|Accuracy-Score|
---|---
|Train|0.689112112333916|
|Test|0.6414555481694857|

|-|Predict:0(up) | Predict:1(stay) | Predict:2(down) |
---|---|---|---
|**Actual:0(up)**         |    9097         |    2635      |       1304|
|**Actual:1(stay)**      |     3442      |       8630      |       2578|
|**Actual:2(down)**        |   1953        |     2651      |       8327|


### 3. Verification in actual data

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
