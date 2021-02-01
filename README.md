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
|0|up|41,565|33.8%|
|1|stay|43,494|35.3%|
|2|down|37,853|30.7%|

**Test Data**

|Label|Explanation|Count|Composition ratio|
---|---|---|---
|0|up|13809|33.8%|
|1|stay|14466|35.3%|
|2|down|12696|30.7%|

#### 2-1. Logistic Regression

|Class|Accuracy-Score|
---|---
|**Train**|0.3763831033585004|
|**Test**|0.3718971955773596|

|-|Predict:0(up) | Predict:1(stay) | Predict:2(down) |
---|---|---|---
|**Actual:0(up)**        |     6623       |      6293                     |       893|
|**Actual:1(stay)**      |     6306       |     7245                     |       915|
|**Actual:2(down)**       |    5408                    |      5919                    |      1369|

#### 2-2. Gradient Boost Classification

* Only change *max_depth* parameter(set to 19)

|Class|Accuracy-Score|
---|---
|**Train**|0.9902531892736267|
|**Test**|0.6847282224012106|

|-|Predict:0(up) | Predict:1(stay) | Predict:2(down) |
---|---|---|---
|**Actual:0(up)**        |     9758                    |      2878                    |       1173|
|**Actual:1(stay)**      |     2598                    |     9944                    |       1924|
|**Actual:2(down)**       |    1568                    |      2776                    |      8352|

#### 2-3. GRU Classification
* Define *batch_size* as 128

**Model Structure**

![Extract the frame](https://github.com/takanyanta/Technical-Stock-Trade-Analysis2/blob/main/Pic/GRU_structure.png "process1")

**Learning History**

![Extract the frame](https://github.com/takanyanta/Technical-Stock-Trade-Analysis2/blob/main/Pic/history.png "process1")

|Class|Accuracy-Score|
---|---
|Train|0.7076607654256704|
|Test|0.6496790412730956|

|-|Predict:0(up) | Predict:1(stay) | Predict:2(down) |
---|---|---|---
|**Actual:0(up)**         |    9097                      |    2635      |       1304|
|**Actual:1(stay)**      |     3442      |       8630      |       2578|
|**Actual:2(down)**        |   1953        |     2651      |       8327|


### 3. Verification in actual data

* At 2021/2/1, the prediction for each tikers by GRU is as below;

|Ticker|Company Name|Prediction label|Explanation|Price at Prediction(2020/2/1)|Price at Verification(2021/3/17)|Result|
---|---|---|---|---|---|---
|3443|TOCALO Co., Ltd.|0|Up|4145|||
|3466|LaSalle LOGIPORT REIT|2|Down|164000|||
|3577|Tokai Senko K.K.|0|Up|1119|||
|3578|Soko Seiren Co.,Ltd.|2|Down|361|||
|3624|Axel Mark Inc. |0|Up|331|||
|3674|Aucfan Co., Ltd. |2|Down|2205|||
|3738|T-Gaia Corporation|1|Stay|1907|||
|3836|Avant Corporation|2|Down|1416|||
|3865|Hokuetsu Corporation|2|Down|437|||
|3901|MarkLines Co., Ltd. |1|Stay|2484|||
|3907|Silicon Studio Corporation|0|Up|1227|||
|4080|Tanaka Chemical Corporation|0|Up|1241|||
|4428|sinops Inc. |2|Down|1619|||
|4499|Speee, Inc.|2|Down|2900|||
|4555|Sawai Pharmaceutical Co., Ltd. |2|Down|4730|||
|4635|Tokyo Printing Ink Mfg. Co., Ltd. |2|Down|2060|||
|4653|Daiohs Corporation|0|Up|961|||
|4669|NIPPAN RENTAL Co.,Ltd. |0|Up|742|||
|4777|Gala Incorporated|2|Down|219|||
|5108| Bridgestone Co.|2|Down|3906|||
|5194|Sagami Rubber Industries Co., Ltd.|0|Up|1190|||
|5208|Arisawa Mfg. Co., Ltd.|1|Stay|971|||
|5987|Onex Co.|2|Down|1351|||
|5999|Ihara Science Co.|0|Up|1750|||
|6067|Impact HD Inc.|2|Down|2860|||
|6072|Jibannet Holdings Co., Ltd.|1|Stay|203|||
|6094| Freakout Holdings, Inc. |0|Up|850|||
|6134|Fuji Co.|2|Down|2797|||
|6188|Fuji Soft Service Bureau Inc.|0|Up|515|||
|6189|Global Kids Company Corp.|2|Down|885|||
|6303|Sasakura Engineering Co., Ltd.|1|Stay|2256|||
|6594| Nidec Co. |2|Down|14195|||
|6722|A&T Co. |2|Down|1807|||
|6736|Suncorporation|2|Down|3935|||
|6744|Nohmi Bosai Ltd.|0|Up|2218|||
|6775|TB Group Inc.|0|Up|171|||
|6839|Funai Electric Co., Ltd.|2|Down|430|||
|6958|CMK Co. |0|Up|436|||
|7057|New Constructor's Network Co., Ltd. |2|Down|1222|||
|7245|Daido Metal Co., Ltd.|0|Up|508|||
|7255|Sakurai Ltd.|2|Down|492|||
|7268|Tatsumi Co.|0|Up|341|||
|7578|Nichiryoku Co., Ltd.|0|Up|1211|||
|7597|Tokyo Kiho Co., Ltd.|0|Up|1899|||
|7618| PC Depot Co. |0|Up|562|||
|7676|Goodspeed. Co., Ltd.|2|Down|1650|||
|7769|Rhythm Co., Ltd.|2|Down|728|||
|7863|Hiraga Co., Ltd.|2|Down|850|||
