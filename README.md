# Technical-Stock-Trade-Analysis2
Using Logistic Regression, Gradient Boost Classification, GRU Classification

## 1. Purpose
* To predict that the stock price will whether go up or down.
* Compare some machine learning techniques about classification.

## 2. Process

* Choose randomly 50 tickers

### 2-1. Extract stock price data by using pandas-datareader

```python
def return_series_data(ticker):
    label = 0
    cols = ["x{:02d}".format(-i) for i in np.arange(-60, 0) ]  + ["target"]
    try:
        df = DataReader("{}.T".format(ticker), "yahoo", datetime(2000, 1, 1))
    except RemoteDataError:
        return ticker
    except KeyError:
        return ticker

    try:
        temp_df = df[df["Volume"] != 0]

        X, y = [], []
        for i in range(60, len(temp_df)-30, 1):

            gain = temp_df["Adj Close"].iloc[i+30]/temp_df["Adj Close"].iloc[i]-1

            if gain >= 0.35:
                label=1
            elif 0.25  <= gain and gain < 0.35:
                label=2
            elif 0.15  <= gain and gain < 0.25:
                label=3
            elif 0.05  <= gain and gain < 0.15:
                label=4
            elif -0.05  <= gain and gain < 0.05:
                label=5
            elif -0.15  <= gain and gain < -0.05:
                label=6
            elif -0.25  <= gain and gain < -0.15:
                label=7
            elif -0.35  <= gain and gain < -0.25:
                label=8
            elif gain < -0.35 :
                label=9
            else:
                label=np.nan
            
            X.append(temp_df.iloc[i-60:i]["Adj Close"].values.tolist())
            y.append(label)

        df_ = pd.DataFrame(np.hstack([np.array(X), np.array(y).reshape(-1, 1)]), columns=cols)
        df_.to_csv(r"D:\ticker\{}.csv".format(ticker), index=None)
        return ticker
        
    except TypeError:
        return ticker
    except ValueError:
        return ticker
```

### 2-2. Preprocess both the input and the target data

* Define price at time point *t* as  <img src="https://latex.codecogs.com/gif.latex?p_{t}" />

**Input Data**

* For prediction, use time series data(<img src="https://latex.codecogs.com/gif.latex?p_{t-60}" /> ~ <img src="https://latex.codecogs.com/gif.latex?p_{t-1}" />)
* Each data are transformed to [0, 1] by MinMaxScaler

![Extract the frame](https://github.com/takanyanta/Technical-Stock-Trade-Analysis2/blob/main/Pic/plots.png "process1")

**Target Data**

| Label |Explanation| Definition |
---|---|---
| 0 |Up|   <img src="https://latex.codecogs.com/gif.latex?\frac{p_{t&plus;30}}{p_t}&space;\geq&space;5%" /> |
| 1 |Stay| <img src="https://latex.codecogs.com/gif.latex?-5%&space;\leq&space;\frac{p_{t&plus;30}}{p_t}&space;\leq&space;5%" /> |
| 2 |Down| <img src="https://latex.codecogs.com/gif.latex?\frac{p_{t&plus;30}}{p_t}&space;\leq&space;-5%" /> |

```python
file_list = glob(r"D:\ticker\*csv")

df = pd.DataFrame()
random.seed(50)
file_list_random = random.sample(file_list, 50)
k = 0
for i in tqdm( file_list_random ):
    temp_df = pd.read_csv(i)
    mn = MinMaxScaler()
    X_std = mn.fit_transform(temp_df.iloc[:, :-1])
    y = temp_df.iloc[:, -1].values

    temp = np.hstack([X_std, y.reshape(-1, 1)])
    temp_df = pd.DataFrame(temp, columns=temp_df.columns.values)
    df = pd.concat([df, temp_df])
    k += 1
  #if k > 200:
  #  break
```

### 2-3. Training and Testing

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

### 2-4. Logistic Regression

```python
LogReg = LogisticRegression(max_iter=1000, verbose=1, n_jobs=-1)
LogReg.fit(X_train1, y_train1)
print(LogReg.score(X_train1, y_train1))
print(LogReg.score(X_test1, y_test1))
res_LogReg = pd.DataFrame(confusion_matrix(y_test1, LogReg.predict(X_test1)), 
             columns=["Predict:0(up)", "Predict:1(stay)", "Predict:2(down)"], 
             index=["Actual:0(up)", "Actual:1(stay)", "Actual:2(down)"])
print( res_LogReg )
```

|Class|Accuracy-Score|
---|---
|**Train**|0.362714787815673|
|**Test**|0.35578824046276636|

|-|Predict:0(up) | Predict:1(stay) | Predict:2(down) |
---|---|---|---
|**Actual:0(up)**        |     5044                    |      7651                                  |       1114|
|**Actual:1(stay)**      |     5072                    |     8259                                  |       1135|
|**Actual:2(down)**       |    3979                                 |      7443                                 |      1274|

### 2-5. Gradient Boost Classification

* Only change *max_depth* parameter(set to 20)

```python
XGBC = XGBClassifier(max_depth = 20,  tree_method='gpu_hist')
XGBC.fit(X_train1, y_train1)
print(XGBC.score(X_train1, y_train1))
print(XGBC.score(X_test1, y_test1))
res_XGBC = pd.DataFrame(confusion_matrix(y_test1, XGBC.predict(X_test1)), 
             columns=["Predict:0(up)", "Predict:1(stay)", "Predict:2(down)"], 
             index=["Actual:0(up)", "Actual:1(stay)", "Actual:2(down)"])
print( res_XGBC )
```

|Class|Accuracy-Score|
---|---
|**Train**|1.0|
|**Test**|0.48700300212345315|

|-|Predict:0(up) | Predict:1(stay) | Predict:2(down) |
---|---|---|---
|**Actual:0(up)**        |  6490               |     4634                  |  2685|
|**Actual:1(stay)**      |   3318                 |   8215               |  2933|
|**Actual:2(down)**       |   2792                  |   4656                | 5248|

![Extract the frame](https://github.com/takanyanta/Technical-Stock-Trade-Analysis2/blob/main/Pic/GBC.png "process1")

### 2-6. GRU(Gated Recurrent Unit) Classification
* Define *batch_size* as 128

**Model Structure**

![Extract the frame](https://github.com/takanyanta/Technical-Stock-Trade-Analysis2/blob/main/Pic/GRU_structure.png "process1")

**Learning History**

![Extract the frame](https://github.com/takanyanta/Technical-Stock-Trade-Analysis2/blob/main/Pic/history.png "process1")

```python
length = X_train_keras.shape[1]
num_feature = X_train_keras.shape[2]

model = Sequential()

#model.add(GRU(128, input_shape=(length, num_feature), return_sequences=False))

model.add(GRU(128, input_shape=(length, num_feature), return_sequences=True))
model.add(GRU(128, input_shape=(length, num_feature), return_sequences=False))
model.add(Dense(3, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="Adam")
print(model.summary())

with tf.device('/GPU:0'):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
    history = model.fit(X_train_keras, y_train2, 
              batch_size=128, 
              epochs=200,
              validation_split=0.1,
              callbacks = [es],
              shuffle=False)
```

|Class|Accuracy-Score|
---|---
|Train|0.7484541785993231|
|Test|0.6737204364062386|

|-|Predict:0(up) | Predict:1(stay) | Predict:2(down) |
---|---|---|---
|**Actual:0(up)**         |    9626                      |    3005                   |       1178|
|**Actual:1(stay)**      |     2627                   |       8957                   |       2882|
|**Actual:2(down)**        |   883        |     2793                   |       9020|


### 3. Verification in actual data

* At 2021/2/1, the prediction for each tikers by GRU is as below;

|ticker|Company Name|Prediction Label|Explanation|Price at Prediction(2020/2/1)|Price at Verification(2021/3/17)|Result|
---|---|---|---|---|---|---
|3443|TOCALO Co., Ltd.|1|Stay|4145|4950|Up|
|3466|LaSalle LOGIPORT REIT|2|Down|164000|160300|Stay|
|3577|Tokai Senko K.K.|1|Stay|1119|1140|Stay|
|3578|Soko Seiren Co.,Ltd.|0|Up|361|472|Up|
|3624|Axel Mark Inc. |0|Up|331|334|Stay|
|3674|Aucfan Co., Ltd. |2|Down|2205|1858|Down|
|3738|T-Gaia Corporation|2|Down|1907|1983|Stay|
|3836|Avant Corporation|1|Stay|1416|1581|Up|
|3865|Hokuetsu Corporation|2|Down|437|546|Up|
|3901|MarkLines Co., Ltd. |0|Up|2484|2447|Stay|
|3907|Silicon Studio Corporation|1|Stay|1227|1261|Stay|
|4080|Tanaka Chemical Corporation|2|Down|1241|1278|Stay|
|4428|sinops Inc. |2|Down|1619|1879|Up|
|4499|Speee, Inc.|0|Up|2900|2726|Down|
|4555|Sawai Pharmaceutical Co., Ltd. |2|Down|4730|5410|Up|
|4635|Tokyo Printing Ink Mfg. Co., Ltd. |1|Stay|2060|2196|Up|
|4653|Daiohs Corporation|1|Stay|961|1000|Stay|
|4669|NIPPAN RENTAL Co.,Ltd. |1|Stay|742|1071|Up|
|4777|Gala Incorporated|0|Up|219|222|Stay|
|5108| Bridgestone Co.|0|Up|3906|4347|Up|
|5194|Sagami Rubber Industries Co., Ltd.|0|Up|1190|1177|Stay|
|5208|Arisawa Mfg. Co., Ltd.|1|Stay|971|1076|Up|
|5987|Onex Co.|1|Stay|1351|1231|Down|
|5999|Ihara Science Co.|2|Down|1750|1709|Stay|
|6067|Impact HD Inc.|2|Down|2860|2515|Down|
|6072|Jibannet Holdings Co., Ltd.|1|Stay|203|||
|6094| Freakout Holdings, Inc. |1|Stay|850|||
|6134|Fuji Co.|2|Down|2797|||
|6188|Fuji Soft Service Bureau Inc.|2|Down|515|||
|6189|Global Kids Company Corp.|2|Down|885|||
|6303|Sasakura Engineering Co., Ltd.|1|Stay|2256|||
|6594| Nidec Co. |1|Stay|14195|||
|6722|A&T Co. |1|Stay|1807|||
|6736|Suncorporation|0|Up|3935|||
|6744|Nohmi Bosai Ltd.|0|Up|2218|||
|6775|TB Group Inc.|1|Stay|171|||
|6839|Funai Electric Co., Ltd.|1|Stay|430|||
|6958|CMK Co. |2|Down|436|||
|7057|New Constructor's Network Co., Ltd. |1|Stay|1222|||
|7245|Daido Metal Co., Ltd.|2|Down|508|||
|7255|Sakurai Ltd.|0|Up|492|||
|7268|Tatsumi Co.|2|Down|341|||
|7578|Nichiryoku Co., Ltd.|2|Down|1211|||
|7597|Tokyo Kiho Co., Ltd.|1|Stay|1899|||
|7618| PC Depot Co. |2|Down|562|||
|7676|Goodspeed. Co., Ltd.|1|Stay|1650|||
|7769|Rhythm Co., Ltd.|2|Down|728|||
|7863|Hiraga Co., Ltd.|2|Down|850|||


### 4. Conclusion
* 
* 
* 
