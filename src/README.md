


# Perlib

This repository hosts the development of the Perlib library.

## About Perlib

Perlib is a framework written in Python where you can use deep and machine learning algorithms.

## Perlib is
Feature to use many deep or machine learning models easily
Feature to easily generate estimates in a single line with default parameters
Understanding data with simple analyzes with a single line
Feature to automatically preprocess data in a single line
Feature to easily create artificial neural networks
Feature to manually pre-process data, extract analysis or create models with detailed parameters, produce tests and predictions

## Usage
The core data structures are layers and models. For quick results with default parameters
To set up more detailed operations and structures, you should use the Perflib functional API, which allows you to create arbitrary layers or write models completely from scratch via subclassing.


## Install
```python
pip install perlib
```

```python
from perlib.forecaster import *
```

This is how you can use sample datasets.

```python
from perlib import datasets # or  from perlib.datasets import *
import pandas as pd
dataset = datasets.load_airpassengers()
data = pd.DataFrame(dataset)
data.index = pd.date_range(start="2022-01-01",periods=len(data),freq="d")
```

To read your own dataset;
```python 
import perlib
pr = perlib.dataPrepration()
data = pr.read_data("./datasets/winequality-white.csv",delimiter=";")
```

The easiest way to get quick results is with the 'get_result' function.
You can choice modelname ;
"rnn", "lstm", "bilstm", "convlstm", "tcn", "lstnet", "arima" ,"sarima" or all machine learning algorithms


```python 
forecast,evaluate = get_result(dataFrame=data,
                    y="Values",
                    modelName="Lstnet",
                    dateColumn=False,
                    process=False,
                    forecastNumber=24,
                    metric=["mape","mae","mse"],
                    epoch=50,
                    forecastingStartDate=False,
                    verbose=1
                    )
```
```python 

Parameters created
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 24, 1)]      0           []                               
                                                                                                  
 reshape (Reshape)              (None, 24, 1, 1)     0           ['input_1[0][0]']                
                                                                                                  
 conv2d (Conv2D)                (None, 19, 1, 100)   700         ['reshape[0][0]']   

 dropout (Dropout)              (None, 19, 1, 100)   0           ['conv2d[0][0]']                 
                                                                                                  
 reshape_1 (Reshape)            (None, 19, 100)      0           ['dropout[0][0]']                
                                                                                                  
 pre_skip_trans (PreSkipTrans)  (None, 1, 100)       0           ['reshape_1[0][0]']              
                                                                                                  
 gru (GRU)                      [(None, 100),        60600       ['reshape_1[0][0]']              
                                 (None, 100)]                                                     
                                                                                                  
 gru_1 (GRU)                    [(None, 5),          1605        ['pre_skip_trans[0][0]']         
                                 (None, 5)]                                                       
                                                                                                  
 dropout_1 (Dropout)            (None, 100)          0           ['gru[0][1]']                    
                                                                                                  
 post_skip_trans (PostSkipTrans  (None, 95)          0           ['gru_1[0][1]',                  
 )                                                                'input_1[0][0]']                
                                                                                                  
 pre_ar_trans (PreARTrans)      (None, 24)           0           ['input_1[0][0]']                
                                                                                                  
 concatenate (Concatenate)      (None, 195)          0           ['dropout_1[0][0]',              
                                                                  'post_skip_trans[0][0]']        
                                                                                                  
 flatten_1 (Flatten)            (None, 24)           0           ['pre_ar_trans[0][0]']  
 
 flatten (Flatten)              (None, 195)          0           ['concatenate[0][0]']            
                                                                                                  
 dense_1 (Dense)                (None, 1)            25          ['flatten_1[0][0]']              
                                                                                                  
 dense (Dense)                  (None, 1)            196         ['flatten[0][0]']                
                                                                                                  
 post_ar_trans (PostARTrans)    (None, 1)            0           ['dense_1[0][0]',                
                                                                  'input_1[0][0]']                
                                                                                                  
 add (Add)                      (None, 1)            0           ['dense[0][0]',                  
                                                                  'post_ar_trans[0][0]']          
                                                                                                  
==================================================================================================
Total params: 63,126
Trainable params: 63,126
Non-trainable params: 0
__________________________________________________________________________________________________
The model training process has been started.
Epoch 1/2
500/500 [==============================] - 14s 23ms/step - loss: 0.2693 - val_loss: 0.0397
Epoch 2/2
500/500 [==============================] - 12s 24ms/step - loss: 0.0500 - val_loss: 0.0092
Model training process completed

The model is being saved
1/1 [==============================] - 0s 240ms/step
1/1 [==============================] - 0s 16ms/step
1/1 [==============================] - 0s 10ms/step
1/1 [==============================] - 0s 16ms/step
            Salecount   Predicts
Date                            
2022-03-07         71  79.437263
2022-03-14         84  84.282906
2022-03-21         90  88.096298
2022-03-28         87  82.875603
MAPE: 3.576822717339706

```

```python 
forecast

            Predicts   Actual
Date                            
2022-03-07         71  79.437263
2022-03-14         84  84.282906
2022-03-21         90  88.096298
2022-03-28         87  82.875603
```

```python 
evaluate

{'mean_absolute_percentage_error': 3.576822717339706,
 'mean_absolute_error': 14.02137889193878,
 'mean_squared_error': 3485.26570064559}
```


he Time Series module helps to create many basic models
without using much code and helps to understand which models 
work better without any parameter adjustments.
```python 
from perlib.piplines.dpipline import Timeseries
pipline = Timeseries(dataFrame=data,
                       y="Values",
                       dateColumn=False,
                       process=False,
                       epoch=2,
                       forecastNumber= 7,
                       models="all")
predictions = pipline.fit()

            mean_absolute_percentage_error | mean_absolute_error  | mean_squared_error
LSTNET                              14.05  |                67.70 |  5990.35
LSTM                                7.03   |                38.28 |  2250.69
BILSTM                              13.21  |                68.22 |  6661.60
CONVLSTM                            9.62   |                48.06 |  2773.69
TCN                                 12.03  |                65.44 |  6423.10
RNN                                 11.53  |                59.33 |  4793.62
ARIMA                               50.18  |                261.14|  74654.48
SARIMA                              10.48  |                51.25 |  3238.20
```


With the 'summarize' function you can see quick and simple analysis results.
```python 
summarize(dataFrame=data)
```


With the 'auto' function under 'preprocess', you can prepare the data using general preprocessing.
```python 
preprocess.auto(dataFrame=data)

12-2022 15:04:36.22    - DEBUG - Conversion to DATETIME succeeded for feature "Date"
27-12-2022 15:04:36.23 - INFO - Completed conversion of DATETIME features in 0.0097 seconds
27-12-2022 15:04:36.23 - INFO - Started encoding categorical features... Method: "AUTO"
27-12-2022 15:04:36.23 - DEBUG - Skipped encoding for DATETIME feature "Date"
27-12-2022 15:04:36.23 - INFO - Completed encoding of categorical features in 0.001252 seconds
27-12-2022 15:04:36.23 - INFO - Started feature type conversion...
27-12-2022 15:04:36.23 - DEBUG - Conversion to type INT succeeded for feature "Salecount"
27-12-2022 15:04:36.24 - DEBUG - Conversion to type INT succeeded for feature "Day"
27-12-2022 15:04:36.24 - DEBUG - Conversion to type INT succeeded for feature "Month"
27-12-2022 15:04:36.24 - DEBUG - Conversion to type INT succeeded for feature "Year"
27-12-2022 15:04:36.24 - INFO - Completed feature type conversion for 4 feature(s) in 0.00796 seconds
27-12-2022 15:04:36.24 - INFO - Started validation of input parameters...
27-12-2022 15:04:36.24 - INFO - Completed validation of input parameters
27-12-2022 15:04:36.24 - INFO - AutoProcess process completed in 0.034259 seconds
```



If you want to build it yourself;
```python 
from perlib.core.models.dmodels import models
from perlib.core.train import dTrain
from perlib.core.tester import dTester
```
 You can use many features by calling the 'dataPrepration' function for data preparation operations.
```python 
data = dataPrepration.read_data(path="./dataset/Veriler/ayakkabı_haftalık.xlsx")
```
```python 
data = dataPrepration.trainingFordate_range(dataFrame=data,dt1="2013-01-01",dt2="2022-01-01")
```

You can use the 'preprocess' function for data preprocessing.
```python 
data = preprocess.missing_num(dataFrame=data)
```
```python 
data = preprocess.find_outliers(dataFrame=data)
```
```python 
data = preprocess.encode_cat(dataFrame=data)
```
```python 
data = preprocess.dublicates(dataFrame=data,mode="auto")
```

You should create an architecture like below.
```python 
layers = {"Layer": {"unit":[150,100]
                  ,"activation":["tanh","tanh"],
                    "dropout"  :[0.2,0.2]
                  }}
```

You can set each parameter below it by calling the 'req_info' object.
```python 
req_info.layers = layers
req_info.modelname = "lstm"
req_info.epoch  =  3
req_info.targetCol = "Salecount"
req_info.forecastingStartDate = "2021-05-01"
req_info.period = "daily"
req_info.forecastNumber = 18
req_info.scaler = "standard"
```

It will be prepared after importing it into models.
```python 
s = models(req_info)
```

.build() func. You can see your architecture with
```python 
s.build_model()
```
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 24, 150)           91200     
                                                                 
 dropout_2 (Dropout)         (None, 24, 150)           0         
                                                                 
 lstm_1 (LSTM)               (None, 24, 100)           100400    
                                                                 
 dropout_3 (Dropout)         (None, 24, 100)           0         
                                                                 
 lstm_2 (LSTM)               (None, 50)                30200     
                                                                 
 dropout_4 (Dropout)         (None, 50)                0         
                                                                 
 dense_2 (Dense)             (None, 1)                 51        
                                                                 
=================================================================
Total params: 221,851
Trainable params: 221,851
Non-trainable params: 0
_________________________________________________________________
```

After sending the dataframe and the prepared architecture to the dTrain, you can start the training process by calling the .fit() function.
```python 
train = dTrain(dataFrame=data,object=s)
train.fit()
```
After the training is completed, you can see the results by giving the dataFrame,object,path,metric parameters to 'dTester'.
```python 
t = dTester(dataFrame=data,object=s,path="Data-Lstm-2022-12-14-19-56-28.h5",metric="mape")
```
```python 
t.forecast()
```
```
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 20ms/step
1/1 [==============================] - 0s 19ms/step
1/1 [==============================] - 0s 20ms/step
1/1 [==============================] - 0s 20ms/step
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 20ms/step
1/1 [==============================] - 0s 20ms/step
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 21ms/step
1/1 [==============================] - 0s 22ms/step
1/1 [==============================] - 0s 21ms/step

```

```python 
t.evaluate()

MAPE: 3.35
```

```python 
from perlib.core.models.smodels import models as armodels
from perlib.core.train import  sTrain
from perlib.core.tester import sTester
```
```python 
aR_info.modelname = "sarima"
aR_info.forcastingStartDate = "2022-6-10"
```
```python 
ar = armodels(aR_info)
#train = sTrain(dataFrame=data,object=ar)
res = train.fit()
```
```python 
r = sTester(dataFrame=data,object=ar,path="Data-sarima-2022-12-30-23-49-03.pkl")
r.forecast()
r.evaluate()
```

```python 
from perlib.core.models.mmodels import models
from perlib.core.train import mTrain
```

```python 
m_info.testsize = .01
m_info.y        = "quality"
m_info.modelname= "SVC"
m_info.auto  = False
```

```python 
m = models(m_info)
train = mTrain(dataFrame=data,object=m)
preds, evaluate = train.predict()
```

```python 
# If you want to make any other data predictions you can use the train.tester
# func after train.predict. You can make predictions with
predicts = train.tester(path="Data-SVR-2023-01-08-09-50-37.pkl", testData=data.iloc[:,1:][-20:])
```

