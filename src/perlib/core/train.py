import os
import pandas as pd
import joblib
from .models.smodels import *
from ..preprocessing.preparate import dataPrepration
import tensorflow as tf
from datetime import datetime
from .models.lstnet import LSTNetModel
import json
from .req_utils import *
from itertools import product
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from ..piplines.mpipline import Regressor,Classifier
from .models.mmodels import models as mmodels
from sklearn.svm import *
from sklearn.ensemble import *
from xgboost import *
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.linear_model import *
from sklearn.cluster import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV,IsotonicRegression
from .metrics.regression import *
from .metrics.classification import *
from .metrics.regression import __ALL__ as r__ALL__
from .metrics.classification import __ALL__ as c__ALL__
from ..preprocessing._utils.tools import to_df
from .tester import dTester
from ..preprocessing._split import train_test_split
class mTrain:
    def __init__(self,
                 dataFrame: pd.DataFrame,
                 object = None
                 ):
        self.pr = dataPrepration()
        self.dataFrame = dataFrame
        self.object = object

    def _save_request_param(self,name):
        with open(str(name)+".json", "w") as outfile:
            json.dump(str(self.object.m_info.__dict__), outfile)

    def get_name_model(self,model=None):
        if self.object.m_info.auto is False:
            return self.object.m_info.modelname
        else:
            return type(model).__name__
    def check_mod(self):
        if self.object.m_info.modelname:
            if self.object.m_info.auto is False:
                if self.object.m_info.modelname in reg:
                    return Regressor
                else:
                    return Classifier
            else:
                self.dataFrame[[col for col in self.dataFrame.columns if self.dataFrame[col].dtypes == self.dataFrame\
                    [self.object.m_info.y].dtypes.name]] = \
                    self.dataFrame[[col for col in self.dataFrame.columns if self.dataFrame\
                        [col].dtypes == self.dataFrame[self.object.m_info.y].dtypes.name]].astype('category')
                if hasattr(self.dataFrame[self.object.m_info.y], "cat"):
                    return Classifier
                else:
                    return Regressor
    def c_opt(self):
        if self.object.m_info.auto:
            mod = self.check_mod()
            model, predictions = mmodels.opt(dataFrame=self.dataFrame,
                                             testsize=self.object.m_info.testsize,
                                             y=self.object.m_info.y,
                                             mod=mod)
            print(model)
            return model.sort_values("Accuracy", ascending=False).head(1).index.tolist()[0]
        else:
            return False

    #def calculate(self,actual, Yhat):
    #    if self.check_mod().__name__ == "Regression":
    #        for m, n in r__ALL__.items():
    #            if bool(self.object.m_info.metric):
    #                if self.object.m_info.metric == m or self.object.m_info.metric == n:
    #                    if m == "mean_absolute_percentage_error":
    #                        return "MAPE: " + str(mean_absolute_percentage_error(actual, Yhat))
    #            else:
    #                return "MAPE: " + str(mean_absolute_percentage_error(actual, Yhat))
    #    else:
    #        for m, n in c__ALL__.items():
    #            if bool(self.object.m_info.metric):
    #                if self.object.m_info.metric == m or self.object.m_info.metric == n:
    #                    if m == "accuracy_score":
    #                        return "ACC: " + str(accuracy_score(actual, Yhat))
    #            else:
    #                return "ACC: " + str(accuracy_score(actual, Yhat))



    def save_model(self,model):
        prefix="models/"
        model_name = f'Data-{mTrain.get_name_model(self,model=model)}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        joblib.dump(model,prefix+model_name + '.pkl')
        mTrain._save_request_param(self,name=prefix+model_name)

    def check_dataFrame(self):
        if type(self.dataFrame) is pd.DataFrame or isinstance(self.dataFrame, pd.DataFrame):
            if self.dataFrame.shape[0] == 0:
                raise ValueError('Data is empty.')
            return True
        else:
            raise TypeError("must be datafarame")

    def data_split_(self):
        X = self.dataFrame.loc[:, self.dataFrame.columns != self.object.m_info.y]
        y = self.dataFrame[[self.object.m_info.y]]
        return X,y

    def _scaler(self,X,y):
        self.scalerX = self.pr.get_scaler(self.object.m_info.scaler)
        self.scalery = self.pr.get_scaler(self.object.m_info.scaler)
        self.X = self.scalerX.fit_transform(X)
        if self.check_mod().__name__ == "Regressor":
            self.y = self.scalery.fit_transform(y)
        else:
            self.y=y
        return self.X,self.y,self.scalerX,self.scalery

    def tr(self):
        if self.check_dataFrame():
            if self.object.m_info.auto:
                self.object.m_info.modelname = self.c_opt()
            X,y = self.data_split_()
            X,y,scalerX,scalery =self._scaler(X,y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.object.m_info.testsize)
            m = eval(self.object.m_info.modelname)
            return m,scalerX,scalery,X_train,X_test, y_train, y_test

    def fit(self):
        m,scalerX,scalery,X_train,X_test, y_train, y_test = self.tr()
        model = m()
        model.fit(X_train, y_train)
        return model,scalery,scalerX,X_test, y_test

    def predict(self,metric= "mape"):
        """
        :param metric:
        :return:
        """
        model_fit,scalery,scalerX,X_test,y_test = self.fit()
        if self.check_mod().__name__ == "Regressor":
            preds = scalery.inverse_transform(model_fit.predict(X_test).reshape(-1, 1))
            y_test = scalery.inverse_transform(y_test)
        else:
            preds = model_fit.predict(X_test)
        preds = to_df(preds, columns=["Predicts"])
        preds["Actual"] = y_test
        self.save_model(model=model_fit)
        evaluate = dTester.calculate(preds.Actual.values, preds.Predicts.values,metric=metric)
        return preds, evaluate

    def tester(self,path,testData):
        model = joblib.load(path)
        preds = model.predict(self.scalerX.fit_transform(testData))
        preds = preds.reshape(-1,1)
        inverse_data = self.scalery.inverse_transform(preds)
        predicts_data = pd.DataFrame(inverse_data,columns=["Predicts"])
        return predicts_data

class sTrain:
    def __init__(self,
                 dataFrame: pd.DataFrame,
                 verbose = None,
                 object = None
                 ):
        self.dataFrame = dataFrame
        self.object    = object
        self.verbose   = verbose
    def _save_request_param(self,name):
        with open(str(name)+".json", "w") as outfile:
            json.dump(str(self.object.aR_info.__dict__), outfile)
    def get_name_model(self):
        return self.object.aR_info.modelname


    def fit(self):
        prefix="models/"
        if type(self.dataFrame) is pd.DataFrame or isinstance(self.dataFrame, pd.DataFrame):
            if self.dataFrame.shape[0] == 0:
                raise ValueError('Data is empty.')
        else:
            raise TypeError("must be datafarame")
        check_forecast_date(
            dataFrame=self.dataFrame,
            info=self.object.aR_info
        )
        column = self.dataFrame.columns.tolist()[0]
        self.dataFrame[column] = self.dataFrame[column].astype(float)
        data = np.log(self.dataFrame)
        data_train = data[data.index < self.object.aR_info.forecastingStartDate]
        data_test = data[data.index > data_train.index[-1]]
        params = product(self.object.aR_info.max_p,
                         self.object.aR_info.max_d,
                         self.object.aR_info.max_q,
                         self.object.aR_info.max_P,
                         self.object.aR_info.max_D,
                         self.object.aR_info.max_Q)

        params_list = list(params)
        res_df = models.opt(name=self.object.aR_info.modelname,params=params_list, s=self.object.aR_info.s, train=data_train,test=data_test)
        order = res_df.iloc[0][0]
        or_a = order[:3]
        or_s = order[3:]
        if self.object.aR_info.modelname.lower() == "sarima":
            model_fit = SARIMAX(data_train, order=or_a, seasonal_order=(or_s[0],
                                                                        or_s[1],
                                                                        or_s[2],
                                                                        self.object.aR_info.s)).fit()
        else:
            model_fit = ARIMA(data_train, order=(or_a)).fit()
        model_fit.summary()
        model_name = f'Data-{self.get_name_model()}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        model_fit.save(prefix+model_name+'.pkl')
        self._save_request_param(name=prefix+model_name)
        return model_fit

class dTrain:
    def __init__(self,
                 dataFrame: pd.DataFrame,
                 object = None,
                 verbose = 1
                 ):
        self.verbose = verbose
        self.pr = dataPrepration()
        self.dataFrame = dataFrame
        self.object = object
        self.model = self.model()

    def model(self):
        scaler = self.pr.get_scaler(self.object.req_info.scaler)
        if type(self.dataFrame) is pd.DataFrame or isinstance(self.dataFrame, pd.DataFrame):
            if self.dataFrame.shape[0] == 0:
                raise ValueError('Data is empty.')
        else:
            raise TypeError("must be datafarame")
        check_forecast_date(
            dataFrame=self.dataFrame,
            info=self.object.req_info
        )
        self.dataFrame = self.pr.trainingFordate_range(dataFrame=self.dataFrame,
                                                              dt1=self.dataFrame.index[0],
                                                              dt2=self.object.req_info.forecastingStartDate)

        dataset = scaler.fit_transform(self.dataFrame)
        X, y = self.pr.unvariate_data_create_dataset(dataset=dataset, window=self.object.req_info.lookback)
        BATCH_SIZE = self.object.req_info.batch_size
        self.BUFFER_SIZE = 150
        train_data_multi = tf.data.Dataset.from_tensor_slices((X, y))
        train_data_multi = train_data_multi.cache().shuffle(self.BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        val_data_multi = tf.data.Dataset.from_tensor_slices((X, y))
        val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
        if self.object.req_info.modelname.lower() == "lstnet":
            model = LSTNetModel(input_shape=X.shape,
                                lookback=self.object.req_info.lookback)
        else:
            self.object.set_inputShape((X.shape[-2:]))
            self.object.build_model()
            model = self.object.model_multi
        self.train_data_multi = train_data_multi
        self.val_data_multi   = val_data_multi
        try:
            self.name = model.input_names[0]
        except:
            self.name = "Bilstm"
        return model

    def __multiDataSplit(self, dataFrame = pd.DataFrame):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        Xn = self.dataFrame.iloc[:,0:].values
        yn = self.dataFrame.loc[:,self.dataFrame.columns == self.object.req_info.targetCol].values
        return Xn,yn

    def get_name_model(self):
        return self.object.req_info.modelname

    def _save_json_model_param(self, model, name):
        json_model = model.to_json()
        with open(str(name)+'.json', 'w') as json_file:
            json_file.write(json_model)

    def _save_request_param(self,name):
        with open(str(name)+".json", "w") as outfile:
            json.dump(self.object.req_info.__dict__, outfile)

    def _check_modelName(self):
        return self.object.req_info.modelname

    def _save_format(self):
        if self.object.req_info.modelname == "tcn":
            return ".tf"
        else:
            return ".h5"

    def fit(self):
        prefix="models/"
        model_name = f'Data-{self.get_name_model()}-{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,),
                          tf.keras.callbacks.ModelCheckpoint(filepath=prefix+model_name+str(self._save_format()), monitor='val_loss', mode='min',
                                                          save_freq='epoch', save_best_only=True, )]
        self.model =  self.model.fit(self.train_data_multi, batch_size=self.object.req_info.batch_size,
                              steps_per_epoch = 500,
                              epochs=self.object.req_info.epoch,
                              validation_data=self.val_data_multi,
                              validation_steps=50,
                              verbose=self.verbose,
                              callbacks = callbacks_list)
        self._save_request_param(name=prefix+model_name)
        #self._save_json_model_param(self.model.model,model_name)
        return self.model


    """
    Multivariate
    """
    #def model(self):
    #    scX, scY, X_data, Y_data = \
    #        self.pr.scaler(dataFrame=self.dataFrame, col=self.col)
    #    X,y = self.pr.multivariate_data_create_dataset(dataset=X_data,
    #                                                   target=Y_data,window=self.object.req_lstm.lookback)
    #    BATCH_SIZE = self.object.req_lstm.batch_size
    #    self.BUFFER_SIZE = 150
#
    #    train_data_multi = tf.data.Dataset.from_tensor_slices((X, y))
    #    train_data_multi = train_data_multi.cache().shuffle(self.BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    #    val_data_multi = tf.data.Dataset.from_tensor_slices((X, y))
    #    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()
    #    self.object.set_inputShape((X.shape[-2:]))
    #    model = self.object.build_model()
    #    self.train_data_multi = train_data_multi
    #    self.val_data_multi   = val_data_multi
    #    self.name = model.input_names[0]
    #    self.scX = scX
    #    self.scY = scY
    #    return model