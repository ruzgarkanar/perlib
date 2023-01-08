from keras.models import load_model
from .req_utils import *
from tcn import TCN
from datetime import timedelta
import pandas as pd
import pickle
import os
from .metrics.regression import *
from .metrics.regression import __ALL__
from .models.lstnet import *
from ..preprocessing.preparate import dataPrepration
import joblib


custom_objects_tcn ={'TCN': TCN}
custom_objects = {
        'PreSkipTrans': PreSkipTrans,
        'PostSkipTrans': PostSkipTrans,
        'PreARTrans': PreARTrans,
        'PostARTrans': PostARTrans
        }


class sTester:
    def __init__(self,
                 dataFrame:pd.DataFrame,
                 object,
                 path    :str,
                 metric  : str = "mape"
                 ):

        self.dataFrame         = dataFrame
        self.path              = path
        self.object            = object
        self.metric            = metric
        check_forecast_date(
            dataFrame=self.dataFrame,
            date=self.object.aR_info.forecastingStartDate,
            number=self.object.aR_info.forecastNumber,
            info=self.object.aR_info
        )

    def _current_folder(self):
        if os.getcwd()[-6:] != "models":
            try:
                os.chdir("./models/")
            except:
                os.mkdir("./models")


    def date_range(self):
        return pd.date_range(start=self.object.aR_info.forecastingStartDate,
                                          periods=self.object.aR_info.forecastNumber)
    def forecast(self):
        self._current_folder()
        self.dataFrame = np.log(self.dataFrame)
        if os.listdir(os.getcwd()).__len__() > 0:
            try:
                with open(self.path, 'rb') as pkl:
                    self.model = pickle.load(pkl)
            except:
                raise OSError("No file or directory found at : {}".format(self.path))
        if bool(self.object.aR_info.forecastingStartDate) is True:
            if str(self.dataFrame.index[-1].date()) != self.object.aR_info.forecastingStartDate:
                forecasts = self.model.predict(start=len(self.dataFrame[:-len(self.dataFrame[self.dataFrame.index > self.object.aR_info.forecastingStartDate]):]), end=len(self.dataFrame), dynamic=True)
                data = pd.DataFrame(np.exp(forecasts.values),columns=["Predicts"],index=self.date_range())
                data["Actual"] = np.exp(dTester.a_data(info=self.object.aR_info,dataFrame=self.dataFrame)).values
                self.actual,self.Yhat = data.Actual.values,forecasts.values
                return data
            else:
                forecasts= self.model.predict(start=len(self.dataFrame)+1, end=len(self.dataFrame)+self.object.aR_info.forecastNumber, dynamic=True)
        else:
            forecasts = self.model.forecast(self.object.aR_info.forecastNumber,alpha=0.05,dynamic = True)
        data = pd.DataFrame(np.exp(forecasts.values),columns=["Predicts"],index=self.date_range())
        return data

    def evaluate(self):
        return dTester.calculate(self.actual,np.exp(self.Yhat),self.metric)

class dTester:
    def __init__(self,
                 dataFrame:pd.DataFrame,
                 object,
                 path    :str,
                 metric  : str = "mape"
                 ):
        self.dataFrame         = dataFrame
        self.path              = path
        self.object            = object
        self.metric            = metric
        self.pr                = dataPrepration()
        self.scaler  = \
            self.pr.get_scaler(self.object.req_info.scaler)
        dataset = self.scaler.fit_transform(self.dataFrame)
        self.count = self.object.req_info.lookback
        check_forecast_date(
            dataFrame=self.dataFrame,
            date=self.object.req_info.forecastingStartDate,
            number=self.object.req_info.forecastNumber,
            info=self.object.req_info
        )

    def get_testdata(self ):
        if self.object.req_info.forecastingStartDate is False:
            return self.dataFrame[-self.object.req_info.lookback:]
        else:
            return self.dataFrame[self.dataFrame.index >
                                  str(self.dataFrame[self.dataFrame.index
                    < self.object.req_info.forecastingStartDate][-self.object.req_info.lookback:].index[0])]

    def get_s_data( self ):
        return self.get_testdata()[:self.count][-self.object.req_info.lookback:]

    def fit_transform(self, array):
        return self.scaler.fit_transform(array.reshape(-1, 1))

    def inverse_transform(self, array):
        return self.scaler.inverse_transform(array)

    def __check(self):
        return self.object.req_info.period.lower()

    def __exist(self):
        if self.create_date_range().__len__():
            return 1

    def create_date_range(self):
        if self.__check() == "montly":
            return pd.date_range(start=str(self.dataFrame.index[-1] + timedelta(days=24)),
                                 periods=self.object.req_info.forecastNumber, freq="m")
        if self.__check() == "hourly":
            return pd.date_range(start=str(self.dataFrame.index[-1] + timedelta(hours=1)),
                                 periods=self.object.req_info.forecastNumber, freq="h")
        elif self.__check() == "daily":
            return pd.date_range(start=str(self.dataFrame.index[-1] + timedelta(days=1)),
                                 periods=self.object.req_info.forecastNumber, freq="d")
        elif self.__check() == "weekly":
            return pd.date_range(start=str(self.dataFrame.index[-1] + timedelta(weeks=1)),
                                 periods=self.object.req_info.forecastNumber, freq="w")
        elif self.__check() == "30min":
            return pd.date_range(start=str(self.dataFrame.index[-1] + timedelta(minutes=30)),
                                 periods=self.object.req_info.forecastNumber, freq="30min")
        elif self.__check() == "15min":
            return pd.date_range(start=str(self.dataFrame.index[-1] + timedelta(minutes=15)),
                                 periods=self.object.req_info.forecastNumber, freq="15min")
        elif self.__check() == "10min":
            return pd.date_range(start=str(self.dataFrame.index[-1] + timedelta(minutes=10)),
                                 periods=self.object.req_info.forecastNumber, freq="10min")
        elif self.__check() == "5min":
            return pd.date_range(start=str(self.dataFrame.index[-1] + timedelta(minutes=5)),
                                 periods=self.object.req_info.forecastNumber, freq="5min")
        elif self.__check() == "1min":
            return pd.date_range(start=str(self.dataFrame.index[-1] + timedelta(minutes=1)),
                                 periods=self.object.req_info.forecastNumber, freq="1min")

    def predict( self,pr_data):
        val_rescaled = self.fit_transform(pr_data)
        val_rescaled = val_rescaled.reshape((1, val_rescaled.shape[0], 1))
        Yhat = self.model.predict(val_rescaled)
        Yhat = self.inverse_transform(Yhat)
        return Yhat

    def _current_folder(self):
        if os.getcwd()[-6:] != "models":
            try:
                os.chdir("./models/")
            except:
                os.mkdir("./models")

    def forecast(self):
        forecasts  = [ ]
        self._current_folder()
        if os.listdir(os.getcwd()).__len__() > 0:
            try:
                if self.object.req_info.modelname.lower() == "lstnet":
                    self.model = load_model(self.path,custom_objects=custom_objects)
                elif self.object.req_info.modelname.lower() == "tcn":
                    self.model = load_model(self.path, custom_objects=custom_objects_tcn)
                else:
                    self.model = load_model(self.path)
            except:
                raise OSError("No file or directory found at : {}".format(self.path))
        if bool(self.object.req_info.forecastingStartDate) is True:
            if str(self.dataFrame.index[-1].date()) != self.object.req_info.forecastingStartDate:
                for i in range(self.object.req_info.forecastNumber):
                    pr_data = np.array(self.get_s_data())
                    forecasts.append(self.predict(pr_data)[0][0])
                    if self.__exist() == 1:
                        if bool(self.__check()):
                            self.count+=1
                data = self.a_data(info=self.object.req_info,dataFrame=self.dataFrame)
                data["Predicts"] = forecasts
                self.actual = self.a_data(info=self.object.req_info,dataFrame=self.dataFrame)[self.object.req_info.targetCol].values
                self.Yhat = forecasts
                return data
            else:
                pr_data = np.array(self.get_s_data())
                for i in range(self.object.req_info.forecastNumber):
                    Yhat = self.predict(pr_data)[0][0]
                    forecasts.append(Yhat)
                    pr_data = np.append(pr_data, Yhat)[-self.object.req_info.lookback:]
        else:
            pr_data = np.array(self.get_s_data())
            for i in range(self.object.req_info.forecastNumber):
                Yhat = self.predict(pr_data)[0][0]
                forecasts.append(Yhat)
                pr_data = np.append(pr_data, Yhat)[-self.object.req_info.lookback:]
        data = pd.DataFrame(forecasts,columns=["Predicts"])
        data.index = self.create_date_range()
        return data

    @staticmethod
    def a_data(info,dataFrame : pd.DataFrame):
        if bool(info.forecastingStartDate) is True and bool(info.forecastNumber) is False:
            return dataFrame[dataFrame.index >=
                                  pd.to_datetime(info.forecastingStartDate)]
        elif bool(info.forecastingStartDate) is True and bool(info.forecastNumber) is True:
            return dataFrame[dataFrame.index >=
                             pd.to_datetime(info.forecastingStartDate)][:info.forecastNumber]
    def evaluate(self):
        return dTester.calculate(self.actual,pd.Series(self.Yhat),self.metric)

    @staticmethod
    def calculate(actual, Yhat, metric):
        dict_ = {}
        if metric:
            if isinstance(metric, list):
                for m in metric:
                    try:
                        name, value = evaluate(m)
                    except:
                        raise ValueError(evaluate(metric))
                    dict_[name] = value(actual, Yhat)
                return dict_
            else:
                try:
                    name, metric = evaluate(metric)
                except:
                    raise ValueError(evaluate(metric))
                return f'{name} : {str(metric(actual, Yhat))}'
