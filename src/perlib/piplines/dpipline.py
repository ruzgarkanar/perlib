from tqdm.auto import tqdm
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from ..preprocessing._utils.tools import to_df
from ..forecaster import get_result

MODELS = ["LSTNET","LSTM","BILSTM","CONVLSTM","TCN","RNN","ARIMA","SARIMA"]

class Timeseries:
    def __init__(self,
                 dataFrame             : pd.DataFrame,
                 y                     : str,
                 dateColumn            : str,
                 models                : str  ="all",
                 verbose               : int  = 0,
                 epoch                 : int  = 2,
                 metrics               : str = "mape",
                 process               : bool = False,
                 forecastNumber        : int  = 24,
                 forecastingStartDate  : str  =False
                 ):

        self.dataFrame            = dataFrame
        self.y                    = y
        self.dateColumn           = dateColumn
        self.models               = models
        self.verbose              = verbose
        self.process              = process
        self.epoch                = epoch
        self.forecastNumber       = forecastNumber
        self.forecastingStartDate = forecastingStartDate
        self.metrics               = metrics

    def fit(self):
        if type(self.dataFrame) is pd.DataFrame or isinstance(self.dataFrame,pd.DataFrame):
            if self.dataFrame.shape[0] == 0:
                raise ValueError('Data is empty.')
        else:
            raise TypeError("must be datafarame")

        if self.models == "all":
            self.models = MODELS
        else:
            if isinstance(self.models,list):
                try:
                    self.models = [x.upper() for x in self.models]
                    temp_list = []
                    for models in self.models:
                        if models in self.models:
                            temp_list.append(models)
                    self.models = temp_list
                except:
                    raise ValueError("Invalid Models(s)")

        names = []
        metrics_ = []
        for model in tqdm(self.models):
            start = time.time()
            forecast,evaluate = tqdm(get_result(dataFrame=self.dataFrame,
                                            y=self.y,
                                            modelName=model,
                                            metric= self.metrics,
                                            dateColumn=self.dateColumn,
                                            process=self.process,
                                            forecastNumber=self.forecastNumber,
                                            epoch=self.epoch,
                                            forecastingStartDate=self.forecastingStartDate,
                                            verbose= self.verbose
                                            ))
            names.append(model)
            metrics_.append(evaluate)
        predictions = pd.DataFrame()
        for m,n in zip(metrics_,names):
            if self.metrics.__len__() == 1:
                predictions = predictions.append(to_df(data=m.split(":")[1], index=[n], columns=[m.split(":")[0]]))
            else:
             predictions =  predictions.append(to_df(m,[n]))
        return  predictions