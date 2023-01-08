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
                 dataFrame           ,
                 y              : str,
                 dateColumn     : str,
                 models         = "all",
                 verbose        = 0,
                 epoch          = 2,
                 process        = False,
                 forecastNumber = 24
                 ):

        self.dataFrame      = dataFrame
        self.y              = y
        self.dateColumn     = dateColumn
        self.models         = models
        self.verbose        = verbose
        self.process        = process
        self.epoch          = epoch
        self.forecastNumber = forecastNumber

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
                    temp_list = []
                    for models in self.models:
                        full_name = (models.__name__, models)
                        temp_list.append(full_name)
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
                                            dateColumn=self.dateColumn,
                                            process=self.process,
                                            forecastNumber=self.forecastNumber,
                                            metric=["mape","mae","mse"],
                                            epoch=self.epoch,
                                            forecastingStartDate=False,
                                            verbose= self.verbose
                                            ))
            names.append(model)
            metrics_.append(evaluate)
        predictions = pd.DataFrame()
        for m,n in zip(metrics_,names):
             predictions =  predictions.append(to_df(m,[n]))
        return  predictions