import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler, RobustScaler
from statsmodels.tsa.stattools import adfuller as ADF
from sklearn.model_selection import train_test_split
#import pymrmr
from sklearn.feature_selection import f_regression
import math
import operator
import sqlite3
from ._utils.dataframe import read_pandas

columnsDate = ["Time","TIME","time","Datetime","datetime","DATETİME","TARİH",
                       "Tarih","tarih","timestamp","TIMESTAMP","Timestamp","date","Date","DATE"]

class dataPrepration:

    def __init__(self):
        self.dataFrame = False
        self.col       = False
        #self.dataFrame = self.__datatimeinsert()
        #self.dataFrame = self.insertFirstcolumn(col=self.col)

    def read_data(self, path:str,delimiter=None) -> pd.DataFrame:
        self.dataFrame = read_pandas(path,delimiter=delimiter)
        return self.dataFrame
#
    def load_sql( self,query:str,path:str):
        con = sqlite3.connect(path)
        self.dataFrame = pd.read_sql(query,con=con)
        return self.dataFrame

    def _date_check(self):
        if self.dataFrame.index.name in columnsDate:
            self.dataFrame = self.dataFrame.reset_index()
            dcol = list(set(self.dataFrame.columns.tolist()).intersection(columnsDate))[0]
            self.dataFrame[dcol] = pd.to_datetime(self.dataFrame[dcol])
        elif len(list(set(self.dataFrame.columns.tolist()).intersection(columnsDate))) > 0:
            dcol = list(set(self.dataFrame.columns.tolist()).intersection(columnsDate))[0]
            self.dataFrame[dcol] = pd.to_datetime(self.dataFrame[dcol])
        else:
            pass
        return self.dataFrame,dcol

    def datatimeinsert(self, dataFrame:pd.DataFrame = None) -> pd.DataFrame:
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        self.dataFrame,dcol = self._date_check()
        try:
            self.dataFrame[dcol] = self.dataFrame[dcol].astype('datetime64[ns]')
            self.dataFrame.index = self.dataFrame[dcol]
            del self.dataFrame[dcol]
        except: pass
        return self.dataFrame

    def insertFirstcolumn(self , col  , dataFrame=None):

        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame

        self.dataFrame = self.dataFrame.sort_index()
        first_column = self.dataFrame.pop(col)
        self.dataFrame.insert(0, col, first_column)
        return self.dataFrame

    def trainingFordate_range(self, dt1, dt2, dataFrame = None):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        try:
            self.dataFrame = self.datatimeinsert().sort_index()
        except:
            pass
        return self.dataFrame[(self.dataFrame.index > dt1) & (self.dataFrame.index < dt2)]


    def train_test_split(self, dataFrame = None, target=None, test_size=None, tX=None, tY=None,
                         train_size=None,
                         random_state=None,
                         shuffle=True,
                         stratify=None,
                         ):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        try:
            Y = self.dataFrame.loc[:, [target]].values
            X = self.dataFrame.loc[:, self.dataFrame.columns != target].values
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size,
                                                                random_state=random_state, shuffle=shuffle,
                                                                stratify=stratify)
        except:
            X_train, X_test, y_train, y_test = train_test_split(tX, tY, test_size=test_size, train_size=train_size,
                                                                random_state=random_state, shuffle=shuffle,
                                                                stratify=stratify)

        print("X_train shape :", X_train.shape)
        print("X_test shape  :", X_test.shape)
        print("Y_train shape :", y_train.shape)
        print("Y test shape  :", y_test.shape)
        return X_train, X_test, y_train, y_test

    def clean_dataset(self,df:pd.DataFrame):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    def split( X_data , y_data , test_split : int  ):
        # Splitting the data into train and test
        X_train= X_data[:-test_split]
        X_test= X_data[-test_split:]
        y_train=y_data[:-test_split]
        y_test=y_data[-test_split:]

        print("X_train shape :", X_train.shape)
        print("X_test shape  :", X_test.shape)
        print("Y_train shape :", y_train.shape)
        print("Y test shape  :", y_test.shape)

        return X_train , X_test , y_train , y_test

    def diff( self , col  ,dataFrame = None) :
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        self.dataFrame["Diff1"] = self.dataFrame[col].diff()
        return self.dataFrame

    def reverse_diff(col, dataFrame):
        return np.r_[dataFrame[col].iloc[0], \
                     dataFrame[dataFrame.loc[:,dataFrame.columns.str.startswith("Diff")]].iloc[1:]].cumsum().astype(int)

    def gauss_Filter(self, col, sigma=0.3, dataFrame = None):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        self.dataFrame[col + "_Gauss" + str(sigma)] = pd.Series(gaussian_filter(self.dataFrame[col], sigma=sigma),
                                                           index=self.dataFrame.index).astype(float)
        return self.dataFrame

    def moving_average(self, col, window=3, dataFrame = None):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        self.dataFrame[col + 'MA' + "_" + str(window)] = self.dataFrame[col].rolling(window=window).mean()
        self.dataFrame = self.dataFrame.dropna()
        return self.dataFrame

    def exponential_Smoothing(self, col, dataFrame = None):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        self.dataFrame['ExponentialSmoothing'] = sm.tsa.ExponentialSmoothing(self.dataFrame[col],
                                                                 trend='add',
                                                                 seasonal_periods=4).fit().fittedvalues.shift(1)
        return self.dataFrame

    def rolling_mean_diff(self, col, window=3, dataFrame = None):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        rolling_mean = self.dataFrame.rolling(window=window).mean()
        self.dataFrame['rolling_mean_diff'] = rolling_mean[col] - rolling_mean[col].shift()
        self.dataFrame = self.dataFrame.dropna()
        return self.dataFrame

    def circ(self,dataFrame = None):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        hours_in_week = 7 * 24
        self.dataFrame,dcol = self._date_check()
        self.dataFrame['CircHourX'] = self.dataFrame[dcol].apply(lambda x: np.cos(x.hour / 24 * 2 * np.pi))
        self.dataFrame['CircHourY'] = self.dataFrame[dcol].apply(lambda x: np.sin(x.hour / 24 * 2 * np.pi))
        self.dataFrame['CircWeekdayX'] = self.dataFrame[dcol].apply(lambda x: np.cos(x.weekday() * 24 + x.hour / hours_in_week * 2 * np.pi))
        self.dataFrame['CircWeekdayY'] = self.dataFrame[dcol].apply(lambda x: np.sin(x.weekday() * 24 + x.hour / hours_in_week * 2 * np.pi))
        self.dataFrame['CircDayX'] = self.dataFrame[dcol].apply(lambda x: np.cos(x.day * 24 + x.hour / x.daysinmonth * 2 * np.pi))
        self.dataFrame['CircDayY'] = self.dataFrame[dcol].apply(lambda x: np.sin(x.day * 24 + x.hour / x.daysinmonth * 2 * np.pi))
        self.dataFrame['CircMonthX'] = self.dataFrame[dcol].apply(lambda x: np.cos(x.dayofyear / 365 * 2 * np.pi))
        self.dataFrame['CircMonthY'] = self.dataFrame[dcol].apply(lambda x: np.sin(x.dayofyear / 365 * 2 * np.pi))
        self.dataFrame = self.dataFrame.set_index(dcol)
        return self.dataFrame

    def generate_time_lags(self, col, n_lags=False, th=False, firstN=False,dataFrame = None):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        def glag(df, columns, n_lags: int):
            df_L = df.copy()
            df_L = df_L[[columns]]
            for n in range(1, n_lags + 1):
                df_L[f"lag{n}"] = df_L[columns].shift(n)
            return pd.concat([df, df_L.iloc[:, 1:]], axis=1).dropna()

        dict_ = {'Lag': [],
                 'Autocor': []}

        for lag in range(1, int(np.sqrt(self.dataFrame.shape[0]))):
            shift = self.dataFrame[col].autocorr(lag)
            dict_['Lag'].append(lag)
            dict_['Autocor'].append(shift)
        autocorr_df = pd.DataFrame(dict_)
        autocorr_df = autocorr_df.sort_values("Autocor", ascending=False).reset_index(drop=True)

        if bool(n_lags) is True:
            return glag(self.dataFrame, col, n_lags).dropna()

        elif bool(th) is True:
            autocorr_df = autocorr_df[autocorr_df.Autocor > th]
            lags = ["lag" + str(x) for x in autocorr_df.Lag.tolist()]
            df_c = self.dataFrame.copy()
            df_c = glag(df_c, "Smfdolar", autocorr_df.Lag.max())
            return pd.concat([self.dataFrame, df_c.loc[:, lags]], axis=1).dropna()

        elif bool(firstN) is True:
            autocorr_df = autocorr_df[:firstN]
            lags = ["lag" + str(x) for x in autocorr_df.Lag.tolist()]
            df_c = self.dataFrame.copy()
            df_c = glag(df_c, "Smfdolar", autocorr_df.Lag.max())
            return pd.concat([self.dataFrame, df_c.loc[:, lags]], axis=1).dropna()
        else:
            pass

    def adf_test(self,dataFrame = None, columns=[]):

        if len(columns) == 0:
            raise  TypeError("adf_test() missing 1 required positional argument: columns")

        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        self.dataFrame = self.dataFrame.dropna()
        for col in columns:
            print(f'Augmented Dickey-Fuller Test: {col}')
            result = ADF(self.dataFrame[col], autolag='AIC')

            labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
            out = pd.Series(result[0:4], index=labels)

            for key, val in result[4].items():
                out[f'critical value ({key})'] = val
            print(out.to_string())

            if result[1] <= 0.05:
                print("Strong evidence against the null hypothesis")
                print("Reject the null hypothesis")
                print("Data has no unit root and is stationary")
            else:
                print("Weak evidence against the null hypothesis")
                print("Fail to reject the null hypothesis"),
                print("Data has a unit root and is non-stationary")

    def date_transform(self,dataFrame=None):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        self.dataFrame, dcol = self._date_check()

        if self.dataFrame.index.name in columnsDate:
            self.dataFrame = self.dataFrame.reset_index()

        self.dataFrame['Year'] = self.dataFrame[dcol].dt.year
        self.dataFrame['Month'] = self.dataFrame[dcol].dt.month
        self.dataFrame['Day'] = self.dataFrame[dcol].dt.day
        self.dataFrame['WeekofYear'] = self.dataFrame[dcol].dt.weekofyear
        self.dataFrame['DayofWeek'] = self.dataFrame[dcol].dt.weekday
        self.dataFrame['Hour'] = self.dataFrame[dcol].dt.hour
        try:
            self.dataFrame[dcol] = self.dataFrame[dcol].astype('datetime64[ns]')
            self.dataFrame.index = self.dataFrame[dcol]
            del self.dataFrame[dcol]
        except:
            pass

        return self.dataFrame

    #def mRMR(self, dataFrame = None, method="MIQ", n_features=3):
#
    #    """
    #    First parameter is a pandas DataFrame (http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html) containing the input dataset, discretised as defined in the original paper (for ref. see http://home.penglab.com/proj/mRMR/). The rows of the dataset are the different samples. The first column is the classification (target) variable for each sample. The remaining columns are the different variables (features) which may be selected by the algorithm. (see “Sample Data Sets” at http://home.penglab.com/proj/mRMR/ to download sample dataset to test this algorithm). IMPORTANT: the column names (feature names) should be of type string;
    #    Second parameter is a string which defines the internal Feature Selection method to use (defined in the original paper): possible values are “MIQ” or “MID”;
    #    Third parameter is an integer which defines the number of features that should be selected by the algorithm.
#
    #    """
    #    if isinstance(dataFrame, pd.DataFrame):
    #        self.dataFrame = dataFrame
    #    return pymrmr.mRMR(self.dataFrame, method, n_features)

    def likelihood(self,dataFrame = None, n_features=4):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame

        Xn = self.dataFrame.iloc[:, 1:].values
        yn = self.dataFrame.iloc[:, 0].values
        scX = MinMaxScaler(feature_range=(0, 1))
        scY = MinMaxScaler(feature_range=(0, 1))
        X = scX.fit_transform(Xn)
        y = scY.fit_transform(yn.reshape(-1, 1))
        X_train, X_test, y_train, y_test = self.train_test_split(tX=X, tY=y, test_size=24, random_state=42)
        f_val, p_val = f_regression(X_train, y_train)
        f_val_dict = {}
        p_val_dict = {}
        for i in range(len(f_val)):
            if math.isnan(f_val[i]):
                f_val[i] = 0.0
            f_val_dict[i] = f_val[i]
            if math.isnan(p_val[i]):
                p_val[i] = 0.0
            p_val_dict[i] = p_val[i]

        sorted_f = sorted(f_val_dict.items(), key=operator.itemgetter(1), reverse=True)
        sorted_p = sorted(p_val_dict.items(), key=operator.itemgetter(1), reverse=True)

        feature_indexs = []

        for i in range(0, n_features):
            feature_indexs.append(sorted_f[i][0])

        return self.dataFrame.iloc[:, 1:].iloc[:, feature_indexs].columns.tolist()

    def plb( self, col : str,  period : int , timelag : int,dataFrame=None ):

        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        """
        df      : DataFrame
        col     : Columns
        period  : Period Number
        timelag : Lookback

        """
        for i in range(1, period + 1):
            self.dataFrame["plb" + "_" + str(i)] = np.tile(self.dataFrame[:-timelag].iloc[-i * timelag][col], (self.dataFrame.shape[0], 1))
        return self.dataFrame


    def normalizeZeroValues(self,columns ,dataFrame = None):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame

        columnsDate = ['Year', 'Month', 'Day', 'WeekofYear', 'DayofWeek', 'Hour', columns]
        for col in self.dataFrame.loc[:, (self.dataFrame == 0).any()].columns.tolist():
            if col not in columnsDate:
                self.dataFrame.loc[self.dataFrame[col] < 1, col] = np.nan
                self.dataFrame = self.dataFrame.groupby(self.dataFrame.index.date).transform(lambda x: x.fillna(x.mean()))
        return self.dataFrame


    def get_scaler(self,scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

    def inf_clean(self,dataFrame = None):
        if isinstance(dataFrame, pd.DataFrame):
            self.dataFrame = dataFrame
        return self.dataFrame.replace([np.inf, -np.inf], 0, inplace=True)


    def multivariate_data_create_dataset(self, dataset, target, start = 0 , window = 24, horizon = 1,end=None):
        X = []
        y = []
        start = start + window
        if end is None:
            end = len(dataset) - horizon
        for i in range(start, end):
            indices = range(i - window, i)
            X.append(dataset[indices])
            indicey = range(i + 1, i + 1 + horizon)
            y.append(target[indicey])
        X_data,y_data =  np.array(X), np.array(y)
        print('trainX shape == {}.'.format(X_data.shape))
        print('trainY shape == {}.'.format(y_data.shape))

        return X_data, y_data

    def unvariate_data_create_dataset(self, dataset, start=0, window = 24, horizon = 1,end = None):
        dataX = []
        dataY = []

        start = start + window
        if end is None:
          end = len(dataset) - horizon
        for i in range(start, end):
          indicesx = range(i-window, i)
          dataX.append(np.reshape(dataset[indicesx], (window, 1)))
          indicesy = range(i,i+horizon)
          dataY.append(dataset[indicesy])

        return np.array(dataX), np.array(dataY)


