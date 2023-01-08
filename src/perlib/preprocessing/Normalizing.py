import pandas as pd
import numpy as np
import logging
import pickle
import datetime as dt
from pandas.api.types import is_object_dtype, is_datetime64_any_dtype, is_numeric_dtype


class Normalizer:

    # Autmatic normalization, can do mean-std norm and [0,1] (minmax) norm

    def __init__(self, meanstd=None, minmax=None):
        # meanstd: Column names that will be normalized with mean-std
        # minmax: column names that will be normalized with minmax

        self.meanstd = meanstd
        self.minmax = minmax

        # Parameters that will be inferred from training data
        self.meanstd_means = {}
        self.meanstd_stds = {}
        self.meanstd_count = 0 # is not used at the moment, will be used for accumulative parameter update

        self.minmax_mins = {}
        self.minmax_maxes = {}

        # After parameters are set, this attr. will be set to True
        self.is_params_set = False
        
        logging.info(f"Normalizer created.\nmeanstd columns: {self.meanstd}\nminmax columns: {self.minmax}")

    def normalize(self, dfnow, init=False, accum=False, inplace=True): # todo implement accum
        
        # init: if true, parameters will be set from the input data,
        #       if false, parameters will be fixed and will be used to norm. input data
        # inplace: if true, input dataframe will be modified
        #          if false, a new dataframe with additional normalized columsn will be returned

        if inplace is False:
            dfnow = dfnow.copy()

        if init is False and self.is_params_set is False:
            message = "No parameters to normalize! You may want to set init=True"
            logging.error(message)
            raise Exception(message)

        if self.meanstd is not None:
            for col in self.meanstd:

                if col not in dfnow.columns:
                    logging.warning(f"Column-{col} not in dataframe, skipping!")
                    continue

                logging.info(f"Normalizing {col}")
                
                if init is True:
                    # get parameters from data
                    meannow = dfnow[col].mean()
                    stdnow = dfnow[col].std()
                else:
                    # use parameters to normalize
                    meannow = self.meanstd_means[col]
                    stdnow = self.meanstd_stds[col]
                
                # Normalize:
                dfnow[col] = (dfnow[col] - meannow) / stdnow

                if init is True:
                    # Set parameters 
                    self.meanstd_means[col] = meannow
                    self.meanstd_stds[col] = stdnow
                    self.meanstd_count += dfnow.shape[0]

        if self.minmax is not None:
            for col in self.minmax:

                if col not in dfnow.columns:
                    logging.warning(f"Column-{col} not in dataframe, skipping!")
                    continue
                
                logging.info(f"Normalizing {col}")

                if init is True:
                    # set parameters from data and normalize
                    minnow = dfnow[col].min()
                    dfnow[col] = dfnow[col] - minnow
                    maxnow = dfnow[col].max()
                    dfnow[col] = dfnow[col] / maxnow

                    self.minmax_mins[col] = minnow
                    self.minmax_maxes[col] = maxnow
                else:
                    # use parameters to normalize
                    minnow = self.minmax_mins[col]
                    dfnow[col] = dfnow[col] - minnow
                    maxnow = self.minmax_maxes[col]
                    dfnow[col] = dfnow[col] / maxnow

        if init:
            self.is_params_set = True

        return dfnow

    def denormalize(self, dfnow, inplace=False):
        # Given a dataframe with normalized columns, de-normalize them

        if inplace is False:
            dfnow = dfnow.copy()

        if self.is_params_set is False:
            message = f"No parameters to denormalize with!"
            logging.error(message)
            raise Exception(message)

        if self.meanstd is not None:
            for col in self.meanstd:

                meannow = self.meanstd_means[col]
                stdnow = self.meanstd_stds[col]
                dfnow[col] = (dfnow[col] - meannow) / stdnow
           
        if self.minmax is not None:
            for col in self.minmax:
                
                minnow = self.minmax_mins[col]
                dfnow[col] = dfnow[col] - minnow
                maxnow = self.minmax_maxes[col]
                dfnow[col] = dfnow[col] / maxnow

        return dfnow

    def save(self, path):
        # Save normalizer to pickle object
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def denormalize_arr(self, arrnow, col):

        # Given a normalized numpy array of values and column name, de-normalize
        if col in self.meanstd:
            return arrnow * self.meanstd_stds[col] + self.meanstd_means[col]

        elif col in self.minmax:
            return arrnow *self.minmax_maxes[col] + self.minmax_mins[col] 

        else:
            msg = f"No such column: {col}."
            logging.error(msg)
            raise Exception(msg)
