import os
from .encode import *
from .imputer import *
from .outliers import *
import sys
columnsDate = ["Time","TIME","time","Datetime","datetime","DATETİME","TARİH",
                       "Tarih","tarih","timestamp","TIMESTAMP","Timestamp","date","Date","DATE"]

#Taken as reference https://github.com/elisemercury/AutoClean.
# preprocessing algorithms will be developed.

class Adjust:

    def convert_datetime(self, df):
        # function for extracting of datetime values in the data
        if self.extract_datetime:
            logger.info('Started conversion of DATETIME features... Granularity: {}', self.extract_datetime)
            start = timer()
            cols = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns)
            for feature in cols:
                try:
                    # convert features encoded as strings to type datetime ['D','M','Y','h','m','s']
                    df[feature] = pd.to_datetime(df[feature], infer_datetime_format=True)
                    try:
                        df['Day'] = pd.to_datetime(df[feature]).dt.day

                        if self.extract_datetime in ['auto', 'M', 'Y', 'h', 'm', 's']:
                            df['Month'] = pd.to_datetime(df[feature]).dt.month

                            if self.extract_datetime in ['auto', 'Y', 'h', 'm', 's']:
                                df['Year'] = pd.to_datetime(df[feature]).dt.year

                                if self.extract_datetime in ['auto', 'h', 'm', 's']:
                                    df['Hour'] = pd.to_datetime(df[feature]).dt.hour

                                    if self.extract_datetime in ['auto', 'm', 's']:
                                        df['Minute'] = pd.to_datetime(df[feature]).dt.minute

                                        if self.extract_datetime in ['auto', 's']:
                                            df['Sec'] = pd.to_datetime(df[feature]).dt.second

                        logger.debug('Conversion to DATETIME succeeded for feature "{}"', feature)

                        try:
                            # check if entries for the extracted dates/times are non-NULL, otherwise drop
                            if (df['Hour'] == 0).all() and (df['Minute'] == 0).all() and (df['Sec'] == 0).all():
                                df.drop('Hour', inplace=True, axis=1)
                                df.drop('Minute', inplace=True, axis=1)
                                df.drop('Sec', inplace=True, axis=1)
                            elif (df['Day'] == 0).all() and (df['Month'] == 0).all() and (df['Year'] == 0).all():
                                df.drop('Day', inplace=True, axis=1)
                                df.drop('Month', inplace=True, axis=1)
                                df.drop('Year', inplace=True, axis=1)
                        except:
                            pass
                    except:
                        # feature cannot be converted to datetime
                        logger.warning('Conversion to DATETIME failed for "{}"', feature)
                except:
                    pass
            end = timer()
            logger.info('Completed conversion of DATETIME features in {} seconds', round(end - start, 4))
        else:
            logger.info('Skipped datetime feature conversion')
        return df

    def round_values(self, df, input_data):
        # function that checks datatypes of features and converts them if necessary
        if self.duplicates or self.missing_num or self.missing_categ or self.outliers or self.encode_categ or self.extract_datetime:
            logger.info('Started feature type conversion...')
            start = timer()
            counter = 0
            cols_num = df.select_dtypes(include=np.number).columns
            for feature in cols_num:
                # check if all values are integers
                if (df[feature].fillna(-9999) % 1 == 0).all():
                    try:
                        # encode FLOATs with only 0 as decimals to INT
                        df[feature] = df[feature].astype('Int64')
                        counter += 1
                        logger.debug('Conversion to type INT succeeded for feature "{}"', feature)
                    except:
                        logger.warning('Conversion to type INT failed for feature "{}"', feature)
                else:
                    try:
                        df[feature] = df[feature].astype(float)
                        # round the number of decimals of FLOATs back to original
                        dec = None
                        for value in input_data[feature]:
                            try:
                                if dec == None:
                                    dec = str(value)[::-1].find('.')
                                else:
                                    if str(value)[::-1].find('.') > dec:
                                        dec = str(value)[::-1].find('.')
                            except:
                                pass
                        df[feature] = df[feature].round(decimals=dec)
                        counter += 1
                        logger.debug('Conversion to type FLOAT succeeded for feature "{}"', feature)
                    except:
                        logger.warning('Conversion to type FLOAT failed for feature "{}"', feature)
            end = timer()
            logger.info('Completed feature type conversion for {} feature(s) in {} seconds', counter,
                        round(end - start, 6))
        else:
            logger.info('Skipped feature type conversion')
        return df

class Duplicates:

    def handle(self, df, subset=None):
        if self.duplicates:
            logger.info('Started handling of duplicates... Method: "{}"', str(self.duplicates).upper())
            start = timer()
            original = df.shape
            try:
                df.drop_duplicates(inplace=True, ignore_index=False,subset=subset)
                df = df.reset_index(drop=True)
                new = df.shape
                count = original[0] - new[0]
                if count != 0:
                    logger.debug('Deletion of {} duplicate(s) succeeded', count)
                else:
                    logger.debug('{} missing values found', count)
                end = timer()
                logger.info('Completed handling of duplicates in {} seconds', round(end-start, 6))

            except:
                logger.warning('Handling of duplicates failed')
        else:
            logger.info('Skipped handling of duplicates')
        return df



class Process:

    def __init__(self):
        pass
    def auto(self, dataFrame:pd.DataFrame,verbose=False,logfile=False) -> pd.DataFrame:
        # function for starting the AutoProcess
        self.duplicates = "auto"
        self.missing_num = "auto"
        self.missing_categ = "auto"
        self.outliers = 'winz'
        self.encode_categ = ["auto"]
        self.extract_datetime = "s"   #TODO : insightOut projesi extract datetime kısımları eklenecek. ( geliştirme )
        self.outlier_param = 1.5
        start = timer()
        self._initialize_logger(verbose, logfile)
        df = dataFrame.copy()
        if df.index.name in columnsDate:
            df = df.reset_index()
        else:
            df = df.reset_index(drop=True)
        df = Adjust.convert_datetime(self, df)
        df = Duplicates.handle(self, df)
        df = MissingValues.handle(self, df)
        #df = Outliers.handle(self, df) #TODO : Her data için uygulanmalı mı ? aykırı değerlere son değeri ekliyor.
        df = EncodeCateg.handle(self, df)
        df = Adjust.round_values(self, df, dataFrame)
        self._validate_params(df, verbose, logfile)
        end = timer()
        logger.info('AutoProcess process completed in {} seconds', round(end-start, 6))

        if not verbose:
            print('AutoProcess process completed in', round(end-start, 6), 'seconds')
        if logfile:
            print('Logfile saved to:', os.path.join(os.getcwd(), 'AutoProcess.log'))
        return df

    def find_outliers(self , dataFrame : pd.DataFrame,mode="winz",outlier_param= 1.5) -> pd.DataFrame:
        '''
        outliers (str)..................define how outliers are handled
        'winz' = replaces outliers through winsorization
        'delete' = deletes observations containing outliers
        oberservations are considered outliers if they are outside the lower and upper bound [Q1-1.5*IQR, Q3+1.5*IQR], where IQR is the interquartile range
        to set a custom multiplier use the 'outlier_param' parameter
        False = skips this step
        :param dataFrame:
        :return:
        '''
        self.outliers = mode
        self.outlier_param = outlier_param
        return Outliers.handle(self,dataFrame)

    def missing_num(self,dataFrame : pd.DataFrame,mode = "auto") -> pd.DataFrame:
        '''
        missing_num (str)...............define how NUMERICAL missing values are handled
        'mode' =  automated handling
        'linreg' = uses Linear Regression for predicting missing values
        'knn' = uses K-NN algorithm for imputation
        'mean','median' or 'most_frequent' = uses mean/median/mode imputatiom
        'delete' = deletes observations with missing values
        False = skips this step
        '''
        self.missing_num = mode
        self.missing_categ = mode
        return MissingValues.handle(self,df=dataFrame)

    def encode_cat(self,dataFrame : pd.DataFrame,mode = "auto") -> pd.DataFrame:
        '''
        encode_categ (list).............encode CATEGORICAL features, takes a list as input
        ['auto'] = automated encoding
        ['onehot'] = one-hot-encode all CATEGORICAL features
        ['label'] = label-encode all categ. features
        to encode only specific features add the column name or index: ['onehot', ['col1', 2]]
        False = skips this step
        '''
        self.encode_categ = [mode]
        return EncodeCateg.handle(self, df=dataFrame)

    def dublicates(self,dataFrame : pd.DataFrame,mode = "auto",subset : list =None) -> pd.DataFrame:
        '''
        duplicates (str)................define if duplicates in the data should be handled
        duplicates are rows where all features are identical
        'auto' = automated handling, deletes all copies of duplicates except one
        subset = Default None
        False = skips this step
        :return:
        '''
        self.duplicates =  mode
        return Duplicates.handle(self, dataFrame,subset=None)


    def _initialize_logger(self, verbose=False, logfile = False):
        # function for initializing the logging process
        logger.remove()
        if verbose == True:
            logger.add(sys.stderr, format='{time:DD-MM-YYYY HH:mm:ss.SS} - {level} - {message}')
        if logfile == True:
            logger.add('AutoProcess.log', mode='w', format='{time:DD-MM-YYYY HH:mm:ss.SS} - {level} - {message}')
        return

    def _validate_params(self, df, verbose=False, logfile = False):
        # function for validating the input parameters of the autolean process
        logger.info('Started validation of input parameters...')

        if type(df) != pd.core.frame.DataFrame:
            raise ValueError('Invalid value for "df" parameter.')
        if self.duplicates not in [False, 'auto']:
            raise ValueError('Invalid value for "duplicates" parameter.')
        if self.missing_num not in [False, 'auto', 'knn', 'mean', 'median', 'most_frequent', 'delete']:
            raise ValueError('Invalid value for "missing_num" parameter.')
        if self.missing_categ not in [False, 'auto', 'knn', 'most_frequent', 'delete']:
            raise ValueError('Invalid value for "missing_categ" parameter.')
        if self.outliers not in [False, 'auto', 'winz', 'delete']:
            raise ValueError('Invalid value for "outliers" parameter.')
        if isinstance(self.encode_categ, list):
            if len(self.encode_categ) > 2 and self.encode_categ[0] not in ['auto', 'onehot', 'label']:
                raise ValueError('Invalid value for "encode_categ" parameter.')
            if len(self.encode_categ) == 2:
                if not isinstance(self.encode_categ[1], list):
                    raise ValueError('Invalid value for "encode_categ" parameter.')
        else:
            if not self.encode_categ in ['auto', False]:
                raise ValueError('Invalid value for "encode_categ" parameter.')
        if not isinstance(self.outlier_param, int) and not isinstance(self.outlier_param, float):
            raise ValueError('Invalid value for "outlier_param" parameter.')
        if self.extract_datetime not in [False, 'auto', 'D', 'M', 'Y', 'h', 'm', 's']:
            raise ValueError('Invalid value for "extract_datetime" parameter.')
        if not isinstance(verbose, bool):
            raise ValueError('Invalid value for "verbose" parameter.')
        if not isinstance(logfile, bool):
            raise ValueError('Invalid value for "logfile" parameter.')

        logger.info('Completed validation of input parameters')
        return


