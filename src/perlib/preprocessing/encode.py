from timeit import default_timer as timer
import numpy as np
import pandas as pd
from math import isnan
from sklearn import preprocessing
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class EncodeCateg:

    def handle(self, df):
        # function for encoding of categorical features in the data
        if self.encode_categ:
            if not isinstance(self.encode_categ, list):
                self.encode_categ = ['auto']
            # select non numeric features
            cols_categ = set(df.columns) ^ set(df.select_dtypes(include=np.number).columns) 
            # check if all columns should be encoded
            if len(self.encode_categ) == 1:
                target_cols = cols_categ # encode ALL columns
            else:
                target_cols = self.encode_categ[1] # encode only specific columns
            logger.info('Started encoding categorical features... Method: "{}"', str(self.encode_categ[0]).upper())
            start = timer()
            for feature in target_cols:
                if feature in cols_categ:
                    # columns are column names
                    feature = feature
                else:
                    # columns are indexes
                    feature = df.columns[feature]
                try:
                    # skip encoding of datetime features
                    pd.to_datetime(df[feature])
                    logger.debug('Skipped encoding for DATETIME feature "{}"', feature)
                except:
                    try:
                        if self.encode_categ[0] == 'auto':
                            # ONEHOT encode if not more than 10 unique values to encode
                            if df[feature].nunique() <=10:
                                df = EncodeCateg._to_onehot(self, df, feature)
                                logger.debug('Encoding to ONEHOT succeeded for feature "{}"', feature)
                            # LABEL encode if not more than 20 unique values to encode
                            elif df[feature].nunique() <=20:
                                df = EncodeCateg._to_label(self, df, feature)
                                logger.debug('Encoding to LABEL succeeded for feature "{}"', feature)
                            # skip encoding if more than 20 unique values to encode
                            else:
                                logger.debug('Encoding skipped for feature "{}"', feature)   

                        elif self.encode_categ[0] == 'onehot':
                            df = EncodeCateg._to_onehot(df, feature)
                            logger.debug('Encoding to {} succeeded for feature "{}"', str(self.encode_categ[0]).upper(), feature)
                        elif self.encode_categ[0] == 'label':
                            df = EncodeCateg._to_label(df, feature)
                            logger.debug('Encoding to {} succeeded for feature "{}"', str(self.encode_categ[0]).upper(), feature)      
                    except:
                        logger.warning('Encoding to {} failed for feature "{}"', str(self.encode_categ[0]).upper(), feature)    
            end = timer()
            logger.info('Completed encoding of categorical features in {} seconds', round(end-start, 6))
        else:
            logger.info('Skipped encoding of categorical features')
        return df

    def _to_onehot(self, df, feature, limit=10):  
        # function that encodes categorical features to OneHot encodings    
        one_hot = pd.get_dummies(df[feature], prefix=feature)
        if one_hot.shape[1] > limit:
            logger.warning('ONEHOT encoding for feature "{}" creates {} new features. Consider LABEL encoding instead.', feature, one_hot.shape[1])
        # join the encoded df
        df = df.join(one_hot)
        return df

    def _to_label(self, df, feature):
        # function that encodes categorical features to label encodings 
        le = preprocessing.LabelEncoder()

        df[feature + '_lab'] = le.fit_transform(df[feature].values)
        mapping = dict(zip(le.classes_, range(len(le.classes_))))
        
        for key in mapping:
            try:
                if isnan(key):               
                    replace = {mapping[key] : key }
                    df[feature].replace(replace, inplace=True)
            except:
                pass
        return df