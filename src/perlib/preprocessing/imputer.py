from timeit import default_timer as timer
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

'''
Modules are used by the AutoProcess pipeline for data cleaning and preprocessing.
'''

class MissingValues:

    def handle(self, df, _n_neighbors=3):
        # function for handling missing values in the data
        if self.missing_num or self.missing_categ:
            logger.info('Started handling of missing values...', str(self.missing_num).upper())
            start = timer()
            self.count_missing = df.isna().sum().sum()

            if self.count_missing != 0:
                logger.info('Found a total of {} missing value(s)', self.count_missing)
                df = df.dropna(how='all')
                df.reset_index(drop=True)
                
                if self.missing_num: # numeric data
                    logger.info('Started handling of NUMERICAL missing values... Method: "{}"', str(self.missing_num).upper())
                    # automated handling
                    if self.missing_num == 'auto': 
                        self.missing_num = 'linreg'
                        lr = LinearRegression()
                        df = MissingValues._lin_regression_impute(self, df, lr)
                        self.missing_num = 'knn'
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='num')
                    # linear regression imputation
                    elif self.missing_num == 'linreg':
                        lr = LinearRegression()
                        df = MissingValues._lin_regression_impute(self, df, lr)
                    # knn imputation
                    elif self.missing_num == 'knn':
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='num')
                    # mean, median or mode imputation
                    elif self.missing_num in ['mean', 'median', 'most_frequent']:
                        imputer = SimpleImputer(strategy=self.missing_num)
                        df = MissingValues._impute(self, df, imputer, type='num')
                    # delete missing values
                    elif self.missing_num == 'delete':
                        df = MissingValues._delete(self, df, type='num')
                        logger.debug('Deletion of {} NUMERIC missing value(s) succeeded', self.count_missing-df.isna().sum().sum())      

                if self.missing_categ: # categorical data
                    logger.info('Started handling of CATEGORICAL missing values... Method: "{}"', str(self.missing_categ).upper())
                    # automated handling
                    if self.missing_categ == 'auto':
                        self.missing_categ = 'logreg'
                        lr = LogisticRegression()
                        df = MissingValues._log_regression_impute(self, df, lr)
                        self.missing_categ = 'knn'
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='categ')
                    elif self.missing_categ == 'logreg':
                        lr = LogisticRegression()
                        df = MissingValues._log_regression_impute(self, df, lr)
                    # knn imputation
                    elif self.missing_categ == 'knn':
                        imputer = KNNImputer(n_neighbors=_n_neighbors)
                        df = MissingValues._impute(self, df, imputer, type='categ')  
                    # mode imputation
                    elif self.missing_categ == 'most_frequent':
                        imputer = SimpleImputer(strategy=self.missing_categ)
                        df = MissingValues._impute(self, df, imputer, type='categ')
                    # delete missing values                    
                    elif self.missing_categ == 'delete':
                        df = MissingValues._delete(self, df, type='categ')
                        logger.debug('Deletion of {} CATEGORICAL missing value(s) succeeded', self.count_missing-df.isna().sum().sum())
            else:
                logger.debug('{} missing values found', self.count_missing)
            end = timer()
            logger.info('Completed handling of missing values in {} seconds', round(end-start, 6))  
        else:
            logger.info('Skipped handling of missing values')
        return df

    def _impute(self, df, imputer, type):
        # function for imputing missing values in the data
        cols_num = df.select_dtypes(include=np.number).columns 

        if type == 'num':
            # numerical features
            for feature in df.columns: 
                if feature in cols_num:
                    if df[feature].isna().sum().sum() != 0:
                        try:
                            df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)))
                            counter = df[feature].isna().sum().sum() - df_imputed.isna().sum().sum()

                            if (df[feature].fillna(-9999) % 1  == 0).all():
                                df[feature] = df_imputed
                                # round back to INTs, if original data were INTs
                                df[feature] = df[feature].round()
                                df[feature] = df[feature].astype('Int64')                                        
                            else:
                                df[feature] = df_imputed
                            if counter != 0:
                                logger.debug('{} imputation of {} value(s) succeeded for feature "{}"', str(self.missing_num).upper(), counter, feature)
                        except:
                            logger.warning('{} imputation failed for feature "{}"', str(self.missing_num).upper(), feature)
        else:
            # categorical features
            for feature in df.columns:
                if feature not in cols_num:
                    if df[feature].isna().sum()!= 0:
                        try:
                            mapping = dict()
                            mappings = {k: i for i, k in enumerate(df[feature].dropna().unique(), 0)}
                            mapping[feature] = mappings
                            df[feature] = df[feature].map(mapping[feature])

                            df_imputed = pd.DataFrame(imputer.fit_transform(np.array(df[feature]).reshape(-1, 1)), columns=[feature])    
                            counter = sum(1 for i, j in zip(list(df_imputed[feature]), list(df[feature])) if i != j)

                            # round to integers before mapping back to original values
                            df[feature] = df_imputed
                            df[feature] = df[feature].round()
                            df[feature] = df[feature].astype('Int64')  

                            # map values back to original
                            mappings_inv = {v: k for k, v in mapping[feature].items()}
                            df[feature] = df[feature].map(mappings_inv)
                            if counter != 0:
                                logger.debug('{} imputation of {} value(s) succeeded for feature "{}"', self.missing_categ.upper(), counter, feature)
                        except:
                            logger.warning('{} imputation failed for feature "{}"', str(self.missing_categ).upper(), feature)
        return df

    def _lin_regression_impute(self, df, model):
        # function for predicting missing values with linear regression
        cols_num = df.select_dtypes(include=np.number).columns
        mapping = dict()
        for feature in df.columns:
            if feature not in cols_num:
                # create label mapping for categorical feature values
                mappings = {k: i for i, k in enumerate(df[feature])}
                mapping[feature] = mappings
                df[feature] = df[feature].map(mapping[feature])
        for feature in cols_num: 
                try:
                    test_df = df[df[feature].isnull()==True].dropna(subset=[x for x in df.columns if x != feature])
                    train_df = df[df[feature].isnull()==False].dropna(subset=[x for x in df.columns if x != feature])
                    if len(test_df.index) != 0:
                        pipe = make_pipeline(StandardScaler(), model)

                        y = np.log(train_df[feature]) # log-transform the data
                        X_train = train_df.drop(feature, axis=1)
                        test_df.drop(feature, axis=1, inplace=True)
                        
                        try:
                            model = pipe.fit(X_train, y)
                        except:
                            y = train_df[feature] # use non-log-transformed data
                            model = pipe.fit(X_train, y)
                        if (y == train_df[feature]).all():
                            pred = model.predict(test_df)
                        else:
                            pred = np.exp(model.predict(test_df)) # predict values

                        test_df[feature]= pred

                        if (df[feature].fillna(-9999) % 1  == 0).all():
                            # round back to INTs, if original data were INTs
                            test_df[feature] = test_df[feature].round()
                            test_df[feature] = test_df[feature].astype('Int64')
                            df[feature].update(test_df[feature])                          
                        else:
                            df[feature].update(test_df[feature])  
                        logger.debug('LINREG imputation of {} value(s) succeeded for feature "{}"', len(pred), feature)
                except:
                    logger.warning('LINREG imputation failed for feature "{}"', feature)
        for feature in df.columns: 
            try:   
                # map categorical feature values back to original
                mappings_inv = {v: k for k, v in mapping[feature].items()}
                df[feature] = df[feature].map(mappings_inv)
            except:
                pass
        return df

    def _log_regression_impute(self, df, model):
        # function for predicting missing values with logistic regression
        cols_num = df.select_dtypes(include=np.number).columns
        mapping = dict()
        for feature in df.columns:
            if feature not in cols_num:
                # create label mapping for categorical feature values
                mappings = {k: i for i, k in enumerate(df[feature])} #.dropna().unique(), 0)}
                mapping[feature] = mappings
                df[feature] = df[feature].map(mapping[feature])

        target_cols = [x for x in df.columns if x not in cols_num]
            
        for feature in df.columns: 
            if feature in target_cols:
                try:
                    test_df = df[df[feature].isnull()==True].dropna(subset=[x for x in df.columns if x != feature])
                    train_df = df[df[feature].isnull()==False].dropna(subset=[x for x in df.columns if x != feature])
                    if len(test_df.index) != 0:
                        pipe = make_pipeline(StandardScaler(), model)

                        y = train_df[feature]
                        train_df.drop(feature, axis=1, inplace=True)
                        test_df.drop(feature, axis=1, inplace=True)

                        model = pipe.fit(train_df, y)
                        
                        pred = model.predict(test_df) # predict values
                        test_df[feature]= pred

                        if (df[feature].fillna(-9999) % 1  == 0).all():
                            # round back to INTs, if original data were INTs
                            test_df[feature] = test_df[feature].round()
                            test_df[feature] = test_df[feature].astype('Int64')
                            df[feature].update(test_df[feature])                             
                        logger.debug('LOGREG imputation of {} value(s) succeeded for feature "{}"', len(pred), feature)
                except:
                    logger.warning('LOGREG imputation failed for feature "{}"', feature)
        for feature in df.columns: 
            try:
                # map categorical feature values back to original
                mappings_inv = {v: k for k, v in mapping[feature].items()}
                df[feature] = df[feature].map(mappings_inv)
            except:
                pass     
        return df

    def _delete(self, df, type):
        # function for deleting missing values
        cols_num = df.select_dtypes(include=np.number).columns 
        if type == 'num':
            # numerical features
            for feature in df.columns: 
                if feature in cols_num:
                    df = df.dropna(subset=[feature])
                    df.reset_index(drop=True)
        else:
            # categorical features
            for feature in df.columns:
                if feature not in cols_num:
                    df = df.dropna(subset=[feature])
                    df.reset_index(drop=True)
        return df       