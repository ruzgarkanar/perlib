from timeit import default_timer as timer
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class Outliers:

    def handle(self, df):
        # function for handling of outliers in the data
        if self.outliers:
            logger.info('Started handling of outliers... Method: "{}"', str(self.outliers).upper())
            start = timer()  

            if self.outliers in ['auto', 'winz']:  
                df = Outliers._winsorization(self, df)
            elif self.outliers == 'delete':
                df = Outliers._delete(self, df)
            
            end = timer()
            logger.info('Completed handling of outliers in {} seconds', round(end-start, 6))
        else:
            logger.info('Skipped handling of outliers')
        return df     

    def _winsorization(self, df):
        # function for outlier winsorization
        cols_num = df.select_dtypes(include=np.number).columns    
        for feature in cols_num:           
            counter = 0
            # compute outlier bounds
            lower_bound, upper_bound = Outliers._compute_bounds(self, df, feature)    
            for row_index, row_val in enumerate(df[feature]):
                if row_val < lower_bound or row_val > upper_bound:
                    if row_val < lower_bound:
                        if (df[feature].fillna(-9999) % 1  == 0).all():
                                df.loc[row_index, feature] = lower_bound
                                df[feature] = df[feature].astype(int) 
                        else:    
                            df.loc[row_index, feature] = lower_bound
                        counter += 1
                    else:
                        if (df[feature].fillna(-9999) % 1  == 0).all():
                            df.loc[row_index, feature] = upper_bound
                            df[feature] = df[feature].astype(int) 
                        else:
                            df.loc[row_index, feature] = upper_bound
                        counter += 1
            if counter != 0:
                logger.debug('Outlier imputation of {} value(s) succeeded for feature "{}"', counter, feature)        
        return df

    def _delete(self, df):
        # function for deleting outliers in the data
        cols_num = df.select_dtypes(include=np.number).columns    
        for feature in cols_num:
            counter = 0
            lower_bound, upper_bound = Outliers._compute_bounds(self, df, feature)    
            # delete observations containing outliers            
            for row_index, row_val in enumerate(df[feature]):
                if row_val < lower_bound or row_val > upper_bound:
                    df = df.drop(row_index)
                    counter +=1
            df = df.reset_index(drop=True)
            if counter != 0:
                logger.debug('Deletion of {} outliers succeeded for feature "{}"', counter, feature)
        return df

    def _compute_bounds(self, df, feature):
        # function that computes the lower and upper bounds for finding outliers in the data
        featureSorted = sorted(df[feature])
        
        q1, q3 = np.percentile(featureSorted, [25, 75])
        iqr = q3 - q1

        lb = q1 - (self.outlier_param * iqr) 
        ub = q3 + (self.outlier_param * iqr) 

        return lb, ub    