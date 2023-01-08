import pandas as pd
from sklearn.model_selection import train_test_split as tts
from ..req_utils import *
def train_test_split(*arrays,
                     test_size=None,
                     train_size=None,
                     random_state=123,
                     shuffle=False,
                     stratify=None
                     ):
    """Split arrays or matrices into sequential train and test subsets
    Creates train/test splits over endogenous arrays an optional exogenous
    arrays. This is a wrapper of scikit-learn's ``train_test_split`` that
    does not shuffle.
    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float, int or None, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    Examples
    --------
    """
    return tts(
        *arrays,
        random_state=123,
        shuffle=False,
        stratify=None,
        test_size=test_size,
        train_size=train_size)

class models():
    def __init__(self, m_info):
        self.m_info = m_info
        check_scaler(self.m_info.scaler)
        check_M_modelname(self.m_info.modelname,self.m_info.auto)
        if bool(self.m_info.metric):
            evaluate(self.m_info.metric)
    @staticmethod
    def opt(dataFrame : pd.DataFrame, testsize,y,mod):
        X = dataFrame.loc[:, dataFrame.columns != y]
        y = dataFrame[[y]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=123)
        reg = mod(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
        return models,predictions
