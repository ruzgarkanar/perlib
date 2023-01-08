from .req_utils import _periods

class m_info(object):

    """
    Desired parameters for machine learning algorithms  ( ALL machine learnin algorithms )
    """
    def __init__(self,
                 y                      = None,
                 modelname              = "SVR",
                 scaler                 ="minmax",
                 testsize               = None,
                 metric                 = None,
                 modelparams            = None,
                 auto                   = False
                 ):
        self.y                   = y
        self.modelname           = modelname
        self.modelparams         = modelparams
        self.testsize            = testsize
        self.scaler              = scaler
        self.auto                = auto
        self.metric              = metric

class aR_info(object):

    """
    Desired parameters for statistical learning algorithms  ( ARIMA , SARIMA )
    """
    def __init__(self,
                 modelname              = "arima",
                 max_p                  = 5,
                 max_d                  = 2,
                 max_q                  = 5,
                 max_P                  = 2,
                 max_D                  = 2,
                 max_Q                  = 2,
                 s                      = 12,
                 period_                = "daily",
                 forecastNumber          = 24,
                 metric                 = None,
                 forecastingStartDate    = False
                 ):
        _periods(period_)
        self.modelname           = modelname
        self.max_p               = max_p
        self.max_d               = max_d
        self.mad_q               = max_q
        self.max_P               = max_P
        self.max_D               = max_D
        self.max_Q               = max_Q
        self.s                   = s
        self.period              = period_
        self.forecastNumber       = forecastNumber
        self.forecastingStartDate = forecastingStartDate
        self.metric              = metric
        self.max_p = range(0, max_p, 1)
        self.max_d = range(0, max_d, 1)
        self.max_q = range(0, max_q, 1)
        self.max_P = range(0, max_P, 1)
        self.max_D = range(0, max_D, 1)
        self.max_Q = range(0, max_Q, 1)


class req_info(object):

    """
    dict_ = For layers : unit,activation,dropout
    Desired parameters for deep learning algorithms  ( LSTNET,LSTM,BILSTM,CONVLSTM,RNN,TCN )
    """

    dict_ = {"Layer": {"unit": [150, 100,50]
           , "activation": ["tanh", "tanh","tanh"],
             "dropout": [0.2, 0.2,0.2]
                       }}
    def __init__(self,
                 modelname              = "lstm",
                 layers                 =  dict_,
                 lookback               = 24,
                 epoch                  = 2,
                 batch_size             = 200,
                 scaler                 = "standard",
                 period_                = "daily",
                 forecastNumber         = 24,
                 targetCol              = None,
                 forecastingStartDate   = False,
                 metric                 = "mape"
                 ) :

        _periods(period_)
        self.modelname           : str   = modelname
        self.layers              : dict  = layers
        self.lookback            : int   = lookback
        self.epoch               : int   = epoch
        self.batch_size          : int   = batch_size
        self.scaler              : str   = scaler
        self.period              : int   = period_
        self.forecastingStartDate: str   = forecastingStartDate
        self.forecastNumber      : int   = forecastNumber
        self.targetCol           : str   = targetCol
        self.metric              : str   = metric
