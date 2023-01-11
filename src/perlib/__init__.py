__version__ = "2.0.7"
from .forecaster import *
from .core._requests import req_info,aR_info,m_info
from .core.models.dmodels import models
from .core.models.smodels import models as armodels
from .core.train import dTrain,sTrain
from .core.tester import dTester,sTester
from .core.req_utils import *
from .core.metrics.regression import __ALL__
from .preprocessing.preparate import dataPrepration as pr
from .preprocessing.autopreprocess import Process
from .preprocessing._utils.dataframe import read_pandas
from .preprocessing._utils.tools import extract_archive
from .analysis.multivariate import MultiVariable
from .datasets import *
from .analysis import *
from .core import *
from .piplines import *
from .preprocessing import *
