DTYPE = "float64"

from .airpassengers import *
from .ausbeer import *
from .austres import *
from .heartrate import *
from .lynx import *
from .taylor import *
from .wineind import *
from .woolyrnq import *

__all__ = [s for s in dir() if not s.startswith("_")]