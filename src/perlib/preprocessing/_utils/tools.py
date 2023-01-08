import zipfile
import os
import shutil
import tarfile
from keras.utils import io_utils
from copy import deepcopy
from scipy import sparse
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from sklearn.utils.validation import check_is_fitted
import numpy as np
import pandas as pd
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


def get_class_name(class_var: Any) -> str:
    return str(class_var)[8:-2]


def get_package_name(class_var: Any) -> str:
    if not isinstance(str, class_var):
        class_var = get_class_name(class_var)
    return class_var.split(".")[0]

def is_sklearn_pipeline(object):
    from sklearn.pipeline import Pipeline

    return isinstance(object, Pipeline)


def is_sklearn_cv_generator(object):
    return not isinstance(object, str) and hasattr(object, "split")


def is_fitted(estimator) -> bool:
    try:
        check_is_fitted(estimator)
        return True
    except Exception:
        return False


class fit_if_not_fitted(object):
    """
    Context which fits an estimator if it's not fitted.
    """

    def __init__(
        self,
        estimator,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        groups=None,
        **fit_kwargs,
    ):
        #logger = get_logger()
        self.estimator = deepcopy(estimator)
        if not is_fitted(self.estimator):
            if not is_fitted(self.estimator):
                logger.info(f"fit_if_not_fitted: {estimator} is not fitted, fitting")
                try:
                    self.estimator.fit(X_train, y_train, groups=groups, **fit_kwargs)
                except Exception:
                    self.estimator.fit(X_train, y_train, **fit_kwargs)

    def __enter__(self):
        return self.estimator

    def __exit__(self, type, value, traceback):
        return



def _check_custom_transformer(transformer):
    actual_transformer = transformer
    if isinstance(transformer, tuple):
        if len(transformer) != 2:
            raise ValueError("Transformer tuple must have a size of 2.")
        if not isinstance(transformer[0], str):
            raise TypeError("First element of transformer tuple must be a str.")
        actual_transformer = transformer[1]
    if not (
        (
            hasattr(actual_transformer, "fit")
            and hasattr(actual_transformer, "transform")
            and hasattr(actual_transformer, "fit_transform")
        )
        or (
            hasattr(actual_transformer, "fit")
            and hasattr(actual_transformer, "fit_resample")
        )
    ):
        raise TypeError(
            "Transformer must be an object implementing methods 'fit', 'transform' and 'fit_transform'/'fit_resample'."
        )

def normalize_custom_transformers(
    transformers: Union[Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]]
) -> list:
    if isinstance(transformers, dict):
        transformers = list(transformers.items())
    if isinstance(transformers, list):
        for i, x in enumerate(transformers):
            _check_custom_transformer(x)
            if not isinstance(x, tuple):
                transformers[i] = (f"custom_step_{i}", x)
    else:
        _check_custom_transformer(transformers)
        if not isinstance(transformers, tuple):
            transformers = ("custom_step", transformers)
        if is_sklearn_pipeline(transformers[0]):
            return transformers.steps
        transformers = [transformers]
    return transformers


def get_columns_to_stratify_by(
    X: pd.DataFrame, y: pd.DataFrame, stratify: Union[bool, List[str]]
) -> pd.DataFrame:
    if not stratify:
        stratify = None
    else:
        if isinstance(stratify, list):
            data = pd.concat([X, y], axis=1)
            if not all(col in data.columns for col in stratify):
                raise ValueError("Column to stratify by does not exist in the dataset.")
            stratify = data[stratify]
        else:
            stratify = y
    return stratify

def check_features_exist(features: List[str], X: pd.DataFrame):
    """Raise an error if the features are not in the feature dataframe X.
    Parameters
    ----------
    features : List[str]
        List of features to check
    X : pd.DataFrame
        Dataframe of features
    Raises
    ------
    ValueError
        If any feature is not present in the feature dataframe
    """
    missing_features = []
    for fx in features:
        if fx not in X.columns:
            missing_features.append(fx)

    if len(missing_features) != 0:
        raise ValueError(
            f"\n\nColumn(s): {missing_features} not found in the feature dataset!"
            "\nThey are either missing from the features or you have specified "
            "a target column as a feature. Available feature columns are:"
            f"\n{X.columns.to_list()}"
        )

def check_cat(X,categorical_features=None, ignore_features=None):
    _fxs = pd.DataFrame()
    _fxs["Ignore"] = ignore_features or []
    if categorical_features:
        check_features_exist(categorical_features, X)
        _fxs["Categorical"] = categorical_features
    else:
        # Default should exclude datetime and text columns
        _fxs["Categorical"] = [
            col
            for col in X.select_dtypes(include=["object", "category"]).columns
            if col not in _fxs["Date"] + _fxs["Text"]
        ]
    return _fxs

def to_series(data, index=None, name="target"):
    """Convert a column to pd.Series.
    Parameters
    ----------
    data: sequence or None
        Data to convert. If None, return unchanged.
    index: sequence or Index, optional (default=None)
        Values for the indices.
    name: string, optional (default="target")
        Name of the target column.
    Returns
    -------
    series: pd.Series or None
        Transformed series.
    """
    if data is not None and not isinstance(data, pd.Series):
        if isinstance(data, pd.DataFrame):
            try:
                data = data[name]
            except Exception:
                data = data.squeeze()
        elif isinstance(data, np.ndarray) and data.ndim > 1:
            data = data.flatten()
        data = pd.Series(data, index=index, name=name)

    return data

def variable_return(X, y):
    """Return one or two arguments depending on which is None."""
    if y is None:
        return X
    elif X is None:
        return y
    else:
        return X, y


def df_shrink_dtypes(df, skip=[], obj2cat=True, int2uint=False):
    """Shrink a dataframe.
    Return any possible smaller data types for DataFrame columns.
    Allows `object`->`category`, `int`->`uint`, and exclusion.
    From: https://github.com/fastai/fastai/blob/master/fastai/tabular/core.py
    """

    # 1: Build column filter and typemap
    excl_types, skip = {"category", "datetime64[ns]", "bool"}, set(skip)

    # no int16 as orjson in plotly doesn't support it
    typemap = {
        "int": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.int8, np.int32, np.int64)
        ],
        "uint": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.uint8, np.uint32, np.uint64)
        ],
        "float": [
            (np.dtype(x), np.finfo(x).min, np.finfo(x).max)
            for x in (np.float32, np.float64, np.longdouble)
        ],
    }

    if obj2cat:
        # User wants to categorify dtype('Object'), which may not always save space
        typemap["object"] = "category"
    else:
        excl_types.add("object")

    new_dtypes = {}
    exclude = lambda dt: dt[1].name not in excl_types and dt[0] not in skip

    for c, old_t in filter(exclude, df.dtypes.items()):
        t = next((v for k, v in typemap.items() if old_t.name.startswith(k)), None)

        if isinstance(t, list):  # Find the smallest type that fits
            if int2uint and t == typemap["int"] and df[c].min() >= 0:
                t = typemap["uint"]
            new_t = next(
                (r[0] for r in t if r[1] <= df[c].min() and r[2] >= df[c].max()), None
            )
            if new_t and new_t == old_t:
                new_t = None
        else:
            new_t = t if isinstance(t, str) else None

        if new_t:
            new_dtypes[c] = new_t

    return df.astype(new_dtypes)


def to_df(data, index=None, columns=None, dtypes=None):
    """Convert a dataset to pd.Dataframe.
    Parameters
    ----------
    data: list, tuple, dict, np.array, sp.matrix, pd.DataFrame or None
        Dataset to convert to a dataframe.  If None or already a
        dataframe, return unchanged.
    index: sequence or pd.Index
        Values for the dataframe's index.
    columns: sequence or None, optional (default=None)
        Name of the columns. Use None for automatic naming.
    dtypes: str, dict, dtype or None, optional (default=None)
        Data types for the output columns. If None, the types are
        inferred from the data.
    Returns
    -------
    df: pd.DataFrame or None
        Transformed dataframe.
    """
    # Get number of columns (list/tuple have no shape and sp.matrix has no index)
    n_cols = lambda data: data.shape[1] if hasattr(data, "shape") else len(data[0])

    if data is not None:
        if not isinstance(data, pd.DataFrame):
            # Assign default column names (dict already has column names)
            if not isinstance(data, dict) and columns is None:
                columns = [f"feature_{str(i)}" for i in range(1, n_cols(data) + 1)]

            # Create dataframe from sparse matrix or directly from data
            if sparse.issparse(data):
                data = pd.DataFrame.sparse.from_spmatrix(data, index, columns)
            else:
                data = pd.DataFrame(data, index, columns)

            if dtypes is not None:
                data = data.astype(dtypes)

        else:
            # Convert all column names to str
            data.columns = [str(col) for col in data.columns]

    return data



def extract_archive(file_path, path=".", archive_format="auto"):

    if archive_format is None:
        return False
    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    file_path = io_utils.path_to_string(file_path)
    path = io_utils.path_to_string(path)

    for archive_type in archive_format:
        if archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile
        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    archive.extractall(path)
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


import re
def check_datetime(value: str):
    if re.search(
            r"^[A-z\d +/|.|[1-9-{0-31}]+(((0?[1-9]|1[012])(:[0-5]\d){0 2}(x20[AP]M))|([01]\d|2[0-3])(:[0-5]\d){1,2})",
            value) or \
            re.search(
                "^([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])$|^([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])$",
                value) or \
            re.search(r"^(((0?[1-9]|1[012])(:[0-5]\d){0,2}(\x20[AP]M))|([01]\d|2[0-3])(:[0-5]\d){1,2})", value):
        return True