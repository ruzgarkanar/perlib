import logging
from collections.abc import Iterable
from typing import Optional, Union
import pandas as pd
from pandas.api.types import is_numeric_dtype
from .exceptions import (
    EmptyDataError,
    GroupbyVariableError,
    InputError,
)


#Taken as reference............. Development will continue.



def clean_column_labels(data: pd.DataFrame) -> pd.DataFrame:
    if isinstance(data.columns, pd.RangeIndex):
        data.columns = [f"var_{i+1}" for i in data.columns]
    elif is_numeric_dtype(data.columns):
        data.columns = [f"var_{i}" for i in data.columns]
        return data
    else:
        data.columns = data.columns.map(str)
    return data


def check_cardinality(groupby_data: pd.Series, *, threshold: int = 10) -> None:

    if groupby_data.nunique() > threshold:
        message = (
            f"Group-by variable '{groupby_data.name}' not used to group "
            f"values. It has high cardinality ({groupby_data.nunique()}) "
            f"and would clutter graphs."
        )
        logging.warning(message)
        raise GroupbyVariableError(message)

def validate_multivariate_input(data: Iterable) -> pd.DataFrame:

    try:
        data_frame = pd.DataFrame(data)
    except Exception:
        raise InputError(
            f"Expected a pandas.Dataframe object, but got {type(data)}."
        )
    # The data should not be empty
    if len(data_frame) == 0:
        raise EmptyDataError("No data to process.")

    data_frame = (
        data_frame.infer_objects()
        .dropna(axis=1, how="all")
    )
    return clean_column_labels(data_frame)

def validate_univariate_input(
    data: Iterable, *, name: str = None
) -> Optional[pd.Series]:
    if data is None:
        return None
    else:
        try:
            series = pd.Series(data, name=name)
        except Exception:
            raise InputError(
                f"Expected a one-dimensional sequence, but got {type(data)}."
            )
    if series.shape[0] == 0:
        raise EmptyDataError("No data to process.")
    else:
        return series

def validate_groupby_data(
    *, data: pd.DataFrame, groupby_data: Union[int, str]
) -> Optional[pd.Series]:
    if groupby_data is None:
        return None
    elif f"{groupby_data}".isdecimal():
        idx = int(groupby_data)
        try:
            groupby_data = data.iloc[:, idx]
        except IndexError:
            raise GroupbyVariableError(
                f"Column index {groupby_data} is not in the range"
                f" [0, {data.columns.size}]."
            )
        check_cardinality(groupby_data)
        return groupby_data
    elif isinstance(groupby_data, str):
        try:
            groupby_data = data[groupby_data]
        except KeyError:
            raise GroupbyVariableError(
                f"{groupby_data!r} is not in {data.columns.to_list()}"
            )
        check_cardinality(groupby_data)
        return groupby_data
    else:
        logging.warning(
            f"Group-by variable '{groupby_data}' ignored."
            " Not a valid column index or label."
        )
        return None