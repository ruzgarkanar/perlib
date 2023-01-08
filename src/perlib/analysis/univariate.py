from collections.abc import Iterable
from textwrap import shorten
from typing import Optional, Tuple
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from scipy import stats
from .validate import validate_univariate_input

#Taken as reference............. Development will continue.



class Variable:

    def __init__(self, data: Iterable, *, name: str = None) -> None:
        self.data = validate_univariate_input(data, name=name)
        self.name = self.data.name
        self.var_type = self._get_variable_type()
        self.num_unique = self.data.nunique()
        self.unique = sorted(self.data.dropna().unique())
        self.missing = self._get_missing_values_info()
        self._get_summary_statistics()

    def __repr__(self) -> str:
        return repr(self.summary_statistics)

    def rename(self, name: str = None) -> None:
        self.name = self.data.name = name

    def _get_variable_type(self) -> str:
        if is_numeric_dtype(self.data):
            if is_bool_dtype(self.data) or set(self.data.dropna()) == {0, 1}:
                # Consider boolean data as categorical
                self.data = self.data.astype("category")
                return "boolean"
            elif self.data.nunique() <= 10:
                # Consider numeric data with <= 10 unique values categorical
                self.data = self.data.astype("category").cat.as_ordered()
                return "numeric (<10 levels)"
            else:
                return "numeric"

        elif is_datetime64_any_dtype(self.data):
            return "datetime"

        else:
            self.data = self.data.astype("string")
            if (self.data.nunique() / self.data.shape[0]) <= (1 / 3):
                # If 1/3 or less of the values are unique, use categorical
                self.data = self.data.astype("category")

        return "categorical"

    def _get_missing_values_info(self) -> Optional[str]:
        missing_values = self.data.isna().sum()
        if missing_values == 0:
            return None
        else:
            return (
                f"{missing_values:,} ({missing_values / len(self.data):.2%})"
            )

    def _get_summary_statistics(self) -> None:
        if self.var_type == "numeric":
            stats = _NumericStats(self)
        elif self.var_type == "datetime":
            stats = _DatetimeStats(self)
        else:
            stats = _CategoricalStats(self)

        self.summary_statistics = stats


class _CategoricalStats:
    def __init__(self, variable: Variable) -> None:
        self.variable = variable

    def __repr__(self) -> str:
        sample_values = shorten(
            f"{self.variable.num_unique} -> {self.variable.unique}", 60
        )
        return "\n".join(
            [
                "\t\tOverview",
                "\t\t========",
                f"Name: {self.variable.name}",
                f"Type: {self.variable.var_type}",
                f"Number of Observations: {len(self.variable.data)}",
                f"Unique Values: {sample_values}",
                f"Missing Values: {self.variable.missing}\n",
                "\t  Most Common Items",
                "\t  -----------------",
                f"{self._get_most_common().to_frame(name='')}",
            ]
        )

    def _get_summary_statistics(self) -> pd.Series:
        return self.variable.data.describe().set_axis(
            [
                "Number of observations",
                "Unique values",
                "Mode (Most frequent)",
                "Maximum frequency",
            ],
            axis=0,
        )

    def _get_most_common(self) -> pd.Series:
        most_common_items = self.variable.data.value_counts().head()
        n = len(self.variable.data)
        return most_common_items.apply(lambda x: f"{x:,} ({x / n:.2%})")


class _DatetimeStats:
    def __init__(self, variable: Variable) -> None:
        self.variable = variable

    def __repr__(self) -> str:
        return "\n".join(
            [
                "\t\tOverview",
                "\t\t========",
                f"Name: {self.variable.name}",
                f"Type: {self.variable.var_type}",
                f"Number of Observations: {len(self.variable.data)}",
                f"Missing Values: {self.variable.missing}\n",
                "\t  Summary Statistics",
                "\t  ------------------",
                f"{self._get_summary_statistics().to_frame(name='')}",
            ]
        )

    def _get_summary_statistics(self) -> pd.Series:
        return self.variable.data.describe(datetime_is_numeric=True).set_axis(
            [
                "Number of observations",
                "Average",
                "Minimum",
                "Lower Quartile",
                "Median",
                "Upper Quartile",
                "Maximum",
            ],
            axis=0,
        )


class _NumericStats:
    def __init__(self, variable) -> None:
        self.variable = variable

    def __repr__(self) -> str:
        sample_values = shorten(
            f"{self.variable.num_unique} -> {self.variable.unique}", 60
        )
        return "\n".join(
            [
                "\t\tOverview",
                "\t\t========",
                f"Name: {self.variable.name}",
                f"Type: {self.variable.var_type}",
                f"Unique Values: {sample_values}",
                f"Missing Values: {self.variable.missing}\n",
                "\t  Summary Statistics",
                "\t  ------------------",
                f"{self._get_summary_statistics().to_frame(name='')}\n",
                "\t  Tests for Normality",
                "\t  -------------------",
                f"{self._test_for_normality()}",
            ]
        )

    def _get_summary_statistics(self) -> pd.Series:
        summary_stats = self.variable.data.describe().set_axis(
            [
                "Number of observations",
                "Average",
                "Standard Deviation",
                "Minimum",
                "Lower Quartile",
                "Median",
                "Upper Quartile",
                "Maximum",
            ],
            axis=0,
        )
        summary_stats["Skewness"] = self.variable.data.skew()
        summary_stats["Kurtosis"] = self.variable.data.kurt()

        return summary_stats

    def _test_for_normality(self, alpha: float = 0.05) -> pd.DataFrame:
        data = self.variable.data.dropna()
        # The scikit-learn implementation of the Shapiro-Wilk test reports:
        # "For N > 5000 the W test statistic is accurate but the p-value may
        # not be."
        shapiro_sample = data.sample(5000) if len(data) > 5000 else data
        tests = [
            "D'Agostino's K-squared test",
            "Kolmogorov-Smirnov test",
            "Shapiro-Wilk test",
        ]
        p_values = [
            stats.normaltest(data).pvalue,
            stats.kstest(data, "norm", N=200).pvalue,
            stats.shapiro(shapiro_sample).pvalue,
        ]
        conclusion = f"Conclusion at α = {alpha}"
        results = pd.DataFrame(
            {
                "p-value": p_values,
                conclusion: [p_value > alpha for p_value in p_values],
            },
            index=tests,
        )
        results[conclusion] = results[conclusion].map(
            {
                True: "Possibly normal",
                False: "Unlikely to be normal",
            }
        )
        results["p-value"] = results["p-value"].apply(lambda x: f"{x:.7f}")

        return results


def _analyze_univariate(name_and_data: Tuple) -> Variable:
    name, data = name_and_data
    var = Variable(data, name=name)

    return name, var