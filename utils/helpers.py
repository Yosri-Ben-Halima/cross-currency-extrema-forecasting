import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np


def downsample_df_to_15min(df: pd.DataFrame) -> pd.DataFrame:
    min_interval = 15
    df = df.copy()
    n_base = len(df)
    df = df[df["open_time"].dt.minute % min_interval == 0]
    n_new = len(df)
    print(
        f"✅ Downsampled dataset from {n_base:,} rows to {n_new:,} rows using {min_interval}-minute intervals."
    )
    return df


def check_stationarity(df: pd.DataFrame, significance: float = 0.05):
    """
    Runs Augmented Dickey-Fuller (ADF) test for all numeric columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    significance : float, optional
        Significance level to determine stationarity (default: 0.05).

    Returns
    -------
    dict
        Dictionary where keys are column names and values are booleans:
        True if stationary (reject null of unit root), False otherwise.
    """
    results_list = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for curr in df["currency"].unique():
        results = {}
        print(f"Running ADF test for {curr}...")
        for col in numeric_cols:
            df_curr = df[df["currency"] == curr]
            series = df_curr[col].dropna()
            if len(series) < 10:
                results[col] = None
                continue
            try:
                adf_result = adfuller(series, autolag="AIC")
                p_value = adf_result[1]
                results[col] = p_value < significance
            except Exception as e:
                results[col] = None
                print(
                    f"⚠️ ADF test failed for column '{col}' for currency '{curr}': {e}"
                )
            results["currency"] = curr
        results_list.append(results)

    return pd.DataFrame(data=results_list)


def downcast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(
        lambda col: pd.to_numeric(col, downcast="float")
        if col.dtype.kind == "f"
        else pd.to_numeric(col, downcast="integer")
        if col.dtype.kind in "iu"
        else col
    )


def stationarize_by_currency(df: pd.DataFrame, cols=["y_high", "y_low"]):
    """
    Makes selected columns stationary by differencing within each currency group.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a 'currency' column and target columns.
    cols : list of str
        Columns to stationarize (default: ["y_high", "y_low"]).

    Returns
    -------
    pd.DataFrame
        Copy of df with stationarized columns.
    """
    df = df.copy()
    for col in cols:
        df["D_" + col] = (
            df.sort_values(["currency", "open_time"])
            .groupby("currency")[col]
            .diff()
            .fillna(0)
        )

    return df


def standard_scale_by_currency(df: pd.DataFrame, cols=["y_high", "y_low"]):
    """
    Standard scales selected columns per currency, preserving time order,
    and returns per-currency mean and std for inverse transformation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing 'currency' and 'open_time' columns.
    cols : list of str
        Columns to standard scale (default: ["y_high", "y_low"]).

    Returns
    -------
    df_scaled : pd.DataFrame
        Copy of df with standardized columns.
    stats : dict
        Dictionary of the form {currency: {col: (mean, std)}} for each column.
    """
    df_scaled = df.copy()
    stats = {}

    for currency, group in df_scaled.groupby("currency"):
        stats[currency] = {}
        for col in cols:
            # Sort by time
            x = group.sort_values("open_time")[col]
            mean = x.mean()
            std = x.std()
            stats[currency][col] = (mean, std)
            # Standardize
            df_scaled.loc[group.index, col] = (x - mean) / std

    return df_scaled, stats


def align_dataset(a: pd.DataFrame) -> pd.DataFrame:
    min_dates, max_dates = [], []
    for _, df_curr in a.groupby("currency"):
        min_dates.append(df_curr["open_time"].values[0])
        max_dates.append(df_curr["open_time"].values[-1])

    # print(max(min_dates), " -> ", min(max_dates))

    a = a.loc[(a["open_time"] >= max(min_dates)) & (a["open_time"] <= min(max_dates))]

    return a
