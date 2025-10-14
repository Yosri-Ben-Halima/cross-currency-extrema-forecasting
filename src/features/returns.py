import pandas as pd
import numpy as np


class ReturnFeatures:
    """Computes return features: log returns, cumulative intraday returns, vol-adjusted returns."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def log_return(self):
        # currency-wise log returns
        self.df["log_ret"] = self.df.groupby("currency")["close"].transform(
            lambda x: np.log(x / x.shift())
        )
        return self.df

    def intraday_cum_return(self):
        self.df["date"] = self.df["open_time"].dt.date
        self.df["cum_ret"] = self.df.groupby(["currency", "date"])["log_ret"].cumsum()
        self.df.drop("date", axis=1, inplace=True)
        return self.df

    def vol_adjusted_return(self, window=15):
        self.df["rolling_vol"] = self.df.groupby("currency")["log_ret"].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        self.df["vol_adj_ret"] = self.df["log_ret"] / (self.df["rolling_vol"] + 1e-8)
        self.df.drop(columns="rolling_vol", inplace=True)
        return self.df

    def compute_all(self, window=15):
        """Convenience method to compute all features in one call."""
        self.log_return()
        self.intraday_cum_return()
        self.vol_adjusted_return(window)
        return self.df
