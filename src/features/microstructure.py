import pandas as pd
import numpy as np


class MicrostructureFeatures:
    """OHLCV-based microstructure features :High-Low spread, Close-Open return, relative volume."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.sort_values(["currency", "open_time"], inplace=True)

    def high_low_spread(self):
        self.df["hl_spread"] = self.df["high"] - self.df["low"]
        return self.df

    def close_open_return(self):
        self.df["co_ret"] = np.log(self.df["close"] / self.df["open"])
        return self.df

    def rel_volume(self, window=15):
        self.df["rel_vol"] = self.df.groupby("currency")["volume"].transform(
            lambda x: x / (x.rolling(window, min_periods=1).mean() + 1e-8)
        )
        return self.df

    def compute_all(self, window: int = 15):
        self.high_low_spread()
        self.close_open_return()
        self.rel_volume(window)
        return self.df
