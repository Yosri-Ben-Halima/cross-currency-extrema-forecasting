from typing import List
import pandas as pd


class VolatilityFeatures:
    """Compute volatility and risk features currency-wise."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def realized_vol(
        self,
        return_col: str = "log_ret",
        windows: List[int] = [5, 15, 30],
    ):
        """Rolling standard deviation of returns (realized volatility)."""
        for w in windows:
            self.df[f"rv_{w}"] = self.df.groupby("currency")[return_col].transform(
                lambda x: x.rolling(w, min_periods=1).std()
            )
        return self.df

    def atr(self, window: int = 14):
        """Average True Range (ATR) for range-based volatility."""
        high_low = self.df["high"] - self.df["low"]
        high_close = (self.df["high"] - self.df["close"].shift()).abs()
        low_close = (self.df["low"] - self.df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # ATR computed currency-wise
        self.df["atr"] = self.df.groupby("currency")[
            tr.name if tr.name else "close"
        ].transform(lambda x: tr.rolling(window, min_periods=1).mean())
        return self.df

    def high_low_range(self):
        """Simple high-low range."""
        self.df["hl_range"] = self.df["high"] - self.df["low"]
        return self.df

    def volatility_regime(self, window: int = 15, threshold: float = 1.0):
        """Flag high vs low volatility regime based on rolling realized vol."""
        rv = self.df.groupby("currency")["log_ret"].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        self.df["vol_regime"] = (rv > threshold).astype(int)
        return self.df

    def compute_all(
        self,
        rv_windows: List[int] = [5, 15, 30],
        atr_window: int = 14,
        vol_reg_window: int = 15,
        vol_thresh: float = 1.0,
    ):
        """Compute all volatility features in one call."""
        self.realized_vol(windows=rv_windows)
        self.atr(window=atr_window)
        self.high_low_range()
        self.volatility_regime(window=vol_reg_window, threshold=vol_thresh)
        return self.df
