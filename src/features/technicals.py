from typing import List
import pandas as pd


class TechnicalFeatures:
    """Compute technical indicators: RSI, MACD, MAs."""

    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df.copy()

    def rsi(self, window: int = 14):
        """Relative Strength Index (RSI)."""

        def _rsi(x: pd.DataFrame):
            delta = x.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window, min_periods=1).mean()
            avg_loss = loss.rolling(window, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            return 100 - (100 / (1 + rs))

        self.df["rsi"] = self.df.groupby("currency")["close"].transform(_rsi)
        return self.df

    def macd(self, short_window=12, long_window=26, signal_window=9):
        """Moving Average Convergence Divergence (MACD)."""

        def _ema(x: pd.DataFrame, span):
            return x.ewm(span=span, adjust=False).mean()

        self.df["ema_short"] = self.df.groupby("currency")["close"].transform(
            lambda x: _ema(x, short_window)
        )
        self.df["ema_long"] = self.df.groupby("currency")["close"].transform(
            lambda x: _ema(x, long_window)
        )
        self.df["macd"] = self.df["ema_short"] - self.df["ema_long"]
        self.df["macd_signal"] = self.df.groupby("currency")["macd"].transform(
            lambda x: _ema(x, signal_window)
        )
        self.df["macd_hist"] = self.df["macd"] - self.df["macd_signal"]
        self.df.drop(
            columns=["macd", "ema_long", "ema_short", "macd_signal"], inplace=True
        )
        return self.df

    def moving_averages(self, windows: List[int] = [60]):
        """Simple moving averages."""
        for w in windows:
            self.df[f"sma_{w}"] = self.df.groupby("currency")["close"].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
        return self.df

    def compute_all(self):
        """Convenience wrapper."""
        self.rsi()
        self.macd()
        self.moving_averages()
        return self.df
