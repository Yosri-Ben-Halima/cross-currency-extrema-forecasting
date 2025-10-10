import pandas as pd

from utils.helpers import downcast_numeric_columns
from .returns import ReturnFeatures
from .volatility import VolatilityFeatures
from .microstructure import MicrostructureFeatures
from .technicals import TechnicalFeatures
from .cross_currency import CrossCurrencyFeatures
from .time_features import TimeFeatures


class FeatureCalculator:
    """
    Orchestrates computation of all feature groups:
    returns, volatility, microstructure, technicals, cross-currency, time features.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def compute_all_features(
        self,
        return_window: int = 15,
        vol_window: int = 15,
        micro_window: int = 15,
        # pca_window: int = 15,
        corr_window: int = 15,
        # n_components: int = 3,
    ) -> pd.DataFrame:
        """Compute all features and return final DataFrame."""

        self.df = downcast_numeric_columns(self.df)
        print("🕒 Starting feature computations...")

        # Returns
        print("📊 Computing return figures...")
        self.rf = ReturnFeatures(self.df)
        self.df = self.rf.log_return()
        self.df = self.rf.intraday_cum_return()
        self.df = self.rf.vol_adjusted_return(window=return_window)

        # Volatility
        print("📊 Computing volatility metrics...")
        self.vf = VolatilityFeatures(self.df)
        self.df = self.vf.realized_vol()
        self.df = self.vf.atr(window=vol_window)

        # Microstructure
        print("📊 Computing microstructure features...")
        self.mf = MicrostructureFeatures(self.df)
        self.df = self.mf.high_low_spread()
        self.df = self.mf.close_open_return()
        self.df = self.mf.rel_volume(window=micro_window)

        # Technicals
        print("📊 Computing technical indicators...")
        self.ti = TechnicalFeatures(self.df)
        self.df = self.ti.rsi()
        self.df = self.ti.macd()
        self.df = self.ti.moving_averages()

        # Cross-Currency
        print("📊 Computing cross-currency correlations...")
        self.ccf = CrossCurrencyFeatures(self.df)
        self.df = self.ccf.rolling_cross_currency_corr(window=corr_window)

        self.df = downcast_numeric_columns(self.df)

        # Time Features
        print("📊 Computing time features...")
        self.tf = TimeFeatures(self.df)
        self.df = self.tf.add_cyclic_features()

        print("✅ All features computed.\n")

        return self.df
