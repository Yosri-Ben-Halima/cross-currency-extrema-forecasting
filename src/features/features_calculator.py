import pandas as pd
from features.returns import ReturnFeatures
from features.volatility import VolatilityFeatures
from features.microstructure import MicrostructureFeatures
from features.technicals import TechnicalFeatures
from features.cross_currency import CrossCurrencyFeatures
from features.time_features import TimeFeatures


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
        pca_window: int = 15,
        corr_window: int = 15,
    ) -> pd.DataFrame:
        """Compute all features and return final DataFrame."""

        # Returns
        rf = ReturnFeatures(self.df)
        self.df = rf.log_return()
        self.df = rf.intraday_cum_return()
        self.df = rf.vol_adjusted_return(window=return_window)

        # Volatility
        vf = VolatilityFeatures(self.df)
        self.df = vf.realized_vol()
        self.df = vf.atr(window=vol_window)

        # Microstructure
        mf = MicrostructureFeatures(self.df)
        self.df = mf.high_low_spread()
        self.df = mf.close_open_return()
        self.df = mf.rel_volume(window=micro_window)

        # Technicals
        ti = TechnicalFeatures(self.df)
        self.df = ti.rsi()
        self.df = ti.macd()
        self.df = ti.moving_averages()

        # Cross-Currency
        ccf = CrossCurrencyFeatures(self.df)
        self.df = ccf.rolling_pairwise_correlation(window=corr_window)
        self.df = ccf.pca_factors(window=pca_window, n_components=3)

        # Time Features
        tf = TimeFeatures(self.df)
        self.df = tf.add_cyclic_features()

        return self.df
