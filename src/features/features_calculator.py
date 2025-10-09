import pandas as pd
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

        self.df = self.downcast_numeric_columns()
        print("🕒 Starting feature computations...")

        # Returns
        print("📊 Computing return features...")
        rf = ReturnFeatures(self.df)
        self.df = rf.log_return()
        self.df = rf.intraday_cum_return()
        self.df = rf.vol_adjusted_return(window=return_window)

        # Volatility
        print("📊 Computing volatility features...")
        vf = VolatilityFeatures(self.df)
        self.df = vf.realized_vol()
        self.df = vf.atr(window=vol_window)

        # Microstructure
        print("📊 Computing microstructure features...")
        mf = MicrostructureFeatures(self.df)
        self.df = mf.high_low_spread()
        self.df = mf.close_open_return()
        self.df = mf.rel_volume(window=micro_window)

        # Technicals
        print("📊 Computing technical features...")
        ti = TechnicalFeatures(self.df)
        self.df = ti.rsi()
        self.df = ti.macd()
        self.df = ti.moving_averages()

        # Cross-Currency
        print("📊 Computing cross-currency features...")
        ccf = CrossCurrencyFeatures(self.df)
        self.df = ccf.rolling_pairwise_correlation(window=corr_window)

        self.df = self.downcast_numeric_columns()

        # Time Features
        print("📊 Computing time features...")
        tf = TimeFeatures(self.df)
        self.df = tf.add_cyclic_features()

        print("✅ All features computed.\n")

        return self.df

    def downcast_numeric_columns(self):
        return self.df.apply(
            lambda col: pd.to_numeric(col, downcast="float")
            if col.dtype.kind == "f"
            else pd.to_numeric(col, downcast="integer")
            if col.dtype.kind in "iu"
            else col
        )
