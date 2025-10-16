import pandas as pd
from utils.helpers import (
    align_dataset,
    downcast_numeric_columns,
    downsample_df_to_15min,
)
from .returns import ReturnFeatures
from .volatility import VolatilityFeatures
from .microstructure import MicrostructureFeatures
from .technicals import TechnicalFeatures
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
        micro_window: int = 15,
    ) -> pd.DataFrame:
        """Compute all features and return final DataFrame."""

        self.df = downcast_numeric_columns(self.df)
        print("🕒 Starting feature computations...")

        # Returns
        print("📊 Computing return figures...")
        self.rf = ReturnFeatures(self.df)
        self.df = self.rf.compute_all(window=return_window)

        # Volatility
        print("📊 Computing volatility metrics...")
        self.vf = VolatilityFeatures(self.df)
        self.df = self.vf.compute_all()

        # Microstructure
        print("📊 Computing microstructure features...")
        self.mf = MicrostructureFeatures(self.df)
        self.df = self.mf.compute_all(window=micro_window)

        # Technicals
        print("📊 Computing technical indicators...")
        self.ti = TechnicalFeatures(self.df)
        self.df = self.ti.compute_all()

        self.df = downcast_numeric_columns(self.df)

        # Time Features
        print("📊 Computing time features...")
        self.tf = TimeFeatures(self.df)
        self.df = self.tf.compute_all()

        print("📊 Aligning & downsampling to 15mins intervals...")
        self.df = downsample_df_to_15min(align_dataset(self.df))

        print("✅ All features computed.\n")

        return self.df
