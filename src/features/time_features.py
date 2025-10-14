import pandas as pd
import numpy as np


class TimeFeatures:
    """Compute cyclical time features for ML models."""

    def __init__(self, df: pd.DataFrame):
        """
        df: must have a datetime column 'open_time'
        """
        self.df = df.copy()

    def compute_all(self):
        """Add sin/cos encoded features for hour, minute, day-of-week."""
        self.df["open_time"] = pd.to_datetime(self.df["open_time"])

        minutes = self.df["open_time"].dt.hour * 60 + self.df["open_time"].dt.minute
        self.df["sin_minute"] = np.sin(2 * np.pi * minutes / 1440)
        self.df["cos_minute"] = np.cos(2 * np.pi * minutes / 1440)

        hours = self.df["open_time"].dt.hour
        self.df["sin_hour"] = np.sin(2 * np.pi * hours / 24)
        self.df["cos_hour"] = np.cos(2 * np.pi * hours / 24)

        return self.df
