import pandas as pd
from typing import Optional, Tuple


class DatasetSplitter:
    """
    Time-aware dataset splitter with De Prado-style embargo.

    The embargo is treated as a fraction of the total dataset, such that:
        train_size + val_size + test_size + embargo_size = 1.0

    Example:
    >>> splitter = DatasetSplitter(train_size=0.7, val_size=0.15, test_size=0.14, embargo_size=0.01)
    >>> train_df, val_df, test_df = splitter.split(df, time_col="open_time")
    """

    def __init__(
        self,
        train_size: Optional[float] = 0.7,
        val_size: Optional[float] = 0.15,
        test_size: Optional[float] = 0.14,
        embargo_size: Optional[float] = 0.01,
    ):
        total = train_size + val_size + test_size + embargo_size
        assert abs(total - 1.0) < 1e-6, f"Fractions must sum to 1.0, got {total:.4f}"
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.embargo_size = embargo_size

    def split(
        self,
        df: pd.DataFrame,
        time_col: Optional[str] = "open_time",
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Splits the DataFrame chronologically into train/val/test with embargo in between.

        Args:
            df (pd.DataFrame): Dataset to split.
            time_col (str, optional): Column to sort chronologically. If None, index order is used.

        Returns:
            tuple: (train_df, val_df, test_df)
        """

        print("🔹 Splitting dataset with embargo...")
        df = df.copy()
        if time_col:
            df = df.sort_values(by=time_col)

        n = len(df)

        train_end = int(self.train_size * n)
        val_end = int((self.train_size + self.embargo_size + self.val_size) * n)
        embargo_len = int(self.embargo_size * n)

        train_df = df.iloc[:train_end]
        val_start = train_end + embargo_len
        val_end = val_start + int(self.val_size * n)
        val_df = df.iloc[val_start:val_end]

        test_start = val_end + embargo_len
        test_df = df.iloc[test_start:]

        print(
            f"✅ Split dataset: {len(train_df):,} train | {len(val_df):,} val | {len(test_df):,} test "
            f"(embargo = {embargo_len:,} rows, ~{self.embargo_size:.2%})"
        )

        return train_df, val_df, test_df
