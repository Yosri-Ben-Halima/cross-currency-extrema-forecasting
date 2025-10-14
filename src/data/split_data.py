import pandas as pd
from typing import Optional, Tuple


class DatasetSplitter:
    """
    Time-aware dataset splitter per-currency with embargos.
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

    def _split_single_currency(
        self, df: pd.DataFrame, time_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.sort_values(by=time_col).reset_index(drop=True)
        n = len(df)
        embargo_len = int(self.embargo_size * n)

        # Calculate split indices
        train_end = int(self.train_size * n)
        val_start = train_end + embargo_len
        val_end = val_start + int(self.val_size * n)
        test_start = val_end + embargo_len

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[val_start:val_end].copy()
        test_df = df.iloc[test_start:].copy()

        return train_df, val_df, test_df

    def split(
        self,
        df: pd.DataFrame,
        time_col: Optional[str] = "open_time",
        currency_col: Optional[str] = "currency",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits the DataFrame chronologically into train/val/test per currency with embargo in between.

        Args:
            df (pd.DataFrame): Dataset to split.
            time_col (str, optional): Column to sort chronologically.
            currency_col (str, optional): Column indicating currency grouping.

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        print("🔹 Splitting dataset by currency with embargo...")

        # Split per currency
        train_parts, val_parts, test_parts = [], [], []

        for cur, sub_df in df.groupby(currency_col):
            train_df, val_df, test_df = self._split_single_currency(sub_df, time_col)
            train_parts.append(train_df)
            val_parts.append(val_df)
            test_parts.append(test_df)

        # Concatenate all splits
        train_df = (
            pd.concat(train_parts)
            .sort_values(by=[currency_col, time_col])
            .reset_index(drop=True)
        )
        val_df = (
            pd.concat(val_parts)
            .sort_values(by=[currency_col, time_col])
            .reset_index(drop=True)
        )
        test_df = (
            pd.concat(test_parts)
            .sort_values(by=[currency_col, time_col])
            .reset_index(drop=True)
        )

        # Logging
        # embargo_example = int(
        #     self.embargo_size * len(df.groupby(currency_col).__next__()[1])
        # )
        print(
            f"✅ Split dataset: {len(train_df):,} train | {len(val_df):,} val | {len(test_df):,} test "
            f"(embargo ~{self.embargo_size:.2%} per currency)"
        )

        return train_df, val_df, test_df
