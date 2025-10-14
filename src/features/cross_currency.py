import numpy as np
import pandas as pd


class CrossCurrencyFeatures:
    """Compute rolling correlations across currencies."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.currencies = df["currency"].unique()

    def rolling_cross_currency_corr(self, window=15):
        """
        Vectorized version: for each row, add a column for each currency in df['currency'].unique().
        The value in column 'X' is the rolling correlation between 'X' and the row's currency
        at that timestamp.
        """
        if "log_ret" not in self.df.columns:
            self.df["log_ret"] = np.log(self.df["close"] / self.df["close"].shift())

        # df = df.copy()
        self.df["open_time"] = pd.to_datetime(self.df["open_time"])
        currencies = self.df["currency"].unique()
        currency_idx = {c: i for i, c in enumerate(currencies)}

        # Pivot log returns to wide format
        df_pivot: pd.DataFrame = self.df.pivot(
            index="open_time", columns="currency", values="log_ret"
        )
        pivot_index = df_pivot.index

        # Precompute rolling correlations
        n_rows = len(df_pivot)
        n_curr = len(currencies)
        corr_matrix = np.ones((n_rows, n_curr, n_curr), dtype=np.float32)

        for i, curr1 in enumerate(currencies):
            for j, curr2 in enumerate(currencies):
                if i >= j:
                    continue  # lower triangle + diagonal = 1
                rolling_corr = (
                    df_pivot[curr1]
                    .rolling(window)
                    .corr(df_pivot[curr2])
                    .astype(np.float32)
                )
                corr_matrix[:, i, j] = rolling_corr.values
                corr_matrix[:, j, i] = rolling_corr.values  # symmetric

        # Map each df row to pivot row
        pivot_pos = self.df["open_time"].map(lambda x: pivot_index.get_loc(x)).values
        row_currency_idx = self.df["currency"].map(currency_idx).values

        # Build new columns
        for i, col_curr in enumerate(currencies):
            self.df[f"corr_{col_curr}"] = corr_matrix[pivot_pos, row_currency_idx, i]

        return self.df
