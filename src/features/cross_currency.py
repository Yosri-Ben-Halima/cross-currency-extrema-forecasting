import pandas as pd
from itertools import combinations
from sklearn.decomposition import PCA


class CrossCurrencyFeatures:
    """Compute rolling correlations and PCA factors across currencies."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.currencies = df["currency"].unique()

    def rolling_pairwise_correlation(self, window: int = 15):
        """
        Compute rolling correlations for all currency pairs and add them to self.df.
        Columns will be named like 'corr_EURUSD_GBPUSD'.
        """
        # Pivot to wide format: rows = time, columns = currencies
        df_pivot = self.df.pivot(
            index="open_time", columns="currency", values="log_ret"
        )

        # Prepare dict to hold new columns
        corr_cols = {}

        # Compute pairwise rolling correlations
        for curr1, curr2 in combinations(df_pivot.columns, 2):
            rolling_corr = df_pivot[curr1].rolling(window).corr(df_pivot[curr2])
            col_name = f"corr_{curr1}_{curr2}"
            corr_cols[col_name] = rolling_corr

        # Combine into a DataFrame
        df_corr = pd.DataFrame(corr_cols)

        # Merge back into self.df on open_time
        self.df = self.df.merge(
            df_corr, left_on="open_time", right_index=True, how="left"
        )

        return self.df

    def pca_factors(self, window: int = 15, n_components: int = 2):
        """Compute rolling PCA factors over all currencies' returns."""
        df_pivot = self.df.pivot(
            index="open_time", columns="currency", values="log_ret"
        )
        df_pca = pd.DataFrame(
            index=df_pivot.index, columns=[f"pca_{i + 1}" for i in range(n_components)]
        )

        for i in range(window, len(df_pivot)):
            window_data = df_pivot.iloc[i - window : i].dropna(
                axis=1, how="any"
            )  # only currencies with data
            if window_data.shape[1] == 0:
                continue
            pca = PCA(n_components=min(n_components, window_data.shape[1]))
            pca.fit(window_data)
            df_pca.iloc[i, : pca.n_components_] = pca.transform(window_data.iloc[-1:])[
                0
            ]

        return df_pca
