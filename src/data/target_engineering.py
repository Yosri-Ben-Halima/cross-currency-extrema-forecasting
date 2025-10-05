import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CrossCurrencyLabeler:
    """
    Generates primary regression targets (next-hour high/low),
    optional triple-barrier meta-labels, and sample weights
    for multi-currency OHLCV data.
    """

    K = {
        "USDZAR": 1 / 3,
        "EURJPY": 1 / 30,
        "AUDJPY": 1 / 17,
        "SGDJPY": 1 / 20,
        "USDJPY": 1 / 25,
        "GBPJPY": 1 / 35,
        "EURUSD": 5,
        "GBPUSD": 4,
        "AUDUSD": 9,
        "USDCAD": 4,
        "USDCHF": 7,
        "NZDUSD": 10,
        "AUDSGD": 7,
        "EURGBP": 6,
        "EURAUD": 3,
    }

    def __init__(self, df, horizon=60, vol_window=15):
        """
        Args:
            df (pd.DataFrame): must contain ['open', 'high', 'low', 'close', 'volume', 'currency']
            horizon (int): prediction horizon in minutes
            k_up (float): multiplier for upper barrier
            k_down (float): multiplier for lower barrier
            vol_window (int): window for rolling volatility
        """
        self.df = df.copy()
        self.horizon = horizon
        self.vol_window = vol_window

    def compute_vertical_barrier_targets(self):
        """
        Computes next-hour maximum high and minimum low per currency.
        """
        y_high_list = []
        y_low_list = []

        for curr in self.df["currency"].unique():
            df_curr = self.df[self.df["currency"] == curr].sort_index()
            highs = df_curr["high"].values
            lows = df_curr["low"].values

            future_high = pd.Series(
                [
                    highs[i + 1 : i + self.horizon + 1].max()
                    if i + self.horizon < len(highs)
                    else np.nan
                    for i in range(len(highs))
                ],
                index=df_curr.index,
            )
            future_low = pd.Series(
                [
                    lows[i + 1 : i + self.horizon + 1].min()
                    if i + self.horizon < len(lows)
                    else np.nan
                    for i in range(len(lows))
                ],
                index=df_curr.index,
            )

            y_high_list.append(future_high)
            y_low_list.append(future_low)

        self.df["y_high"] = pd.concat(y_high_list).sort_index()
        self.df["y_low"] = pd.concat(y_low_list).sort_index()

    def compute_triple_barrier_labels(self):
        meta_labels_list = []

        for curr in self.df["currency"].unique():
            df_curr = self.df[self.df["currency"] == curr].sort_index()
            close = df_curr["close"]
            vol = df_curr["close"].rolling(self.vol_window).std()

            # Avoid zero volatility at the beginning
            vol = vol.fillna(vol.mean())
            k = self.K.get(curr, 1)  # default k=1 if currency not found

            upper_barrier = close * (1 + k * vol)
            lower_barrier = close * (1 - k * vol)

            labels = np.zeros(len(close), dtype=int)

            for i in range(len(close)):
                end_idx = min(i + self.horizon, len(close) - 1)
                future_prices = close.iloc[i : end_idx + 1]
                ub = upper_barrier.iloc[i]
                lb = lower_barrier.iloc[i]

                # Find first hit
                hit_up = np.where(future_prices >= ub)[0]
                hit_down = np.where(future_prices <= lb)[0]

                if len(hit_up) > 0 and (len(hit_down) == 0 or hit_up[0] < hit_down[0]):
                    labels[i] = 1
                elif len(hit_down) > 0 and (
                    len(hit_up) == 0 or hit_down[0] < hit_up[0]
                ):
                    labels[i] = -1
                # else: remains 0 (no barrier hit)

            meta_labels_list.append(pd.Series(labels, index=df_curr.index))

        # Concatenate per-currency labels properly
        self.df["meta_label"] = pd.concat(meta_labels_list, axis=0)

    def compute_sample_weights(self):
        """
        Compute sample weights based on overlap of forward windows.
        """
        weight_list = []

        for curr in self.df["currency"].unique():
            df_curr = self.df[self.df["currency"] == curr].sort_index()
            n = len(df_curr)
            weights = np.zeros(n)

            for i in range(n):
                # count overlaps in horizon
                end_idx = min(i + self.horizon, n - 1)
                weights[i] = 1 / (end_idx - i + 1)

            weight_list.append(pd.Series(weights, index=df_curr.index))

        self.df["sample_weight"] = pd.concat(weight_list).sort_index()

    def run_labeling_pipeline(self, compute_meta=True, compute_weights=True):
        """
        Run the full labeling pipeline.
        """
        print("🚀 Running labeling pipeline...")
        self.compute_vertical_barrier_targets()
        print("✅ Vertical-barrier regression labels created.")

        if compute_meta:
            self.compute_triple_barrier_labels()
            print("✅ Triple-barrier meta labels created.")

        if compute_weights:
            self.compute_sample_weights()
            print("✅ Sample weights computed.")

        print("✅ Labeling pipeline complete.")
        return self.df


class LabelingVisualizer:
    """
    Diagnostic visualizations for Cross-Currency High/Low Labeling.
    """

    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): must contain ['close', 'y_high', 'y_low', 'meta_label', 'currency']
        """
        self.df = df.copy()

    def plot_target_distributions(self, currency=None):
        """
        Plot distribution of future high/low relative to current close.
        """
        df = self.df
        if currency:
            df = df[df["currency"] == currency]

        df.loc[:, "high_diff"] = df["y_high"] - df["close"]
        df.loc[:, "low_diff"] = df["close"] - df["y_low"]

        plt.figure(figsize=(14, 6))
        sns.histplot(
            df["high_diff"],
            color="green",
            label="future_high - close",
            kde=True,
            stat="density",
            bins=50,
        )
        sns.histplot(
            df["low_diff"],
            color="red",
            label="close - future_low",
            kde=True,
            stat="density",
            bins=50,
        )
        plt.title(
            f"Distribution of Future High/Low Differences ({currency or 'All Currencies'})"
        )
        plt.xlabel("Price Difference")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_meta_label_balance(self, currency=None):
        """
        Plot the counts of meta-labels (+1, -1, 0)
        """
        df = self.df
        if currency:
            df = df[df["currency"] == currency]

        counts = df["meta_label"].value_counts().sort_index() / len(df)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=counts.index, y=counts.values)  # , palette="coolwarm")
        plt.title(f"Meta-Label Distribution ({currency or 'All Currencies'})")
        plt.xlabel("Meta Label")
        plt.ylabel("Proportion")
        plt.grid()
        plt.show()

    def plot_sample_sequences_with_barriers(self, currency, horizon, n_samples=3):
        """
        Plot a few random sequences of OHLC with regression barriers.
        """
        df_curr = self.df[self.df["currency"] == currency].sort_index()
        df_curr = df_curr.dropna(subset=["y_high", "y_low"])

        sample_starts = np.random.choice(
            df_curr.index[:-horizon], n_samples, replace=False
        )

        for start in sample_starts:
            start_idx = df_curr.index.get_loc(start)
            end_idx = start_idx + horizon
            seq = df_curr.iloc[start_idx:end_idx]
            seq_dates = seq["open_time"]

            plt.figure(figsize=(12, 5))
            plt.plot(seq_dates, seq["close"], label="Close", color="black")
            plt.plot(
                seq_dates,
                seq["y_high"],
                label="Future High",
                color="green",
                linestyle="--",
            )
            plt.plot(
                seq_dates, seq["y_low"], label="Future Low", color="red", linestyle="--"
            )

            plt.title(
                f"{currency} Sequence Starting at {self.df['open_time'].loc[start]}"
            )
            plt.xlabel("Timestamp Index")
            plt.ylabel("Price")
            plt.legend()
            plt.grid()
            plt.show()

    def plot_violin_targets(self, currencies=None):
        """
        Plots violin plots of y_low and y_high targets using seaborn.

        Parameters:
        -----------
        currencies: list[str] or None
            If provided, filters the DataFrame to only these currencies.
        """
        df_plot = self.df.copy()
        if currencies is not None:
            df_plot = df_plot[df_plot["currency"].isin(currencies)]

        # Melt to long format
        df_long = df_plot.melt(
            id_vars=["currency"] if "currency" in df_plot.columns else None,
            value_vars=["y_low", "y_high"],
            var_name="Target",
            value_name="Value",
        )

        plt.figure(figsize=(10, 6))
        sns.violinplot(
            x="Target",
            y="Value",
            hue="currency" if "currency" in df_long.columns else None,
            data=df_long,
            split=False,
            inner="box",
            palette="Set2",
        )
        plt.title("Distribution of y_low and y_high targets")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.grid()
        plt.show()
