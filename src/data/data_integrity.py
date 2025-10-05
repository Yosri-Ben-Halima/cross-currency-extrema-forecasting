import pandas as pd
import numpy as np


class DataIntegrityChecker:
    """
    Performs integrity checks, cleaning, imputations, and basic exploratory diagnostics
    for multi-currency OHLCV data.
    """

    DAILY_NUMBER_OF_BARS = {
        "Monday": 1440,
        "Tuesday": 1440,
        "Wednesday": 1440,
        "Thursday": 1440,
        "Friday": 1320,  # Friday only until 22:00 UTC
        "Saturday": 0,
        "Sunday": 120,  # Sunday only after 22:00 UTC
    }

    def __init__(self, df: pd.DataFrame = None):
        if df is None:
            df = pd.read_parquet("data/raw/currencies_market_data.parquet")
        self.df: pd.DataFrame = df.copy()
        self.initial_n_rows = self.df.shape[0]
        self.expected_cols = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "currency",
        ]

    def check_columns(self):
        """Verify expected columns are present."""
        missing_cols = set(self.expected_cols) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        print("✅ Column structure verified.")

    def check_dtypes(self):
        """Ensure 'open_time' is datetime type."""
        print("📊 Data types:")
        print(self.df.dtypes)
        if not np.issubdtype(self.df["open_time"].dtype, np.datetime64):
            print("⚠️ 'open_time' not in datetime format. Converting...")
            self.df["open_time"] = pd.to_datetime(self.df["open_time"], errors="coerce")
            if self.df["open_time"].isna().any():
                print(
                    "⚠️ Some 'open_time' values could not be converted and are now NaT."
                )

    def check_duplicates(self):
        """Remove duplicate timestamp-currency entries."""
        dup_count = self.df.duplicated(subset=["open_time", "currency"]).sum()
        if dup_count > 0:
            print(
                f"⚠️ Found {dup_count} duplicate entries ({dup_count / self.df.shape[0]:.2%} of all data). Dropping them."
            )
            self.df.drop_duplicates(subset=["open_time", "currency"], inplace=True)
        else:
            print("✅ No duplicate entries detected.")

    def check_date_ranges(self):
        """Print date ranges per currency and overall."""
        print("📅 Data date ranges by currency:")
        for curr in self.df["currency"].unique():
            df_curr = self.df[self.df["currency"] == curr]
            start, end = df_curr["open_time"].min(), df_curr["open_time"].max()
            print(f"   - {curr}: {start} → {end}")
        overall_start, overall_end = (
            self.df["open_time"].min(),
            self.df["open_time"].max(),
        )
        print(f"\n🌐 Overall date range: {overall_start} to {overall_end}")

    def clip_last_date(self):
        """Clip data to exclude entries after a 2025-09-22."""
        cutoff_day = "2025-09-22 00:00:00"
        cutoff = pd.to_datetime(cutoff_day)
        initial_count = self.df.shape[0]
        self.df = self.df[self.df["open_time"] < cutoff].copy()
        dropped = initial_count - self.df.shape[0]
        if dropped > 0:
            print(
                f"⚠️ Dropped {dropped} entries after {cutoff_day} ({dropped / self.df.shape[0]:.2%} of all data) to align dataset (incomplete day)."
            )
        else:
            print(f"✅ No entries found after {cutoff_day}.")

    def check_missing_values(self):
        """Check for NaN values in the dataset."""
        na_summary = self.df.isna().sum()
        if na_summary.any():
            print("⚠️ NaN values detected:")
            print(na_summary[na_summary > 0])
        else:
            print("✅ No NaN values detected.")

    def check_weekend_data(self):
        """Remove rows that fall on Sunday <22:00 UTC or Friday >=22:00 UTC."""
        print("🕒 Checking for weekend boundary data entries for elimination...")
        df = self.df.copy()
        df["weekday"] = df["open_time"].dt.day_name()
        df["hour"] = df["open_time"].dt.hour

        weekend_issues = {}
        total_dropped = 0

        for curr in df["currency"].unique():
            df_curr = df[df["currency"] == curr]
            mask = (
                ((df_curr["weekday"] == "Sunday") & (df_curr["hour"] < 22))
                | ((df_curr["weekday"] == "Friday") & (df_curr["hour"] >= 22))
                | (df_curr["weekday"] == "Saturday")
            )
            count = mask.sum()
            if count > 0:
                weekend_issues[curr] = count
                total_dropped += count

        if weekend_issues:
            print("⚠️ Weekend boundary entries found and will be removed:")
            for k, v in weekend_issues.items():
                pct = v / self.df[self.df["currency"] == k].shape[0]
                print(f"   - {k}: {v} entries ({pct:.2%} of data)")
            print(
                f"\n📊 Total entries dropped: {total_dropped} ({total_dropped / self.df.shape[0]:.2%} of all data)."
            )

            drop_mask = (
                ((df["weekday"] == "Sunday") & (df["hour"] < 22))
                | ((df["weekday"] == "Friday") & (df["hour"] >= 22))
                | (df["weekday"] == "Saturday")
            )
            self.df = df[~drop_mask].drop(columns=["weekday", "hour"]).copy()
        else:
            print("✅ No weekend boundary data detected.")

    def check_time_continuity(self):
        """
        Compute missing 1-min bars per currency and forward-fill gaps
        within valid trading hours, taking into account per-day expected bars.
        """
        print("🕒 Checking time continuity per currency...")
        continuity_issues = {}
        filled_dfs = []
        total_expected_bars = 0
        total_missing_bars = 0

        # Expected number of bars per weekday (adjust to your dataset/timezone)

        for curr in self.df["currency"].unique():
            df_curr = (
                self.df[self.df["currency"] == curr].sort_values("open_time").copy()
            )
            df_curr["date"] = df_curr["open_time"].dt.date
            df_curr["day_of_week"] = df_curr["open_time"].dt.day_name()

            # Count actual bars per day
            bars_per_day = (
                df_curr.groupby(["date", "day_of_week"])
                .size()
                .reset_index(name="bars_count")
            )
            # Map expected bars per day
            bars_per_day["expected_bars"] = bars_per_day["day_of_week"].map(
                self.DAILY_NUMBER_OF_BARS
            )
            bars_per_day["difference"] = (
                bars_per_day["expected_bars"] - bars_per_day["bars_count"]
            )
            bars_per_day["fraction"] = (
                bars_per_day["bars_count"] / bars_per_day["expected_bars"]
            )

            # Total missing vs expected
            total_missing = bars_per_day["difference"].sum()
            total_expected = bars_per_day["expected_bars"].sum()
            missing_fraction = (
                total_missing / total_expected if total_expected > 0 else 0
            )
            total_expected_bars += total_expected
            total_missing_bars += total_missing

            continuity_issues[curr] = (total_missing, missing_fraction)

            # -------------------------------
            # Forward-fill small gaps within valid trading times
            # -------------------------------
            valid_mask = ~(
                (df_curr["day_of_week"] == "Saturday")
                | (
                    (df_curr["day_of_week"] == "Sunday")
                    & (df_curr["open_time"].dt.hour < 22)
                )
                | (
                    (df_curr["day_of_week"] == "Friday")
                    & (df_curr["open_time"].dt.hour >= 22)
                )
            )
            start = df_curr.loc[valid_mask, "open_time"].min()
            end = df_curr.loc[valid_mask, "open_time"].max()
            all_times = pd.date_range(start=start, end=end, freq="1min")

            # Filter all_times by valid trading hours
            valid_times_mask = ~(
                (all_times.day_name() == "Saturday")
                | ((all_times.day_name() == "Sunday") & (all_times.hour < 22))
                | ((all_times.day_name() == "Friday") & (all_times.hour >= 22))
            )
            all_times = all_times[valid_times_mask]

            df_curr = df_curr.set_index("open_time").reindex(all_times)
            df_curr["currency"] = curr
            df_curr[["open", "high", "low", "close", "volume"]] = df_curr[
                ["open", "high", "low", "close", "volume"]
            ].ffill()
            filled_dfs.append(
                df_curr.reset_index().rename(columns={"index": "open_time"})
            )

        # Combine all currencies
        self.df = pd.concat(filled_dfs, ignore_index=True)[
            ["open_time", "currency", "open", "high", "low", "close", "volume"]
        ]

        # Print per-currency missing bars
        print("⚠️ Time discontinuities per currency:")
        for k, (missing, frac) in continuity_issues.items():
            print(f"   - {k}: {missing} missing bars ({frac:.2%})")

        print(
            f"\n📊 Total missing bars across all currencies: {total_missing_bars:,} out of {self.df.shape[0]:,} expected ({total_missing_bars / self.df.shape[0]:.2%})."
        )

        print("✅ Gaps forward-filled within valid trading hours.")

    def summary_statistics(self):
        """Return summary statistics per currency."""
        summary = self.df.groupby("currency")[
            ["open", "high", "low", "close", "volume"]
        ].describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99])
        return summary

    def run_all_checks(self, return_summary: bool = True):
        """Run the full data integrity pipeline."""
        print("🚀 Running data integrity pipeline...\n")
        self.check_columns()
        print()
        self.check_dtypes()
        print()
        self.check_duplicates()
        print()
        self.check_missing_values()
        print()
        self.check_date_ranges()
        print()
        self.clip_last_date()
        print()
        self.check_weekend_data()
        print()
        self.check_time_continuity()
        summary = self.summary_statistics()
        print(
            f"\n✅ Data integrity checks complete. {f'initial number of rows: {self.initial_n_rows:,}, final number of rows: {self.df.shape[0]:,} ({self.df.shape[0] / self.initial_n_rows:.2%} of all data).' if self.initial_n_rows != self.df.shape[0] else ''}"
        )
        if return_summary:
            return self.df, summary
