import pandas as pd
import numpy as np


class DataIntegrityChecker:
    """
    Performs integrity checks, cleaning, imputations, and basic exploratory diagnostics
    for multi-currency OHLCV data.
    """

    def __init__(self, df: pd.DataFrame = None):
        if df is None:
            df = pd.read_parquet("data/raw/currencies_market_data.parquet")
        self.df: pd.DataFrame = df.copy()
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
            print(f"⚠️ Found {dup_count} duplicate entries. Dropping them.")
            self.df.drop_duplicates(subset=["open_time", "currency"], inplace=True)
        else:
            print("✅ No duplicate entries detected.")

    def check_date_ranges(self):
        """Print date ranges per currency and overall."""
        print("📅 Data date ranges by currency:")
        for curr in self.df["currency"].unique():
            df_curr = self.df[self.df["currency"] == curr]
            start, end = df_curr["open_time"].min(), df_curr["open_time"].max()
            print(f"   - {curr}: {start} to {end}")
        overall_start, overall_end = (
            self.df["open_time"].min(),
            self.df["open_time"].max(),
        )
        print(f"\n🌐 Overall date range: {overall_start} to {overall_end}")

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
        """Check for missing 1-min intervals and forward-fill gaps within valid trading hours."""
        print("🕒 Checking time continuity per currency...")
        continuity_issues = {}
        total_missing = 0
        total_expected = 0
        filled_dfs = []

        for curr in self.df["currency"].unique():
            df_curr = self.df[self.df["currency"] == curr].sort_values("open_time")

            # Time differences
            deltas = df_curr["open_time"].diff().dropna()
            missing_minutes = (deltas != pd.Timedelta(minutes=1)).sum()

            # Valid trading times mask
            valid_mask = ~(
                (df_curr["open_time"].dt.day_name() == "Saturday")
                | (
                    (df_curr["open_time"].dt.day_name() == "Sunday")
                    & (df_curr["open_time"].dt.hour < 22)
                )
                | (
                    (df_curr["open_time"].dt.day_name() == "Friday")
                    & (df_curr["open_time"].dt.hour >= 22)
                )
            )

            expected_rows = valid_mask.sum()
            missing_fraction = (
                missing_minutes / expected_rows if expected_rows > 0 else 0
            )

            # Gaps per weekday
            if missing_minutes > 0:
                gap_times = df_curr["open_time"][1:][deltas != pd.Timedelta(minutes=1)]
                gaps_per_weekday = gap_times.dt.day_name().value_counts().to_dict()
                continuity_issues[curr] = (
                    missing_minutes,
                    missing_fraction,
                    gaps_per_weekday,
                )
            else:
                continuity_issues[curr] = (0, 0.0, {})

            total_missing += missing_minutes
            total_expected += expected_rows

            # Forward-fill gaps
            start = df_curr.loc[valid_mask, "open_time"].min()
            end = df_curr.loc[valid_mask, "open_time"].max()
            all_times = pd.date_range(start=start, end=end, freq="1min")
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

        self.df = pd.concat(filled_dfs, ignore_index=True)
        self.df = self.df[
            ["open_time", "currency", "open", "high", "low", "close", "volume"]
        ].copy()

        # Print gaps summary
        if continuity_issues:
            print("⚠️ Time discontinuities found:")
            for k, (count, frac, gaps_by_day) in continuity_issues.items():
                gaps_str = ", ".join(
                    f"{day}: {cnt}" for day, cnt in gaps_by_day.items()
                )
                if gaps_str:
                    print(
                        f"   - {k}: {count} gaps ({frac:.2%} missing) | Gaps by weekday: {gaps_str}"
                    )
                else:
                    print(f"   - {k}: {count} gaps ({frac:.2%} missing)")
        else:
            print("✅ All currency time series are continuous.")

        overall_missing_fraction = (
            total_missing / total_expected if total_expected > 0 else 0
        )
        print(
            f"\n📊 Overall missing data across all currencies: {overall_missing_fraction:.2%} out of {total_expected:,} 1m time bars."
        )
        print("✅ Small gaps forward-filled (only within valid trading window).")

    def summary_statistics(self):
        """Return summary statistics per currency."""
        summary = self.df.groupby("currency")[
            ["open", "high", "low", "close", "volume"]
        ].describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99])
        return summary

    def run_all_checks(self):
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
        self.check_weekend_data()
        print()
        self.check_time_continuity()
        summary = self.summary_statistics()
        print("\n✅ Data integrity checks complete.")
        return self.df, summary
