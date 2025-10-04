import pandas as pd
import numpy as np
# from IPython.display import display


class DataIntegrityChecker:
    """
    Performs integrity checks and basic exploratory diagnostics
    for multi-currency OHLCV data.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
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
        missing_cols = set(self.expected_cols) - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        print("✅ Column structure verified.")

    def log_date_ranges(self):
        print("📅 Data date ranges by currency:")
        for curr in self.df["currency"].unique():
            df_curr = self.df[self.df["currency"] == curr]
            start = df_curr["open_time"].min()
            end = df_curr["open_time"].max()
            print(f"   - {curr}: {start} to {end}")
        overall_start = self.df["open_time"].min()
        overall_end = self.df["open_time"].max()
        print(f"\n🌐 Overall date range: {overall_start} to {overall_end}")

    def check_dtypes(self):
        print("📊 Data types:")
        print(self.df.dtypes)
        # print()
        if not np.issubdtype(self.df["open_time"].dtype, np.datetime64):
            print("⚠️ 'open_time' not in datetime format. Converting...")
            self.df["open_time"] = pd.to_datetime(self.df["open_time"])

    def check_duplicates(self):
        dup_count = self.df.duplicated(subset=["open_time", "currency"]).sum()
        if dup_count > 0:
            print(
                f"⚠️ Found {dup_count} duplicate timestamp-currency entries. Dropping them."
            )
            self.df.drop_duplicates(subset=["open_time", "currency"], inplace=True)
        else:
            print("✅ No duplicate entries detected.")

    def check_sunday_data(self):
        print("🕒 Checking for Sunday data entries for elimination...")
        sunday_issues = {}
        total_sundays = 0

        for curr in self.df["currency"].unique():
            df_curr = self.df[self.df["currency"] == curr].sort_values("open_time")
            sundays = df_curr[df_curr["open_time"].dt.day_name() == "Sunday"]
            sunday_count = len(sundays)
            if sunday_count > 0:
                sunday_issues[curr] = sunday_count
                total_sundays += sunday_count

        if sunday_issues:
            print("⚠️ Sunday data entries found and will be removed:")
            for k, v in sunday_issues.items():
                print(
                    f"   - {k}: {v} Sunday entries ({v / self.df[self.df['currency'] == k].shape[0]:.2%} of data)"
                )
            print(
                f"\n📊 Total Sunday entries across all currencies: {total_sundays} ({total_sundays / self.df.shape[0]:.2%} of data). Dropping them."
            )

            # Remove Sunday rows
            self.df = self.df[self.df["open_time"].dt.day_name() != "Sunday"]
        else:
            print("✅ No Sunday data entries detected.")

    def check_missing_values(self):
        na_summary = self.df.isna().sum()
        if na_summary.any():
            print("⚠️ NaN values detected:")
            print(na_summary[na_summary > 0])
        else:
            print("✅ No NaN values detected.")

    def check_time_continuity(self):
        print("🕒 Checking time continuity per currency...")
        continuity_issues = {}
        total_missing = 0
        total_expected = 0

        for curr in self.df["currency"].unique():
            df_curr = self.df[self.df["currency"] == curr].sort_values("open_time")

            # Compute time differences
            deltas = df_curr["open_time"].diff().dropna()
            missing_minutes = (deltas != pd.Timedelta(minutes=1)).sum()

            # Expected number of rows assuming perfect 1-min frequency
            expected_rows = (
                df_curr["open_time"].iloc[-1] - df_curr["open_time"].iloc[0]
            ).total_seconds() / 60 + 1
            missing_fraction = missing_minutes / expected_rows

            # Compute number of gaps per weekday
            if missing_minutes > 0:
                # Find timestamps where gap occurred (where delta != 1 min)
                gap_times = df_curr["open_time"][1:][deltas != pd.Timedelta(minutes=1)]
                gaps_per_weekday = gap_times.dt.day_name().value_counts().to_dict()
                continuity_issues[curr] = (
                    missing_minutes,
                    missing_fraction,
                    gaps_per_weekday,
                )
            else:
                continuity_issues[curr] = (0, 0.0, {})

            # Aggregate for overall stats
            total_missing += missing_minutes
            total_expected += expected_rows

        # Print per-currency issues
        if continuity_issues:
            print("⚠️ Time discontinuities found:")
            for k, (count, frac, gaps_by_day) in continuity_issues.items():
                gaps_by_day_str = ", ".join(
                    f"{day}: {cnt}" for day, cnt in gaps_by_day.items()
                )
                if gaps_by_day_str:
                    print(
                        f"   - {k}: {count} gaps ({frac:.2%} missing) | Gaps by weekday: {gaps_by_day_str}"
                    )
                else:
                    print(f"   - {k}: {count} gaps ({frac:.2%} missing)")
        else:
            print("✅ All currency time series are continuous.")

        # Print overall missing fraction across all currencies
        overall_missing_fraction = total_missing / total_expected
        print(
            f"\n📊 Overall missing data across all currencies: {overall_missing_fraction:.2%}"
        )

    def summary_statistics(self):
        # print("\n📈 Summary statistics per currency:")
        summary = self.df.groupby("currency")[
            ["open", "high", "low", "close", "volume"]
        ].describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99])
        # display(summary)
        return summary

    def run_all_checks(self):
        print("🚀 Running data integrity pipeline...\n")
        self.check_columns()
        print()
        self.check_dtypes()
        print()
        self.check_duplicates()
        print()
        self.log_date_ranges()
        print()
        self.check_missing_values()
        print()
        self.check_sunday_data()
        print()
        self.check_time_continuity()
        # print()
        summary = self.summary_statistics()
        print("\n✅ Data integrity checks complete.")
        return self.df, summary
