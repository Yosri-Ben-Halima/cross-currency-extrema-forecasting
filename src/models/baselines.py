from typing import Literal, Optional
import pandas as pd


class RuleBasedBenchmark:
    def __init__(
        self,
        df: pd.DataFrame,
        mode: Optional[Literal["naive", "volatility_adjusted"]] = "naive",
        multiplier: Optional[float] = 1.0,
    ):
        self.df = df.copy()
        self.df = self.df.sort_values(by="open_time")
        self.mode = mode
        self.multiplier = multiplier

    def predict(self) -> pd.DataFrame:
        """Generate rule-based predictions for each currency."""
        print(
            f"🔹 Generating {self.mode.replace('_', ' ')} rule-based benchmark predictions..."
        )
        for currency in self.df["currency"].unique():
            df_curr = self.df[self.df["currency"] == currency]
            if self.mode == "naive":
                self.df.loc[df_curr.index, "y_high_pred"] = df_curr["high"].shift(1)
                self.df.loc[df_curr.index, "y_low_pred"] = df_curr["low"].shift(1)
            else:
                vol = df_curr["close"].rolling(window=15).std().shift(1)
                self.df.loc[df_curr.index, "y_high_pred"] = (
                    df_curr["high"].shift(1) + self.multiplier * vol
                )
                self.df.loc[df_curr.index, "y_low_pred"] = (
                    df_curr["low"].shift(1) - self.multiplier * vol
                )

        self.df[
            [
                "y_high_pred",
                "y_low_pred",
            ]
        ] = self.df[
            [
                "y_high_pred",
                "y_low_pred",
            ]
        ].bfill()

        print(
            f"✅ {self.mode.replace('_', ' ').capitalize()} rule-based predictions generated."
        )

        return self.df
