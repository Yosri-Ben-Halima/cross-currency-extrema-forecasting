from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
from typing import List, Optional


class Evaluator:
    """
    Evaluates regression model performance using various metrics.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        targets: Optional[List[str]] = None,
        predictions: Optional[List[str]] = None,
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing true targets and predictions
            targets (list, optional): list of target column names. Defaults to ['y_high', 'y_low']
            predictions (list, optional): list of prediction column names. Defaults to ['y_high_pred', 'y_low_pred']
        """
        self.df = df.copy()
        self.targets = targets or ["y_high", "y_low"]
        self.predictions = predictions or ["y_high_pred", "y_low_pred"]

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluate model performance using MAE, MSE, RMSE, MAPE, and R2 metrics.

        Args:
            average (str, optional): Averaging method for multi-target metrics. Defaults to 'macro'.
        Returns:
            pd.DataFrame: DataFrame summarizing evaluation metrics for each target and overall.
        """
        results = []
        for target, pred in zip(self.targets, self.predictions):
            y_true = self.df[target]
            y_pred = self.df[pred]

            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = mean_absolute_percentage_error(y_true, y_pred)

            results.append(
                {
                    "target": target,
                    "RMSE": rmse,
                    "MAPE": mape,
                }
            )

        results_df = pd.DataFrame(results)

        overall_rmse = results_df["RMSE"].mean()
        overall_mape = results_df["MAPE"].mean()

        overall_results = {
            "target": "overall",
            "RMSE": overall_rmse,
            "MAPE": overall_mape,
        }

        results_df = pd.concat(
            [results_df, pd.DataFrame([overall_results])], ignore_index=True
        )

        print("✅ Evaluation complete.")
        return results_df

    def evaluate_currency(self, currency: str) -> pd.DataFrame:
        """
        Evaluate model performance for a specific currency.

        Args:
            currency (str): Currency to filter the DataFrame.
        Returns:
            pd.DataFrame: DataFrame summarizing evaluation metrics for the specified currency.
        """
        df_curr = self.df[self.df["currency"] == currency]
        if df_curr.empty:
            raise ValueError(f"No data found for currency: {currency}")

        evaluator_curr = Evaluator(df_curr, self.targets, self.predictions)
        return evaluator_curr.evaluate()

    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate model performance for all currencies.

        Returns:
            pd.DataFrame: DataFrame summarizing evaluation metrics for each currency and overall.
        """
        all_results = []
        for currency in self.df["currency"].unique():
            results_curr = self.evaluate_currency(currency=currency)
            results_curr["currency"] = currency
            all_results.append(results_curr)

        all_results_df = pd.concat(all_results, ignore_index=True)

        overall_results = self.evaluate()
        overall_results["currency"] = "overall"
        all_results_df = pd.concat([all_results_df, overall_results], ignore_index=True)

        return all_results_df
