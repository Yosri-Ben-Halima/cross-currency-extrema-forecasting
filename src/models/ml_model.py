import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm


# ----------------------------------------
# Sliding window dataset
# ----------------------------------------
class SlidingWindowDataset:
    def __init__(self, df: pd.DataFrame, feature_cols, seq_len=60, currency_order=None):
        self.df = df.copy()
        self.df["open_time"] = pd.to_datetime(self.df["open_time"])
        self.df = self.df.sort_values(["open_time", "currency"]).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.targets = ["D_y_high", "D_y_low"]
        self.seq_len = seq_len
        self.currencies = currency_order or sorted(self.df["currency"].unique())

    def build(self):
        rows, y_highs, y_lows, times, currencies = [], [], [], [], []

        for c in tqdm(self.currencies, desc="Building windows"):
            df_c = self.df[self.df["currency"] == c].reset_index(drop=True)
            if len(df_c) <= self.seq_len:
                continue
            for i in range(len(df_c) - self.seq_len):
                X_window = df_c.loc[
                    i : i + self.seq_len - 1, self.feature_cols
                ].values.flatten()
                y_high = df_c.loc[i + self.seq_len, self.targets[0]]
                y_low = df_c.loc[i + self.seq_len, self.targets[1]]
                rows.append(X_window)
                y_highs.append(y_high)
                y_lows.append(y_low)
                times.append(df_c.loc[i + self.seq_len, "open_time"])
                currencies.append(c)

        X = np.array(rows, dtype=np.float32)
        y_highs = np.array(y_highs, dtype=np.float32)
        y_lows = np.array(y_lows, dtype=np.float32)
        return X, y_highs, y_lows, times, currencies


# ----------------------------------------
# XGBoost regressor wrapper
# ----------------------------------------
class XGBoostSeq2One:
    def __init__(self, seq_len=60, feature_cols=None, params=None):
        self.seq_len = seq_len
        self.feature_cols = feature_cols or []
        self.params = params or dict(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        self.targets = ["D_y_high", "D_y_low"]
        self.model_high: XGBRegressor = XGBRegressor(**self.params)
        self.model_low: XGBRegressor = XGBRegressor(**self.params)
        self.scaler = StandardScaler()

    def fit(self, df_train, df_val):
        train_ds = SlidingWindowDataset(
            df_train,
            self.feature_cols,
            self.seq_len,
        )
        val_ds = SlidingWindowDataset(
            df_val,
            self.feature_cols,
            self.seq_len,
        )
        X_train, y_high_train, y_low_train, *_ = train_ds.build()
        X_val, y_high_val, y_low_val, *_ = val_ds.build()

        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        print(f"🔹 Training model for {self.targets[0]}...")
        self.model_high.fit(
            X_train, y_high_train, eval_set=[(X_val, y_high_val)], verbose=False
        )
        print(f"🔹 Training model for {self.targets[1]}...")
        self.model_low.fit(
            X_train, y_low_train, eval_set=[(X_val, y_low_val)], verbose=False
        )

        # Evaluate
        pred_high = self.model_high.predict(X_val)
        pred_low = self.model_low.predict(X_val)
        mape_high = root_mean_squared_error(y_high_val, pred_high)
        mape_low = root_mean_squared_error(y_low_val, pred_low)
        print(
            f"✅ Validation RMSE | {self.targets[0]}: {mape_high:.3f} | {self.targets[1]}: {mape_low:.3f}"
        )

    def predict(self, full_df, df):
        ds = SlidingWindowDataset(
            df,
            self.feature_cols,
            self.seq_len,
        )

        X, y_high, y_low, times, currencies = ds.build()

        initial_values_per_currency = full_df[
            full_df["open_time"] == str(times[0] - pd.Timedelta(minutes=15))
        ][["currency", "y_high", "y_low"]]

        X_scaled = self.scaler.transform(X)
        pred_high = self.model_high.predict(X_scaled)
        pred_low = self.model_low.predict(X_scaled)

        df_preds = pd.DataFrame(
            {
                "open_time": times,
                "currency": currencies,
                "y_high": y_high.cumsum(),
                "y_low": y_low.cumsum(),
                "y_high_pred": pred_high.cumsum(),
                "y_low_pred": pred_low.cumsum(),
            }
        )
        for curr in list(set(currencies)):
            df_preds.loc[df_preds["currency"] == curr, "y_high"] = (
                df_preds.loc[df_preds["currency"] == curr, "y_high"]
                + initial_values_per_currency.loc[
                    initial_values_per_currency["currency"] == curr, "y_high"
                ].values[0]
            )
            df_preds.loc[df_preds["currency"] == curr, "y_high_pred"] = (
                df_preds.loc[df_preds["currency"] == curr, "y_high_pred"]
                + initial_values_per_currency.loc[
                    initial_values_per_currency["currency"] == curr, "y_high"
                ].values[0]
            )

            df_preds.loc[df_preds["currency"] == curr, "y_low"] = (
                df_preds.loc[df_preds["currency"] == curr, "y_low"]
                + initial_values_per_currency.loc[
                    initial_values_per_currency["currency"] == curr, "y_low"
                ].values[0]
            )
            df_preds.loc[df_preds["currency"] == curr, "y_low_pred"] = (
                df_preds.loc[df_preds["currency"] == curr, "y_low_pred"]
                + initial_values_per_currency.loc[
                    initial_values_per_currency["currency"] == curr, "y_low"
                ].values[0]
            )
        return df_preds


class XGBoostSeq2OneByCurrency:
    def __init__(self, feature_cols, seq_len=60):
        """
        Trains a separate `XGBoostSeq2One` model per currency.

        Parameters
        ----------
        feature_cols : list[str]
            List of feature column names.
        seq_len : int
            Sequence length for windowing.
        """
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.models = {}

    def train(self, train_df, val_df):
        """
        Train a separate model for each currency.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training set.
        val_df : pd.DataFrame
            Validation set.
        """
        currencies = train_df["currency"].unique()

        for curr in currencies:
            print(f"----------- Training model for {curr} -----------")

            model = XGBoostSeq2One(
                seq_len=self.seq_len,
                feature_cols=self.feature_cols,
            )

            df_train = train_df[train_df["currency"] == curr].copy()
            df_val = val_df[val_df["currency"] == curr].copy()

            model.fit(df_train=df_train, df_val=df_val)
            self.models[curr] = model

            print(f"----------- {curr} training complete -----------\n")

    def predict(self, full_df, test_df):
        """
        Predict using trained models for each currency.

        Parameters
        ----------
        full_df : pd.DataFrame
            Full dataset used to retrieve first target values for
            reconstructing sequences since `y_(high/low) = y_(high/low)_0 + cumsum(D_y_(high/low))`.
        test_df : pd.DataFrame
            Test set.

        Returns
        -------
        pd.DataFrame
            Concatenated predictions for all currencies.
        """
        preds_list = []

        for curr, model in self.models.items():
            print(f"----------- Predicting for {curr} -----------")

            df_test = test_df[test_df["currency"] == curr].copy()
            preds_curr = model.predict(full_df, df_test)
            preds_list.append(preds_curr)

            print(f"----------- {curr} predictions done -----------\n")

        return pd.concat(preds_list, ignore_index=True)
