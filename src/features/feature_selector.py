import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor


class FeatureSelector:
    """
    Performs feature orthogonalization and selection for multi-target ML.
    Orthogonalization is based on the mid-target: (y_high + y_low)/2
    """

    def __init__(self, df, feature_cols=None, targets=None, n_pca_components=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing features + targets
            feature_cols (list, optional): list of feature column names. Defaults to all non-target columns
            targets (list, optional): ['y_high', 'y_low', 'meta_label']
            n_pca_components (int, optional): number of PCA components to keep
        """
        self.df = df.copy().dropna().set_index(["open_time", "currency"])
        self.targets = targets or ["y_high", "y_low", "meta_label"]
        self.feature_cols = feature_cols or [
            c for c in df.columns if c not in self.targets
        ]
        self.n_pca_components = n_pca_components or len(self.feature_cols)
        self.pca_model = None
        self.selected_features = None

    def orthogonalize_features(self):
        """
        Orthogonalize features using supervised dimensionality reduction (PLSRegression)
        based on mid-target: (y_high + y_low)/2
        """
        print("🔹 Performing PLS-based orthogonalization...")

        # --- Compute mid target ---
        mid_target = (self.df["y_high"] + self.df["y_low"]) / 2
        y = mid_target.values.reshape(-1, 1)

        # --- Prepare feature matrix ---
        X = self.df[self.feature_cols].fillna(0).values  # Fill missing temporarily

        # --- Fit PLSRegression ---
        self.pls_model = PLSRegression(n_components=self.n_pca_components)
        X_pls, _ = self.pls_model.fit_transform(X, y)

        # --- Store PLS components as new features ---
        pls_cols = [f"pls_{i}" for i in range(X_pls.shape[1])]
        self.df[pls_cols] = X_pls
        self.selected_features = pls_cols

        print(
            f"✅ Orthogonalized {len(self.feature_cols)} features into {len(pls_cols)} PLS components."
        )

        return self.df

    def permutation_importance_selection(
        self, n_repeats=3, sample_frac=0.5, random_state=42
    ):
        """
        Compute permutation importance using y_mid (=(y_high + y_low)/2),
        with optional row sampling and fast HistGradientBoosting models.
        """
        print("🔹 Computing permutation importance on y_mid...")

        # --- Compute mid target ---
        y_mid = (self.df["y_high"] + self.df["y_low"]) / 2

        # --- Prepare feature matrix ---
        X = self.df[self.selected_features].fillna(0)

        # --- Sample rows to speed up computation ---
        if sample_frac < 1.0:
            X_sample = X.sample(frac=sample_frac, random_state=random_state)
            y_sample = y_mid.loc[X_sample.index]
        else:
            X_sample = X
            y_sample = y_mid

        # --- Choose fast HistGradientBoosting model ---
        model = HistGradientBoostingRegressor(max_iter=200, random_state=random_state)

        # --- Fit model ---
        model.fit(X_sample, y_sample)

        # --- Compute permutation importance ---
        perm_imp = permutation_importance(
            model,
            X_sample,
            y_sample,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1,  # use all cores
        )

        # --- Store results ---
        importances = pd.DataFrame(index=self.selected_features)
        importances["importance"] = perm_imp.importances_mean
        self.selected_features = importances[
            importances["importance"] > 0
        ].index.tolist()

        print(
            f"✅ Selected {len(self.selected_features)} features based on permutation importance."
        )
        return self.selected_features, importances
