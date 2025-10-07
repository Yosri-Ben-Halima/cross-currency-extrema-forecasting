import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.cross_decomposition import PLSRegression


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
        self.df = df.copy()
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

    def permutation_importance_selection(self, n_repeats=5, random_state=42):
        """
        Optional: Compute permutation importance per target and select top features.
        """
        print("🔹 Computing permutation importance across all targets...")
        importances = pd.DataFrame(index=self.selected_features)

        for target in self.targets:
            if target not in self.df.columns:
                continue

            y = self.df[target]
            X = self.df[self.selected_features]

            if target == "meta_label":
                model = RandomForestClassifier(
                    n_estimators=200, random_state=random_state
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=200, random_state=random_state
                )

            model.fit(X, y)
            perm_imp = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=random_state
            )
            importances[target] = perm_imp.importances_mean

        importances["mean_importance"] = importances.mean(axis=1)
        self.selected_features = importances[
            importances["mean_importance"] > 0
        ].index.tolist()

        print(
            f"✅ Selected {len(self.selected_features)} features based on permutation importance."
        )
        return self.selected_features, importances
