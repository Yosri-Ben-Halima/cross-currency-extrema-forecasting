import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from utils.helpers import downcast_numeric_columns


class FeatureSelector:
    """
    Performs feature orthogonalization and selection for multi-target ML.
    Orthogonalization is based on the mid-target: (y_high + y_low)/2
    """

    def __init__(
        self, df: pd.DataFrame, feature_cols=None, targets=None, n_pca_components=None
    ):
        """
        Args:
            df (pd.DataFrame): DataFrame containing features + targets
            feature_cols (list, optional): list of feature column names. Defaults to all non-target columns
            targets (list, optional): ['y_high', 'y_low', 'meta_label']
            n_pca_components (int, optional): number of PCA components to keep
        """
        num_cols = df.select_dtypes(include=[np.number]).columns
        self.df = df.copy()

        self.df[num_cols] = self.df[num_cols].apply(lambda col: col.fillna(col.mean()))

        self.df = self.df.dropna(subset=targets + feature_cols).set_index(
            ["open_time", "currency"]
        )
        self.targets = targets or ["y_high", "y_low"]
        self.feature_cols = feature_cols or [
            c
            for c in self.df.columns
            if c not in self.targets + ["open", "high", "low"]
        ]
        self.n_pca_components = n_pca_components or len(self.feature_cols)
        self.pls_model = None
        self.selected_features = None

    def orthogonalize_features(self):
        """Orthogonalize features using PLSRegression on mid-target."""
        print("🔹 Performing PLS-based orthogonalization...")
        mid_target = (self.df[self.targets[0]] + self.df[self.targets[1]]) / 2
        y = mid_target.values.reshape(-1, 1)
        X = self.df[self.feature_cols].values

        self.pls_model = PLSRegression(n_components=self.n_pca_components)
        X_pls, _ = self.pls_model.fit_transform(X, y)

        pls_cols = [f"pls_{i}" for i in range(X_pls.shape[1])]
        self.df[pls_cols] = X_pls
        self.selected_features = pls_cols
        print(
            f"✅ Orthogonalized {len(self.feature_cols)} features into {len(pls_cols)} PLS components."
        )
        self.df = downcast_numeric_columns(self.df)
        return self.df  # df_orthogonal
