import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

        # Fill NaN in numeric columns safely
        self.df[num_cols] = self.df[num_cols].apply(lambda col: col.fillna(col.mean()))

        # Drop remaining rows with NaN in any target or feature
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

    def select_pls_components(self, corr_threshold=0.1, pls_corr_threshold=0.5):
        """Select PLS components based on relevance to targets and intra-feature redundancy."""
        df = self.df.reset_index()
        pls_cols = self.selected_features
        corr_with_targets = (
            df[pls_cols + self.targets].corr().loc[pls_cols, self.targets].abs()
        )
        max_corr_per_pls = corr_with_targets.max(axis=1)

        # Step 1: Keep components with sufficient correlation with targets
        selected = max_corr_per_pls[max_corr_per_pls > corr_threshold].index.tolist()

        # Step 2: Remove highly correlated PLS components (redundancy)
        final_selected = selected.copy()
        pls_corr_matrix = df[selected].corr().abs()
        np.fill_diagonal(pls_corr_matrix.values, 0)

        for i, pls1 in enumerate(selected):
            for pls2 in selected[i + 1 :]:
                if pls1 not in final_selected or pls2 not in final_selected:
                    continue
                if pls_corr_matrix.loc[pls1, pls2] > pls_corr_threshold:
                    if max_corr_per_pls[pls1] >= max_corr_per_pls[pls2]:
                        final_selected.remove(pls2)
                    else:
                        final_selected.remove(pls1)

        self.selected_features = final_selected
        print(
            f"✅ Selected {len(final_selected)} PLS components after filtering and redundancy removal."
        )
        return self.selected_features

    def plot_mrmr_heatmaps(self):
        """Plot correlations among PLS features and with targets."""
        df_plot = self.df.reset_index()[self.selected_features + self.targets]

        # Correlation among PLS features
        pls_corr = df_plot[self.selected_features].corr()
        plt.figure(figsize=(2 * 10, 8 * 2))
        sns.heatmap(pls_corr, cmap="coolwarm", center=0, annot=False)
        plt.title("Correlation Among PLS Features")
        plt.show()

        # Correlation between PLS features and targets
        pls_target_corr = (
            df_plot[self.selected_features + self.targets]
            .corr()
            .loc[self.selected_features, self.targets]
        )
        plt.figure(figsize=(8, 10))
        sns.heatmap(pls_target_corr, cmap="coolwarm", center=0, annot=True)
        plt.title("Correlation Between PLS Features and Targets")
        plt.show()
