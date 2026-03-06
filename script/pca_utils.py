import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def corr_heatmap(df, cols, title):

    cols = [c for c in cols if c in df.columns]

    if len(cols) < 2:
        print("Not enough variables for heatmap:", cols)
        return

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 3, 0.6 * len(cols) + 3))

    im = ax.imshow(corr.values, vmin=-1, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))

    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    ax.set_title(title)

    for i in range(len(cols)):
        for j in range(len(cols)):

            value = corr.values[i, j]

            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax)

    plt.tight_layout()

    plt.show()


def add_pca_index(df, cols, new_col):

    cols = [c for c in cols if c in df.columns]

    if len(cols) < 2:
        print("Not enough variables for PCA:", cols)
        return df

    X = df[cols].apply(pd.to_numeric, errors="coerce")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=1))
    ])

    component = pipeline.fit_transform(X).reshape(-1)

    df[new_col] = component

    explained_var = pipeline.named_steps["pca"].explained_variance_ratio_[0]

    print(new_col, "explained variance ratio:", explained_var)

    loadings = pipeline.named_steps["pca"].components_[0]

    loading_series = pd.Series(loadings, index=cols)

    print("Loadings for", new_col)

    print(loading_series.sort_values(key=lambda x: abs(x), ascending=False))

    print()

    return df