import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.decomposition import PCA


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed"
)

PCA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pca"
)


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

def load_data():

    X_train = np.load(
        PREPROCESSED_DIR / "X_train.npy"
    )

    X_test = np.load(
        PREPROCESSED_DIR / "X_test.npy"
    )

    y_train = np.load(
        PREPROCESSED_DIR / "y_train.npy"
    )

    y_test = np.load(
        PREPROCESSED_DIR / "y_test.npy"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# FIT PCA
# ============================================================

def fit_pca(
    X_train,
    variance_ratio=0.99
):
    """
    Fit PCA using ONLY training data.

    variance_ratio:
        Amount of variance to retain.

        Example:
        0.99 = retain 99% of variance.
    """

    print(
        f"\nFitting PCA "
        f"with {variance_ratio * 100:.0f}% "
        f"variance retention..."
    )

    pca = PCA(
        n_components=variance_ratio,
        svd_solver="full"
    )

    pca.fit(X_train)

    print(
        f"Original features : {X_train.shape[1]}"
    )

    print(
        f"PCA components    : "
        f"{pca.n_components_}"
    )

    print(
        f"Variance retained : "
        f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
    )

    return pca


# ============================================================
# TRANSFORM DATA
# ============================================================

def transform_data(
    pca,
    X_train,
    X_test
):

    X_train_pca = pca.transform(
        X_train
    )

    X_test_pca = pca.transform(
        X_test
    )

    return (
        X_train_pca,
        X_test_pca
    )


# ============================================================
# SAVE PCA DATA
# ============================================================

def save_data(
    X_train_pca,
    X_test_pca,
    y_train,
    y_test,
    pca
):

    PCA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        PCA_DIR / "X_train_pca.npy",
        X_train_pca
    )

    np.save(
        PCA_DIR / "X_test_pca.npy",
        X_test_pca
    )

    np.save(
        PCA_DIR / "y_train.npy",
        y_train
    )

    np.save(
        PCA_DIR / "y_test.npy",
        y_test
    )

    # Save PCA information
    np.save(
        PCA_DIR / "explained_variance_ratio.npy",
        pca.explained_variance_ratio_
    )

    np.save(
        PCA_DIR / "components.npy",
        pca.components_
    )

    print(
        f"\nPCA data saved to:\n{PCA_DIR}"
    )


# ============================================================
# PLOT EXPLAINED VARIANCE
# ============================================================

def plot_explained_variance(
    pca
):

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        range(
            1,
            len(cumulative_variance) + 1
        ),
        cumulative_variance
    )

    plt.axhline(
        y=0.95,
        linestyle="--",
        label="95% variance"
    )

    plt.axhline(
        y=0.99,
        linestyle="--",
        label="99% variance"
    )

    plt.xlabel(
        "Number of Principal Components"
    )

    plt.ylabel(
        "Cumulative Explained Variance"
    )

    plt.title(
        "PCA Explained Variance"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PCA — DIMENSIONALITY REDUCTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data()

    print("\nOriginal data")

    print(
        f"X_train : {X_train.shape}"
    )

    print(
        f"X_test  : {X_test.shape}"
    )

    # --------------------------------------------------------
    # Fit PCA ONLY on training data
    # --------------------------------------------------------

    pca = fit_pca(
        X_train,
        variance_ratio=0.99
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    (
        X_train_pca,
        X_test_pca
    ) = transform_data(
        pca,
        X_train,
        X_test
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nAfter PCA")

    print(
        f"X_train : {X_train_pca.shape}"
    )

    print(
        f"X_test  : {X_test_pca.shape}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_data(
        X_train_pca,
        X_test_pca,
        y_train,
        y_test,
        pca
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_explained_variance(
        pca
    )

    print("\n" + "=" * 70)
    print("PCA COMPLETE")
    print("=" * 70)