import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_split"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(
    FINAL_SPLIT_DIR / "X_development.npy"
)

y = np.load(
    FINAL_SPLIT_DIR / "y_development.npy"
)

metadata = pd.read_csv(
    FINAL_SPLIT_DIR / "metadata_development.csv"
)

groups = metadata["material"].values


# ============================================================
# USE FIRST CV FOLD
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

train_idx, val_idx = next(
    cv.split(
        X,
        y,
        groups=groups
    )
)


X_train = X[train_idx]


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)


# ============================================================
# PCA
# ============================================================

pca = PCA()

pca.fit(
    X_train
)


explained_variance = (
    pca.explained_variance_ratio_
)

cumulative_variance = np.cumsum(
    explained_variance
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 70)
print("PCA EXPLAINED VARIANCE ANALYSIS")
print("=" * 70)

print(
    f"\nOriginal features: {X.shape[1]}"
)

print(
    f"Total PCA components: "
    f"{len(explained_variance)}"
)


for n in [
    1,
    2,
    5,
    10,
    20,
    30,
    50,
    100,
    200,
    500
]:

    variance = cumulative_variance[
        n - 1
    ]

    print(
        f"{n:4d} components → "
        f"{variance * 100:.4f}% variance"
    )


# ============================================================
# FIND COMPONENTS FOR TARGET VARIANCE
# ============================================================

for target in [
    0.90,
    0.95,
    0.99,
    0.999
]:

    components = (
        np.argmax(
            cumulative_variance >= target
        ) + 1
    )

    actual = cumulative_variance[
        components - 1
    ]

    print(
        f"\n{target * 100:.1f}% variance:"
    )

    print(
        f"  Components = {components}"
    )

    print(
        f"  Actual      = "
        f"{actual * 100:.4f}%"
    )


# ============================================================
# PLOT CUMULATIVE VARIANCE
# ============================================================

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
    0.90,
    linestyle="--",
    label="90%"
)

plt.axhline(
    0.95,
    linestyle="--",
    label="95%"
)

plt.axhline(
    0.99,
    linestyle="--",
    label="99%"
)

plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "PCA Cumulative Explained Variance"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

output_path = (
    FIGURES_DIR
    / "pca_cumulative_variance.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.show()

print(
    f"\nSaved figure to:\n{output_path}"
)