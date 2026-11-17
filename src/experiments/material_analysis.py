import numpy as np

from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.preprocessing.spectral import (
    snv,
    savitzky_golay
)


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


# ============================================================
# CONFIGURATION
# ============================================================

N_SPLITS = 5
RANDOM_STATE = 42

PCA_COMPONENTS = 20
SVM_C = 1


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(
    FINAL_SPLIT_DIR / "X_development.npy"
)

y = np.load(
    FINAL_SPLIT_DIR / "y_development.npy"
)

metadata = np.genfromtxt(
    FINAL_SPLIT_DIR / "metadata_development.csv",
    delimiter=",",
    dtype=str,
    skip_header=1
)

materials = metadata[:, 0]


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(X):

    X = snv(X)

    X = savitzky_golay(
        X,
        window_length=11,
        polyorder=2
    )

    return X


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Store predictions for every sample
all_predictions = np.full(
    len(y),
    -1
)


# ============================================================
# RUN CV
# ============================================================

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups=materials),
    start=1
):

    print(
        f"\nRunning Fold {fold}..."
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_train = X[train_idx]
    X_val = X[val_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    X_train = preprocess(X_train)
    X_val = preprocess(X_val)

    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std == 0] = 1

    X_train = (
        X_train - mean
    ) / std

    X_val = (
        X_val - mean
    ) / std

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=RANDOM_STATE
    )

    X_train_pca = pca.fit_transform(
        X_train
    )

    X_val_pca = pca.transform(
        X_val
    )

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    model = SVC(
        kernel="rbf",
        C=SVM_C,
        gamma="scale"
    )

    model.fit(
        X_train_pca,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_val_pca
    )

    all_predictions[val_idx] = predictions


# ============================================================
# VERIFY
# ============================================================

if np.any(all_predictions == -1):

    raise RuntimeError(
        "Some samples were not assigned predictions."
    )


# ============================================================
# MATERIAL-WISE ANALYSIS
# ============================================================

print("\n")
print("=" * 90)
print("PER-MATERIAL CROSS-VALIDATION PERFORMANCE")
print("=" * 90)

unique_materials = np.unique(
    materials
)


for material in unique_materials:

    mask = (
        materials == material
    )

    y_true = y[mask]
    y_pred = all_predictions[mask]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    label = (
        "Coal"
        if y_true[0] == 1
        else "Non-Coal"
    )

    print(
        f"\n{material}"
    )

    print(
        f"  Class     : {label}"
    )

    print(
        f"  Samples   : {len(y_true)}"
    )

    print(
        f"  Accuracy  : {accuracy:.4f}"
    )

    print(
        f"  Precision : {precision:.4f}"
    )

    print(
        f"  Recall    : {recall:.4f}"
    )

    print(
        f"  F1        : {f1:.4f}"
    )


# ============================================================
# OVERALL
# ============================================================

overall_accuracy = accuracy_score(
    y,
    all_predictions
)

overall_precision = precision_score(
    y,
    all_predictions,
    zero_division=0
)

overall_recall = recall_score(
    y,
    all_predictions,
    zero_division=0
)

overall_f1 = f1_score(
    y,
    all_predictions,
    zero_division=0
)


print("\n")
print("=" * 90)
print("OVERALL OUT-OF-FOLD PERFORMANCE")
print("=" * 90)

print(
    f"Accuracy  : {overall_accuracy:.4f}"
)

print(
    f"Precision : {overall_precision:.4f}"
)

print(
    f"Recall    : {overall_recall:.4f}"
)

print(
    f"F1        : {overall_f1:.4f}"
)