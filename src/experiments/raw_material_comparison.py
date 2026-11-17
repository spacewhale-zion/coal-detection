import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

C_VALUES = [
    0.1,
    1,
    10
]


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
# CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# STORE PREDICTIONS
# ============================================================

predictions_by_c = {}


# ============================================================
# RUN EACH C
# ============================================================

for C in C_VALUES:

    print("\n")
    print("=" * 70)
    print(f"SVM C = {C}")
    print("=" * 70)

    predictions = np.full(
        len(y),
        -1
    )

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(
            X,
            y,
            groups=groups
        ),
        start=1
    ):

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        X_train = X[train_idx]
        X_val = X[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        # ----------------------------------------------------
        # STANDARDIZATION
        # ----------------------------------------------------

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_val = scaler.transform(
            X_val
        )

        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        model = SVC(
            kernel="rbf",
            C=C,
            gamma="scale"
        )

        model.fit(
            X_train_pca,
            y_train
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        y_pred = model.predict(
            X_val_pca
        )

        predictions[val_idx] = y_pred

        accuracy = accuracy_score(
            y_val,
            y_pred
        )

        print(
            f"Fold {fold}: "
            f"{accuracy:.4f}"
        )

    predictions_by_c[C] = predictions


# ============================================================
# MATERIAL ANALYSIS
# ============================================================

unique_materials = np.unique(
    groups
)


print("\n")
print("=" * 100)
print("PER-MATERIAL COMPARISON")
print("=" * 100)

print(
    "\nMaterial"
    " " * 35 +
    "Class     C=0.1     C=1       C=10"
)

print("-" * 100)


for material in unique_materials:

    mask = (
        groups == material
    )

    material_class = (
        "Coal"
        if y[mask][0] == 1
        else "Non-Coal"
    )

    scores = []

    for C in C_VALUES:

        y_true = y[mask]

        y_pred = predictions_by_c[C][mask]

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        scores.append(
            accuracy
        )

    print(
        f"{material:<40}"
        f"{material_class:<10}"
        f"{scores[0]:.4f}     "
        f"{scores[1]:.4f}     "
        f"{scores[2]:.4f}"
    )


# ============================================================
# OVERALL METRICS
# ============================================================

print("\n")
print("=" * 100)
print("OVERALL OUT-OF-FOLD COMPARISON")
print("=" * 100)


for C in C_VALUES:

    y_pred = predictions_by_c[C]

    accuracy = accuracy_score(
        y,
        y_pred
    )

    precision = precision_score(
        y,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0
    )

    print(
        f"\nC = {C}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )