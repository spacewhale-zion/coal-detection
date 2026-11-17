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

PCA_COMPONENTS = [
    10,
    20,
    50,
    100,
    200,
    500
]

SVM_C = [
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


results = []


# ============================================================
# EXPERIMENT
# ============================================================

for n_components in PCA_COMPONENTS:

    for C in SVM_C:

        print("\n")
        print("=" * 70)

        print(
            f"PCA = {n_components}, "
            f"SVM C = {C}"
        )

        print("=" * 70)

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(
                X,
                y,
                groups=groups
            ),
            start=1
        ):

            # ------------------------------------------------
            # SPLIT
            # ------------------------------------------------

            X_train = X[train_idx]
            X_val = X[val_idx]

            y_train = y[train_idx]
            y_val = y[val_idx]

            # ------------------------------------------------
            # STANDARDIZATION
            # ------------------------------------------------

            scaler = StandardScaler()

            X_train = scaler.fit_transform(
                X_train
            )

            X_val = scaler.transform(
                X_val
            )

            # ------------------------------------------------
            # PCA
            # ------------------------------------------------

            pca = PCA(
                n_components=n_components,
                random_state=RANDOM_STATE
            )

            X_train_pca = pca.fit_transform(
                X_train
            )

            X_val_pca = pca.transform(
                X_val
            )

            # ------------------------------------------------
            # SVM
            # ------------------------------------------------

            model = SVC(
                kernel="rbf",
                C=C,
                gamma="scale"
            )

            model.fit(
                X_train_pca,
                y_train
            )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            y_pred = model.predict(
                X_val_pca
            )

            accuracy = accuracy_score(
                y_val,
                y_pred
            )

            precision = precision_score(
                y_val,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_val,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_val,
                y_pred,
                zero_division=0
            )

            fold_scores.append(
                [
                    accuracy,
                    precision,
                    recall,
                    f1
                ]
            )

            print(
                f"Fold {fold}: "
                f"Accuracy={accuracy:.4f}, "
                f"F1={f1:.4f}"
            )

        # ----------------------------------------------------
        # MEAN
        # ----------------------------------------------------

        fold_scores = np.array(
            fold_scores
        )

        mean_accuracy = fold_scores[:, 0].mean()
        std_accuracy = fold_scores[:, 0].std()

        mean_precision = fold_scores[:, 1].mean()
        mean_recall = fold_scores[:, 2].mean()
        mean_f1 = fold_scores[:, 3].mean()

        results.append(
            [
                n_components,
                C,
                mean_accuracy,
                std_accuracy,
                mean_precision,
                mean_recall,
                mean_f1
            ]
        )

        print(
            f"\nMean Accuracy: "
            f"{mean_accuracy:.4f}"
        )

        print(
            f"Mean F1: "
            f"{mean_f1:.4f}"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("RAW PCA + SVM CROSS-VALIDATION RESULTS")
print("=" * 100)

print(
    "\nPCA | C | Accuracy | Std | Precision | Recall | F1"
)

for result in results:

    (
        components,
        C,
        accuracy,
        std,
        precision,
        recall,
        f1
    ) = result

    print(
        f"{components:3d} | "
        f"{C:3.1f} | "
        f"{accuracy:.4f} | "
        f"{std:.4f} | "
        f"{precision:.4f} | "
        f"{recall:.4f} | "
        f"{f1:.4f}"
    )