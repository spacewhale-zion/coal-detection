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

PCA_COMPONENTS = [20, 50, 100]

SVM_C = [1, 10, 100]


# ============================================================
# LOAD DEVELOPMENT DATA
# ============================================================

def load_development_data():

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

    # Material column
    materials = metadata[:, 0]

    print("=" * 70)
    print("DEVELOPMENT DATA")
    print("=" * 70)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Materials: {len(np.unique(materials))}")

    return X, y, materials


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(X):

    # SNV
    X = snv(X)

    # Savitzky-Golay
    X = savitzky_golay(
        X,
        window_length=11,
        polyorder=2
    )

    return X


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_experiment():

    X, y, groups = load_development_data()

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = []

    print("\n")
    print("=" * 70)
    print("STRATIFIED GROUP K-FOLD CROSS VALIDATION")
    print("=" * 70)

    for n_components in PCA_COMPONENTS:

        for C in SVM_C:

            fold_scores = []

            print(
                f"\nPCA = {n_components}, "
                f"SVM C = {C}"
            )

            for fold, (train_idx, val_idx) in enumerate(
                cv.split(X, y, groups),
                start=1
            ):

                X_train = X[train_idx]
                X_val = X[val_idx]

                y_train = y[train_idx]
                y_val = y[val_idx]

                # ------------------------------------------------
                # PREPROCESS
                # ------------------------------------------------

                X_train = preprocess(X_train)
                X_val = preprocess(X_val)

                # ------------------------------------------------
                # STANDARDIZATION
                # ------------------------------------------------

                train_mean = X_train.mean(axis=0)
                train_std = X_train.std(axis=0)

                train_std[train_std == 0] = 1

                X_train = (
                    X_train - train_mean
                ) / train_std

                X_val = (
                    X_val - train_mean
                ) / train_std

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
                # VALIDATION
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
                    (
                        accuracy,
                        precision,
                        recall,
                        f1
                    )
                )

                print(
                    f"Fold {fold}: "
                    f"Accuracy={accuracy:.4f}, "
                    f"F1={f1:.4f}"
                )

            # ----------------------------------------------------
            # AVERAGE
            # ----------------------------------------------------

            fold_scores = np.array(
                fold_scores
            )

            mean_accuracy = fold_scores[:, 0].mean()
            mean_precision = fold_scores[:, 1].mean()
            mean_recall = fold_scores[:, 2].mean()
            mean_f1 = fold_scores[:, 3].mean()

            std_accuracy = fold_scores[:, 0].std()

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
                f"Mean Accuracy: "
                f"{mean_accuracy:.4f} "
                f"+/- {std_accuracy:.4f}"
            )

            print(
                f"Mean F1: "
                f"{mean_f1:.4f}"
            )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results = run_experiment()

    print("\n")
    print("=" * 70)
    print("FINAL CROSS-VALIDATION RESULTS")
    print("=" * 70)

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
            f"{C:3d} | "
            f"{accuracy:.4f} | "
            f"{std:.4f} | "
            f"{precision:.4f} | "
            f"{recall:.4f} | "
            f"{f1:.4f}"
        )