import numpy as np
from pathlib import Path
from sklearn.svm import SVC

from src.evaluation.metrics import (
    print_metrics,
    plot_confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PCA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "pca"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "saved"
)


# ============================================================
# LOAD PCA DATA
# ============================================================

def load_data():

    X_train = np.load(
        PCA_DIR / "X_train_pca.npy"
    )

    X_test = np.load(
        PCA_DIR / "X_test_pca.npy"
    )

    y_train = np.load(
        PCA_DIR / "y_train.npy"
    )

    y_test = np.load(
        PCA_DIR / "y_test.npy"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# CREATE SVM
# ============================================================

def create_model():

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    )

    return model


# ============================================================
# TRAIN
# ============================================================

def train_model(
    model,
    X_train,
    y_train
):

    print("\nTraining PCA-SVM...")

    model.fit(
        X_train,
        y_train
    )

    print(
        "PCA-SVM training completed."
    )

    return model


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    import joblib

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        MODEL_DIR
        / "svm_pca_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nModel saved to:\n{model_path}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PCA + SVM — COAL vs NON-COAL")
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

    print("\nDataset")

    print(
        f"X_train : {X_train.shape}"
    )

    print(
        f"X_test  : {X_test.shape}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model()

    print("\nModel")
    print("-" * 70)

    print(model)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        model,
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print_metrics(
        y_test,
        predictions
    )

    plot_confusion_matrix(
        y_test,
        predictions
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_model(
        model
    )