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

PREPROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "saved"
)


# ============================================================
# LOAD DATA
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
# CREATE SVM MODEL
# ============================================================

def create_model():

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    X_train,
    y_train
):

    print("\nTraining SVM...")

    model.fit(
        X_train,
        y_train
    )

    print("SVM training completed.")

    return model


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    X_test
):

    predictions = model.predict(
        X_test
    )

    return predictions


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
        / "svm_model.pkl"
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
    print("SVM — COAL vs NON-COAL CLASSIFICATION")
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

    print(
        f"y_train : {y_train.shape}"
    )

    print(
        f"y_test  : {y_test.shape}"
    )

    # --------------------------------------------------------
    # Create model
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

    predictions = predict(
        model,
        X_test
    )

    # --------------------------------------------------------
    # Basic accuracy
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

    save_model(model)