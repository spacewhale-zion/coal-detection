import numpy as np
from pathlib import Path

from spectral import (
    snv,
    savitzky_golay,
    standardize
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "split"
)

PREPROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed"
)


# ============================================================
# LOAD SPLIT DATA
# ============================================================

def load_split_data():

    X_train = np.load(
        SPLIT_DIR / "X_train.npy"
    )

    X_test = np.load(
        SPLIT_DIR / "X_test.npy"
    )

    y_train = np.load(
        SPLIT_DIR / "y_train.npy"
    )

    y_test = np.load(
        SPLIT_DIR / "y_test.npy"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_training_data(
    X_train,
    X_test
):

    # --------------------------------------------------------
    # Step 1: SNV
    # --------------------------------------------------------

    print("Applying SNV...")

    X_train_snv = snv(
        X_train
    )

    X_test_snv = snv(
        X_test
    )

    # --------------------------------------------------------
    # Step 2: Savitzky-Golay
    # --------------------------------------------------------

    print(
        "Applying Savitzky-Golay..."
    )

    X_train_smooth = savitzky_golay(
        X_train_snv,
        window_length=11,
        polyorder=2
    )

    X_test_smooth = savitzky_golay(
        X_test_snv,
        window_length=11,
        polyorder=2
    )

    # --------------------------------------------------------
    # Step 3: Standardization
    # --------------------------------------------------------

    print(
        "Applying StandardScaler..."
    )

    (
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = standardize(
        X_train_smooth,
        X_test_smooth
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )


# ============================================================
# SAVE DATA
# ============================================================

def save_preprocessed_data(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler
):

    PREPROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        PREPROCESSED_DIR / "X_train.npy",
        X_train
    )

    np.save(
        PREPROCESSED_DIR / "X_test.npy",
        X_test
    )

    np.save(
        PREPROCESSED_DIR / "y_train.npy",
        y_train
    )

    np.save(
        PREPROCESSED_DIR / "y_test.npy",
        y_test
    )

    # Save scaler parameters
    np.save(
        PREPROCESSED_DIR / "scaler_mean.npy",
        scaler.mean_
    )

    np.save(
        PREPROCESSED_DIR / "scaler_scale.npy",
        scaler.scale_
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PREPARING TRAINING DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_split_data()

    print("\nOriginal data")

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    (
        X_train_processed,
        X_test_processed,
        scaler
    ) = preprocess_training_data(
        X_train,
        X_test
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nProcessed data")

    print(
        "X_train:",
        X_train_processed.shape
    )

    print(
        "X_test :",
        X_test_processed.shape
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_preprocessed_data(
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        scaler
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"\nSaved to:\n{PREPROCESSED_DIR}"
    )