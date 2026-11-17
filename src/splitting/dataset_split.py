import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PROJECT PATHS
# ============================================================

# dataset_split.py
#       ↓
# splitting/
#       ↓
# src/
#       ↓
# coal-detection/

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SPLIT_DIR = (
    PROCESSED_DIR
    / "split"
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X_path = PROCESSED_DIR / "X.npy"
    y_path = PROCESSED_DIR / "y.npy"
    metadata_path = PROCESSED_DIR / "metadata.csv"

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not X_path.exists():

        raise FileNotFoundError(
            f"X.npy not found:\n{X_path}\n\n"
            "Run prepare_dataset.py first."
        )

    if not y_path.exists():

        raise FileNotFoundError(
            f"y.npy not found:\n{y_path}\n\n"
            "Run prepare_dataset.py first."
        )

    if not metadata_path.exists():

        raise FileNotFoundError(
            f"metadata.csv not found:\n{metadata_path}\n\n"
            "Run prepare_dataset.py first."
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X = np.load(X_path)
    y = np.load(y_path)

    metadata = pd.read_csv(
        metadata_path
    )

    return X, y, metadata


# ============================================================
# VALIDATE DATA
# ============================================================

def validate_data(
    X,
    y,
    metadata
):

    # --------------------------------------------------------
    # Number of samples
    # --------------------------------------------------------

    if len(X) != len(y):

        raise ValueError(
            "X and y have different numbers "
            "of samples."
        )

    if len(X) != len(metadata):

        raise ValueError(
            "X and metadata have different "
            "numbers of samples."
        )

    # --------------------------------------------------------
    # Required column
    # --------------------------------------------------------

    if "material" not in metadata.columns:

        raise ValueError(
            "metadata.csv must contain "
            "a 'material' column."
        )

    if "label" not in metadata.columns:

        raise ValueError(
            "metadata.csv must contain "
            "a 'label' column."
        )


# ============================================================
# GROUP-AWARE TRAIN/TEST SPLIT
# ============================================================

def split_dataset(
    X,
    y,
    metadata
):

    groups = metadata["material"].values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    metadata_train = metadata.iloc[
        train_indices
    ].copy()

    metadata_test = metadata.iloc[
        test_indices
    ].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test
    )


# ============================================================
# PRINT SPLIT INFORMATION
# ============================================================

def print_split_information(
    X_train,
    X_test,
    y_train,
    y_test,
    metadata_train,
    metadata_test
):

    print("\n")
    print("=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    print("\nDataset shapes")
    print("-" * 70)

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
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution")
    print("-" * 70)

    print(
        f"Train Coal     : {np.sum(y_train == 1)}"
    )

    print(
        f"Train Non-Coal : {np.sum(y_train == 0)}"
    )

    print(
        f"Test Coal      : {np.sum(y_test == 1)}"
    )

    print(
        f"Test Non-Coal  : {np.sum(y_test == 0)}"
    )

    # --------------------------------------------------------
    # Material groups
    # --------------------------------------------------------

    train_materials = set(
        metadata_train["material"]
    )

    test_materials = set(
        metadata_test["material"]
    )

    overlap = (
        train_materials
        & test_materials
    )

    print("\nMaterial groups")
    print("-" * 70)

    print(
        f"Training materials : "
        f"{len(train_materials)}"
    )

    print(
        f"Testing materials  : "
        f"{len(test_materials)}"
    )

    print(
        f"Material overlap   : "
        f"{len(overlap)}"
    )

    if len(overlap) == 0:

        print(
            "\n✓ No material leakage detected."
        )

    else:

        print(
            "\nWARNING: Material leakage detected!"
        )

        print(
            "Overlapping materials:",
            overlap
        )


# ============================================================
# SAVE SPLIT
# ============================================================

def save_split(
    X_train,
    X_test,
    y_train,
    y_test,
    metadata_train,
    metadata_test
):

    # Create directory

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save numpy arrays
    # --------------------------------------------------------

    np.save(
        SPLIT_DIR / "X_train.npy",
        X_train
    )

    np.save(
        SPLIT_DIR / "X_test.npy",
        X_test
    )

    np.save(
        SPLIT_DIR / "y_train.npy",
        y_train
    )

    np.save(
        SPLIT_DIR / "y_test.npy",
        y_test
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_train.to_csv(
        SPLIT_DIR / "metadata_train.csv",
        index=False
    )

    metadata_test.to_csv(
        SPLIT_DIR / "metadata_test.csv",
        index=False
    )

    print("\n")
    print("=" * 70)
    print("SPLIT SAVED")
    print("=" * 70)

    print(
        f"\nSaved to:\n{SPLIT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("GROUP-AWARE DATASET SPLITTING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y, metadata = load_data()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_data(
        X,
        y,
        metadata
    )

    print(
        f"\nTotal samples: {len(X)}"
    )

    print(
        f"Total materials: "
        f"{metadata['material'].nunique()}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test
    ) = split_dataset(
        X,
        y,
        metadata
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print_split_information(
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_split(
        X_train,
        X_test,
        y_train,
        y_test,
        metadata_train,
        metadata_test
    )