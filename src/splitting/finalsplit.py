import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FINAL_SPLIT_DIR = (
    PROCESSED_DIR
    / "final_split"
)


# ============================================================
# CONFIGURATION
# ============================================================

FINAL_TEST_SIZE = 0.20

RANDOM_STATE = 123


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

def load_data():

    X = np.load(
        PROCESSED_DIR / "X.npy"
    )

    y = np.load(
        PROCESSED_DIR / "y.npy"
    )

    metadata = pd.read_csv(
        PROCESSED_DIR / "metadata.csv"
    )

    return X, y, metadata


# ============================================================
# VALIDATE
# ============================================================

def validate_data(
    X,
    y,
    metadata
):

    if len(X) != len(y):

        raise ValueError(
            "X and y have different "
            "numbers of samples."
        )

    if len(X) != len(metadata):

        raise ValueError(
            "X and metadata have different "
            "numbers of samples."
        )

    if "material" not in metadata.columns:

        raise ValueError(
            "'material' column missing "
            "from metadata.csv."
        )


# ============================================================
# GROUP-AWARE SPLIT
# ============================================================

def create_final_split(
    X,
    y,
    metadata
):

    groups = metadata[
        "material"
    ].values

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=FINAL_TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        splitter.split(
            X,
            y,
            groups=groups
        )
    )

    X_development = X[
        train_indices
    ]

    X_final_test = X[
        test_indices
    ]

    y_development = y[
        train_indices
    ]

    y_final_test = y[
        test_indices
    ]

    metadata_development = (
        metadata
        .iloc[train_indices]
        .copy()
    )

    metadata_final_test = (
        metadata
        .iloc[test_indices]
        .copy()
    )

    return (
        X_development,
        X_final_test,
        y_development,
        y_final_test,
        metadata_development,
        metadata_final_test
    )


# ============================================================
# CHECK FOR GROUP LEAKAGE
# ============================================================

def check_leakage(
    metadata_development,
    metadata_final_test
):

    development_groups = set(
        metadata_development[
            "material"
        ]
    )

    test_groups = set(
        metadata_final_test[
            "material"
        ]
    )

    overlap = (
        development_groups
        & test_groups
    )

    print("\nGroup leakage check")
    print("-" * 70)

    print(
        "Development materials:",
        len(development_groups)
    )

    print(
        "Final-test materials:",
        len(test_groups)
    )

    print(
        "Overlapping materials:",
        len(overlap)
    )

    if len(overlap) == 0:

        print(
            "✓ No material leakage."
        )

    else:

        print(
            "WARNING: leakage detected!"
        )

        print(
            overlap
        )


# ============================================================
# PRINT INFORMATION
# ============================================================

def print_information(
    X_development,
    X_final_test,
    y_development,
    y_final_test
):

    print("\n")
    print("=" * 70)
    print("FINAL HOLDOUT SPLIT")
    print("=" * 70)

    print("\nShapes")
    print("-" * 70)

    print(
        "Development X:",
        X_development.shape
    )

    print(
        "Final Test X :",
        X_final_test.shape
    )

    print(
        "Development y:",
        y_development.shape
    )

    print(
        "Final Test y :",
        y_final_test.shape
    )

    print("\nClass distribution")
    print("-" * 70)

    print(
        "Development Coal:",
        np.sum(y_development == 1)
    )

    print(
        "Development Non-Coal:",
        np.sum(y_development == 0)
    )

    print(
        "Final Test Coal:",
        np.sum(y_final_test == 1)
    )

    print(
        "Final Test Non-Coal:",
        np.sum(y_final_test == 0)
    )


# ============================================================
# SAVE
# ============================================================

def save_data(
    X_development,
    X_final_test,
    y_development,
    y_final_test,
    metadata_development,
    metadata_final_test
):

    FINAL_SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        FINAL_SPLIT_DIR
        / "X_development.npy",
        X_development
    )

    np.save(
        FINAL_SPLIT_DIR
        / "X_final_test.npy",
        X_final_test
    )

    np.save(
        FINAL_SPLIT_DIR
        / "y_development.npy",
        y_development
    )

    np.save(
        FINAL_SPLIT_DIR
        / "y_final_test.npy",
        y_final_test
    )

    metadata_development.to_csv(
        FINAL_SPLIT_DIR
        / "metadata_development.csv",
        index=False
    )

    metadata_final_test.to_csv(
        FINAL_SPLIT_DIR
        / "metadata_final_test.csv",
        index=False
    )

    print("\nSaved to:")
    print(FINAL_SPLIT_DIR)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("CREATING FINAL HOLDOUT DATASET")
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
        "\nTotal samples:",
        len(X)
    )

    print(
        "Total materials:",
        metadata["material"].nunique()
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_development,
        X_final_test,
        y_development,
        y_final_test,
        metadata_development,
        metadata_final_test
    ) = create_final_split(
        X,
        y,
        metadata
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_information(
        X_development,
        X_final_test,
        y_development,
        y_final_test
    )

    # --------------------------------------------------------
    # Leakage
    # --------------------------------------------------------

    check_leakage(
        metadata_development,
        metadata_final_test
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_data(
        X_development,
        X_final_test,
        y_development,
        y_final_test,
        metadata_development,
        metadata_final_test
    )

    print("\n")
    print("=" * 70)
    print("FINAL HOLDOUT CREATED")
    print("=" * 70)