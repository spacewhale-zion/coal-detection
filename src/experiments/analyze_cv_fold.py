import numpy as np

from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold


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
# CROSS VALIDATION
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


print("=" * 70)
print("CV FOLD MATERIAL ANALYSIS")
print("=" * 70)

print(
    f"\nTotal samples   : {len(X)}"
)

print(
    f"Total materials : {len(np.unique(materials))}"
)


for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups=materials),
    start=1
):

    train_materials = np.unique(
        materials[train_idx]
    )

    val_materials = np.unique(
        materials[val_idx]
    )

    train_coal = np.sum(
        y[train_idx] == 1
    )

    train_noncoal = np.sum(
        y[train_idx] == 0
    )

    val_coal = np.sum(
        y[val_idx] == 1
    )

    val_noncoal = np.sum(
        y[val_idx] == 0
    )

    print("\n")
    print("=" * 70)
    print(f"FOLD {fold}")
    print("=" * 70)

    print("\nTraining materials:")
    for material in train_materials:
        print(f"  {material}")

    print("\nValidation materials:")
    for material in val_materials:
        print(f"  {material}")

    print("\nTraining distribution:")
    print(
        f"  Coal     : {train_coal}"
    )
    print(
        f"  Non-Coal : {train_noncoal}"
    )

    print("\nValidation distribution:")
    print(
        f"  Coal     : {val_coal}"
    )
    print(
        f"  Non-Coal : {val_noncoal}"
    )

    print("\nValidation size:")
    print(
        f"  {len(val_idx)}"
    )