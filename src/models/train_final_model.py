import json
import joblib
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "final_split"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DEVELOPMENT DATA
# ============================================================

X_train = np.load(
    FINAL_SPLIT_DIR / "X_development.npy"
)

y_train = np.load(
    FINAL_SPLIT_DIR / "y_development.npy"
)

print("=" * 60)
print("FINAL MODEL TRAINING")
print("=" * 60)

print(f"Development samples : {X_train.shape[0]}")
print(f"Original features   : {X_train.shape[1]}")
print(f"Coal samples        : {np.sum(y_train == 1)}")
print(f"Non-Coal samples    : {np.sum(y_train == 0)}")


# ============================================================
# 1. STANDARDIZATION
# ============================================================

print("\n[1/3] Fitting StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

print("StandardScaler fitted.")


# ============================================================
# 2. PCA
# ============================================================

print("\n[2/3] Fitting PCA...")

pca = PCA(
    n_components=2,
    random_state=42
)

X_train_pca = pca.fit_transform(X_train_scaled)

print(f"PCA components     : {pca.n_components_}")
print(
    "Variance explained  : "
    f"{pca.explained_variance_ratio_.sum() * 100:.4f}%"
)

print(f"Transformed shape   : {X_train_pca.shape}")


# ============================================================
# 3. SVM
# ============================================================

print("\n[3/3] Training SVM...")

svm = SVC(
    kernel="rbf",
    C=1,
    gamma="scale"
)

svm.fit(X_train_pca, y_train)

print("SVM training completed.")


# ============================================================
# SAVE MODEL COMPONENTS
# ============================================================

print("\nSaving model artifacts...")

joblib.dump(
    scaler,
    MODEL_DIR / "scaler.pkl"
)

joblib.dump(
    pca,
    MODEL_DIR / "pca.pkl"
)

joblib.dump(
    svm,
    MODEL_DIR / "svm.pkl"
)

print(f"Saved: {MODEL_DIR / 'scaler.pkl'}")
print(f"Saved: {MODEL_DIR / 'pca.pkl'}")
print(f"Saved: {MODEL_DIR / 'svm.pkl'}")


# ============================================================
# SAVE MODEL CONFIGURATION
# ============================================================

config = {
    "preprocessing": {
        "standard_scaler": True,
        "snv": False,
        "savitzky_golay": False
    },
    "pca": {
        "n_components": 2
    },
    "model": {
        "type": "SVC",
        "kernel": "rbf",
        "C": 1,
        "gamma": "scale"
    },
    "training_samples": int(X_train.shape[0]),
    "original_features": int(X_train.shape[1]),
    "pca_features": int(X_train_pca.shape[1]),
    "explained_variance": float(
        pca.explained_variance_ratio_.sum()
    )
}

with open(
    METRICS_DIR / "final_model_config.json",
    "w"
) as f:
    json.dump(config, f, indent=4)


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL READY")
print("=" * 60)

print("Pipeline:")
print("Raw Spectrum")
print("    ↓")
print("StandardScaler")
print("    ↓")
print("PCA (2 components)")
print("    ↓")
print("RBF SVM (C=1)")
print("    ↓")
print("Coal / Non-Coal")

print("\nFinal test set was NOT used.")
print("Model artifacts saved successfully.")