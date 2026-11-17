import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_split"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "saved"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)

PREDICTIONS_DIR = (
    PROJECT_ROOT
    / "results"
    / "predictions"
)

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FINAL TEST DATA
# ============================================================

X_test = np.load(
    FINAL_SPLIT_DIR / "X_final_test.npy"
)

y_test = np.load(
    FINAL_SPLIT_DIR / "y_final_test.npy"
)

metadata_test = pd.read_csv(
    FINAL_SPLIT_DIR / "metadata_final_test.csv"
)


print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print(f"Final test samples : {X_test.shape[0]}")
print(f"Features           : {X_test.shape[1]}")
print(f"Coal samples       : {np.sum(y_test == 1)}")
print(f"Non-Coal samples   : {np.sum(y_test == 0)}")


# ============================================================
# LOAD TRAINED ARTIFACTS
# ============================================================

print("\nLoading trained model...")

scaler = joblib.load(
    MODEL_DIR / "scaler.pkl"
)

pca = joblib.load(
    MODEL_DIR / "pca.pkl"
)

svm = joblib.load(
    MODEL_DIR / "svm.pkl"
)

print("Scaler loaded.")
print("PCA loaded.")
print("SVM loaded.")


# ============================================================
# TRANSFORM FINAL TEST DATA
# ============================================================

print("\nTransforming final test data...")

# IMPORTANT:
# No fit() is performed here.

X_test_scaled = scaler.transform(X_test)

X_test_pca = pca.transform(X_test_scaled)

print(f"After scaling : {X_test_scaled.shape}")
print(f"After PCA     : {X_test_pca.shape}")


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = svm.predict(X_test_pca)

print("Predictions generated.")


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nDetailed:")
print(f"True Negative  : {tn}")
print(f"False Positive : {fp}")
print(f"False Negative : {fn}")
print(f"True Positive  : {tp}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Non-Coal",
        "Coal"
    ],
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),

    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    },

    "test_samples": int(len(y_test)),
    "coal_samples": int(np.sum(y_test == 1)),
    "non_coal_samples": int(np.sum(y_test == 0)),

    "pca_components": int(
        pca.n_components_
    ),

    "pca_variance_explained": float(
        pca.explained_variance_ratio_.sum()
    )
}

with open(
    METRICS_DIR / "final_test_metrics.json",
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Non-Coal",
        "Coal"
    ],
    output_dict=True,
    zero_division=0
)

with open(
    METRICS_DIR / "final_classification_report.json",
    "w"
) as f:
    json.dump(
        report_dict,
        f,
        indent=4
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = metadata_test.copy()

predictions["true_label"] = y_test
predictions["predicted_label"] = y_pred

predictions["correct"] = (
    predictions["true_label"]
    == predictions["predicted_label"]
)

predictions["true_class"] = predictions[
    "true_label"
].map({
    0: "Non-Coal",
    1: "Coal"
})

predictions["predicted_class"] = predictions[
    "predicted_label"
].map({
    0: "Non-Coal",
    1: "Coal"
})

predictions.to_csv(
    PREDICTIONS_DIR / "final_predictions.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)

print(
    f"Metrics saved to:\n"
    f"{METRICS_DIR / 'final_test_metrics.json'}"
)

print(
    f"\nPredictions saved to:\n"
    f"{PREDICTIONS_DIR / 'final_predictions.csv'}"
)

print("\nThe final test set was evaluated exactly once.")