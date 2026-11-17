import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "final_split"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"

PCA_COMPONENTS = 2
SVM_C = 1


def load_development_data():
    X = np.load(FINAL_SPLIT_DIR / "X_development.npy")
    y = np.load(FINAL_SPLIT_DIR / "y_development.npy")
    metadata = pd.read_csv(
        FINAL_SPLIT_DIR / "metadata_development.csv"
    )
    return X, y, metadata


def create_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=PCA_COMPONENTS,
            svd_solver="full"
        )),
        ("svm", SVC(
            kernel="rbf",
            C=SVM_C,
            gamma="scale"
        ))
    ])


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "samples": int(len(y_true)),
        "errors": int(np.sum(y_true != y_pred))
    }


def main():
    X, y, metadata = load_development_data()
    groups = metadata["material"].to_numpy()
    logo = LeaveOneGroupOut()
    predictions = np.full(len(y), -1, dtype=np.int64)
    rows = []

    print("=" * 70)
    print("LEAVE-ONE-MATERIAL-OUT SVM EVALUATION")
    print("=" * 70)
    print(f"Development samples: {len(X)}")
    print(f"Materials: {len(np.unique(groups))}")

    for fold, (train_idx, validation_idx) in enumerate(
        logo.split(X, y, groups),
        start=1
    ):
        material = groups[validation_idx][0]
        model = create_model()
        model.fit(X[train_idx], y[train_idx])
        fold_predictions = model.predict(X[validation_idx])
        predictions[validation_idx] = fold_predictions

        fold_metrics = calculate_metrics(
            y[validation_idx],
            fold_predictions
        )
        fold_metrics.update({
            "fold": fold,
            "material": material,
            "class": "Coal" if y[validation_idx][0] == 1 else "Non-Coal"
        })
        rows.append(fold_metrics)

        print(
            f"{fold:2d}. {material}: "
            f"accuracy={fold_metrics['accuracy']:.4f}, "
            f"f1={fold_metrics['f1']:.4f}, "
            f"errors={fold_metrics['errors']}"
        )

    if np.any(predictions == -1):
        raise RuntimeError("Some samples received no out-of-fold prediction.")

    per_material = pd.DataFrame(rows).sort_values("material")
    pooled_metrics = calculate_metrics(y, predictions)
    confusion = confusion_matrix(y, predictions)
    tn, fp, fn, tp = confusion.ravel()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    per_material.to_csv(
        RESULTS_DIR / "svm_leave_one_material_metrics.csv",
        index=False
    )

    report = {
        "protocol": "LeaveOneGroupOut",
        "groups": int(len(np.unique(groups))),
        "pca_components": PCA_COMPONENTS,
        "svm_C": SVM_C,
        "pooled_metrics": pooled_metrics,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "mean_material_accuracy": float(per_material["accuracy"].mean()),
        "std_material_accuracy": float(per_material["accuracy"].std()),
        "mean_material_f1": float(per_material["f1"].mean()),
        "std_material_f1": float(per_material["f1"].std())
    }
    with open(RESULTS_DIR / "svm_leave_one_material_summary.json", "w") as file:
        json.dump(report, file, indent=4)

    oof_predictions = metadata.copy()
    oof_predictions["true_label"] = y
    oof_predictions["predicted_label"] = predictions
    oof_predictions["correct"] = (
        oof_predictions["true_label"]
        == oof_predictions["predicted_label"]
    )
    oof_predictions.to_csv(
        PREDICTIONS_DIR / "svm_leave_one_material_predictions.csv",
        index=False
    )

    print("\nPooled metrics")
    print(json.dumps(pooled_metrics, indent=4))
    print("\nClassification report")
    print(classification_report(
        y,
        predictions,
        target_names=["Non-Coal", "Coal"],
        zero_division=0
    ))
    print("Leave-one-material-out artifacts saved.")


if __name__ == "__main__":
    main()