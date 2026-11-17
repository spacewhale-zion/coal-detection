import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "final_split"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"

N_SPLITS = 5
RANDOM_STATE = 42
COMPONENTS = [2, 5, 10, 15, 20, 30]


def load_data():
    X = np.load(FINAL_SPLIT_DIR / "X_development.npy")
    y = np.load(FINAL_SPLIT_DIR / "y_development.npy")
    X_test = np.load(FINAL_SPLIT_DIR / "X_final_test.npy")
    y_test = np.load(FINAL_SPLIT_DIR / "y_final_test.npy")
    development_metadata = pd.read_csv(
        FINAL_SPLIT_DIR / "metadata_development.csv"
    )
    test_metadata = pd.read_csv(
        FINAL_SPLIT_DIR / "metadata_final_test.csv"
    )
    return X, y, X_test, y_test, development_metadata, test_metadata


def create_model(n_components):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pls", PLSRegression(
            n_components=n_components,
            scale=False,
            max_iter=500
        ))
    ])


def predict_labels(model, X):
    scores = model.predict(X).ravel()
    return (scores >= 0.5).astype(np.int64)


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1_score": float(f1_score(
            y_true, y_pred, zero_division=0
        ))
    }


def run_grouped_cv(X, y, groups):
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    results = []

    for n_components in COMPONENTS:
        fold_metrics = []

        for train_idx, validation_idx in cv.split(X, y, groups):
            model = create_model(n_components)
            model.fit(X[train_idx], y[train_idx])
            predictions = predict_labels(model, X[validation_idx])
            fold_metrics.append(calculate_metrics(
                y[validation_idx], predictions
            ))

        results.append({
            "components": n_components,
            "accuracy_mean": np.mean([
                item["accuracy"] for item in fold_metrics
            ]),
            "accuracy_std": np.std([
                item["accuracy"] for item in fold_metrics
            ]),
            "precision_mean": np.mean([
                item["precision"] for item in fold_metrics
            ]),
            "recall_mean": np.mean([
                item["recall"] for item in fold_metrics
            ]),
            "f1_mean": np.mean([
                item["f1_score"] for item in fold_metrics
            ])
        })

    return pd.DataFrame(results)


def main():
    X, y, X_test, y_test, development_metadata, test_metadata = load_data()
    groups = development_metadata["material"].to_numpy()

    print("=" * 70)
    print("PLS-DA — GROUPED MODEL SELECTION")
    print("=" * 70)
    print(f"Development samples: {len(X)}")
    print(f"Final test samples : {len(X_test)}")
    print(f"Candidate components: {COMPONENTS}")

    cv_results = run_grouped_cv(X, y, groups)
    cv_results.to_csv(METRICS_DIR / "pls_da_cv_results.csv", index=False)

    best_row = cv_results.sort_values(
        ["f1_mean", "accuracy_mean"],
        ascending=False
    ).iloc[0]
    best_components = int(best_row["components"])

    print("\nGrouped CV results")
    print(cv_results.to_string(index=False))
    print(f"\nSelected components: {best_components}")

    model = create_model(best_components)
    model.fit(X, y)
    predictions = predict_labels(model, X_test)
    metrics = calculate_metrics(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = matrix.ravel()

    metrics.update({
        "components": best_components,
        "training_samples": int(len(X)),
        "test_samples": int(len(X_test)),
        "coal_samples": int(np.sum(y_test == 1)),
        "non_coal_samples": int(np.sum(y_test == 0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        }
    })

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "pls_da_model.pkl")
    with open(METRICS_DIR / "pls_da_test_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    test_predictions = test_metadata.copy()
    test_predictions["true_label"] = y_test
    test_predictions["predicted_label"] = predictions
    test_predictions["correct"] = (
        test_predictions["true_label"]
        == test_predictions["predicted_label"]
    )
    test_predictions.to_csv(
        PREDICTIONS_DIR / "pls_da_predictions.csv",
        index=False
    )

    print("\nFinal test results")
    print(json.dumps(metrics, indent=4))
    print("\nClassification report")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Non-Coal", "Coal"],
        zero_division=0
    ))
    print("PLS-DA artifacts saved.")


if __name__ == "__main__":
    main()