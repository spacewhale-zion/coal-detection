import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
PCA_COMPONENTS = [2, 10, 20, 50, 100]


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
        ("pca", PCA(
            n_components=n_components,
            svd_solver="full",
            random_state=RANDOM_STATE
        )),
        ("random_forest", RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }


def run_grouped_cv(X, y, groups):
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    rows = []

    for n_components in PCA_COMPONENTS:
        fold_metrics = []

        for train_idx, validation_idx in cv.split(X, y, groups):
            model = create_model(n_components)
            model.fit(X[train_idx], y[train_idx])
            predictions = model.predict(X[validation_idx])
            fold_metrics.append(calculate_metrics(
                y[validation_idx], predictions
            ))

        rows.append({
            "pca_components": n_components,
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
                item["f1"] for item in fold_metrics
            ])
        })

    return pd.DataFrame(rows)


def main():
    X, y, X_test, y_test, development_metadata, test_metadata = load_data()
    groups = development_metadata["material"].to_numpy()

    print("=" * 70)
    print("PCA + RANDOM FOREST — GROUPED MODEL SELECTION")
    print("=" * 70)
    print(f"Development samples: {len(X)}")
    print(f"Final test samples : {len(X_test)}")
    print(f"PCA candidates: {PCA_COMPONENTS}")

    cv_results = run_grouped_cv(X, y, groups)
    cv_results.to_csv(
        METRICS_DIR / "random_forest_pca_cv_results.csv",
        index=False
    )

    best_row = cv_results.sort_values(
        ["f1_mean", "accuracy_mean"],
        ascending=False
    ).iloc[0]
    best_components = int(best_row["pca_components"])

    print("\nGrouped CV results")
    print(cv_results.to_string(index=False))
    print(f"\nSelected PCA components: {best_components}")

    model = create_model(best_components)
    model.fit(X, y)
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = confusion.ravel()

    metrics.update({
        "pca_components": best_components,
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

    joblib.dump(model, MODEL_DIR / "random_forest_pca_model.pkl")
    with open(
        METRICS_DIR / "random_forest_pca_test_metrics.json",
        "w"
    ) as file:
        json.dump(metrics, file, indent=4)

    test_predictions = test_metadata.copy()
    test_predictions["true_label"] = y_test
    test_predictions["predicted_label"] = predictions
    test_predictions["correct"] = (
        test_predictions["true_label"]
        == test_predictions["predicted_label"]
    )
    test_predictions.to_csv(
        PREDICTIONS_DIR / "random_forest_pca_predictions.csv",
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
    print("Random Forest artifacts saved.")


if __name__ == "__main__":
    main()