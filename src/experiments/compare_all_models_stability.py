import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "final_split"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

N_SPLITS = 5
SEEDS = [42, 123, 2024]


def load_data():
    X = np.load(FINAL_SPLIT_DIR / "X_development.npy")
    y = np.load(FINAL_SPLIT_DIR / "y_development.npy")
    metadata = pd.read_csv(
        FINAL_SPLIT_DIR / "metadata_development.csv"
    )
    return X, y, metadata["material"].to_numpy()


def create_models():
    return {
        "pls_da": Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(
                n_components=2,
                scale=False,
                max_iter=500
            ))
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, svd_solver="full")),
            ("svm", SVC(
                kernel="rbf",
                C=1,
                gamma="scale"
            ))
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=10, svd_solver="full")),
            ("random_forest", RandomForestClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, svd_solver="full")),
            ("xgboost", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=2,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42
            ))
        ])
    }


def metrics(y_true, y_pred):
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


def predict(model_name, model, X):
    if model_name == "pls_da":
        return (model.predict(X).ravel() >= 0.5).astype(np.int64)
    return model.predict(X)


def main():
    X, y, groups = load_data()
    fold_rows = []
    prediction_rows = []

    print("=" * 70)
    print(
        "REPEATED GROUPED COMPARISON: "
        "PLS-DA, SVM, RANDOM FOREST, XGBOOST"
    )
    print("=" * 70)
    print(f"Development samples: {len(X)}")
    print(f"Materials: {len(np.unique(groups))}")
    print(f"Seeds: {SEEDS}")

    for seed in SEEDS:
        cv = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed
        )

        for fold, (train_idx, validation_idx) in enumerate(
            cv.split(X, y, groups),
            start=1
        ):
            for model_name, model in create_models().items():
                model.fit(X[train_idx], y[train_idx])
                predictions = predict(
                    model_name,
                    model,
                    X[validation_idx]
                )
                fold_metric = metrics(
                    y[validation_idx],
                    predictions
                )
                fold_metric.update({
                    "model": model_name,
                    "seed": seed,
                    "fold": fold
                })
                fold_rows.append(fold_metric)

                prediction_rows.extend([
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "sample_index": int(index),
                        "material": groups[index],
                        "true_label": int(label),
                        "predicted_label": int(prediction),
                        "correct": bool(label == prediction)
                    }
                    for index, label, prediction in zip(
                        validation_idx,
                        y[validation_idx],
                        predictions
                    )
                ])

    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    summary = (
        fold_results
        .groupby("model")[["accuracy", "precision", "recall", "f1"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()

    material_results = (
        predictions
        .groupby(["model", "material"])
        .agg(
            samples=("true_label", "size"),
            accuracy=("correct", "mean"),
            errors=("correct", lambda values: int((~values).sum()))
        )
        .reset_index()
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(
        RESULTS_DIR / "all_models_stability_folds.csv",
        index=False
    )
    predictions.to_csv(
        RESULTS_DIR / "all_models_stability_predictions.csv",
        index=False
    )
    summary.to_csv(
        RESULTS_DIR / "all_models_stability_summary.csv",
        index=False
    )
    material_results.to_csv(
        RESULTS_DIR / "all_models_material_performance.csv",
        index=False
    )

    report = {
        "seeds": SEEDS,
        "n_splits": N_SPLITS,
        "summary": summary.to_dict(orient="records"),
        "worst_materials_by_model": {
            model: material_results[
                material_results["model"] == model
            ].sort_values("accuracy").head(5).to_dict(
                orient="records"
            )
            for model in material_results["model"].unique()
        }
    }
    with open(RESULTS_DIR / "all_models_stability_summary.json", "w") as file:
        json.dump(report, file, indent=4)

    print("\nRepeated grouped-CV summary")
    print(summary.to_string(index=False))
    print("\nWorst materials by model")
    print(material_results.sort_values(
        ["model", "accuracy"]
    ).groupby("model").head(5).to_string(index=False))
    print("\nAll-model stability artifacts saved.")


if __name__ == "__main__":
    main()