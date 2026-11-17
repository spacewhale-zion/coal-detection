import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "final_split"
RESULTS_DIR = PROJECT_ROOT / "results" / "metrics"

N_SPLITS = 5
SEEDS = [42, 123, 2024]
PLS_COMPONENTS = 2
PCA_COMPONENTS = 2
SVM_C = 1
PLS_THRESHOLDS = np.arange(0.30, 0.71, 0.05)


def load_development_data():
    X = np.load(FINAL_SPLIT_DIR / "X_development.npy")
    y = np.load(FINAL_SPLIT_DIR / "y_development.npy")
    metadata = pd.read_csv(
        FINAL_SPLIT_DIR / "metadata_development.csv"
    )
    return X, y, metadata["material"].to_numpy()


def create_pls_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pls", PLSRegression(
            n_components=PLS_COMPONENTS,
            scale=False,
            max_iter=500
        ))
    ])


def create_svm_model():
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


def evaluate_model(model_name, y_true, scores, threshold):
    if model_name == "pls_da":
        predictions = (scores >= threshold).astype(np.int64)
    else:
        predictions = (scores >= 0).astype(np.int64)

    result = metrics(y_true, predictions)
    result.update({
        "model": model_name,
        "threshold": threshold
    })
    return result


def run_stability_comparison(X, y, groups):
    fold_results = []
    pls_oof_scores = []

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
            pls = create_pls_model()
            svm = create_svm_model()

            pls.fit(X[train_idx], y[train_idx])
            svm.fit(X[train_idx], y[train_idx])

            pls_scores = pls.predict(X[validation_idx]).ravel()
            svm_scores = svm.decision_function(X[validation_idx])

            pls_oof_scores.extend([
                {
                    "seed": seed,
                    "fold": fold,
                    "sample_index": int(index),
                    "true_label": int(label),
                    "pls_score": float(score)
                }
                for index, label, score in zip(
                    validation_idx,
                    y[validation_idx],
                    pls_scores
                )
            ])

            for model_name, scores, threshold in [
                ("pls_da", pls_scores, 0.5),
                ("svm", svm_scores, 0.0)
            ]:
                result = evaluate_model(
                    model_name,
                    y[validation_idx],
                    scores,
                    threshold
                )
                result.update({
                    "seed": seed,
                    "fold": fold
                })
                fold_results.append(result)

    return pd.DataFrame(fold_results), pd.DataFrame(pls_oof_scores)


def threshold_analysis(oof_scores):
    rows = []
    for threshold in PLS_THRESHOLDS:
        predictions = (
            oof_scores["pls_score"].to_numpy() >= threshold
        ).astype(np.int64)
        result = metrics(
            oof_scores["true_label"].to_numpy(),
            predictions
        )
        result["threshold"] = round(float(threshold), 2)
        rows.append(result)
    return pd.DataFrame(rows)


def main():
    X, y, groups = load_development_data()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REPEATED GROUPED PLS-DA VS SVM COMPARISON")
    print("=" * 70)
    print(f"Development samples: {len(X)}")
    print(f"Materials: {len(np.unique(groups))}")
    print(f"Seeds: {SEEDS}")

    fold_results, oof_scores = run_stability_comparison(X, y, groups)
    threshold_results = threshold_analysis(oof_scores)

    fold_results.to_csv(
        RESULTS_DIR / "pls_svm_stability_folds.csv",
        index=False
    )
    oof_scores.to_csv(
        RESULTS_DIR / "pls_da_oof_scores.csv",
        index=False
    )
    threshold_results.to_csv(
        RESULTS_DIR / "pls_da_threshold_analysis.csv",
        index=False
    )

    summary = (
        fold_results
        .groupby("model")[["accuracy", "precision", "recall", "f1"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(
        RESULTS_DIR / "pls_svm_stability_summary.csv",
        index=False
    )

    best_threshold = threshold_results.sort_values(
        ["f1", "recall"],
        ascending=False
    ).iloc[0]
    report = {
        "seeds": SEEDS,
        "n_splits": N_SPLITS,
        "pls_components": PLS_COMPONENTS,
        "svm_pca_components": PCA_COMPONENTS,
        "svm_C": SVM_C,
        "best_oof_pls_threshold_by_f1": float(
            best_threshold["threshold"]
        ),
        "best_oof_pls_f1": float(best_threshold["f1"]),
        "stability_summary": summary.to_dict(orient="records")
    }
    with open(RESULTS_DIR / "pls_svm_stability_summary.json", "w") as file:
        json.dump(report, file, indent=4)

    print("\nStability summary")
    print(summary.to_string(index=False))
    print("\nPLS-DA threshold analysis")
    print(threshold_results.to_string(index=False))
    print(
        "\nBest out-of-fold PLS-DA threshold by F1: "
        f"{best_threshold['threshold']:.2f}"
    )
    print("Stability artifacts saved.")


if __name__ == "__main__":
    main()