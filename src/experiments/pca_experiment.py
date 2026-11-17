import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "metrics"
)


# ============================================================
# CONFIGURATION
# ============================================================

COMPONENTS = [
    9,
    20,
    50,
    100,
    200,
    500
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    X_train = np.load(
        PREPROCESSED_DIR / "X_train.npy"
    )

    X_test = np.load(
        PREPROCESSED_DIR / "X_test.npy"
    )

    y_train = np.load(
        PREPROCESSED_DIR / "y_train.npy"
    )

    y_test = np.load(
        PREPROCESSED_DIR / "y_test.npy"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# RUN ONE PCA EXPERIMENT
# ============================================================

def run_experiment(
    X_train,
    X_test,
    y_train,
    y_test,
    n_components
):

    print(
        f"\nTesting {n_components} components..."
    )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=n_components
    )

    X_train_pca = pca.fit_transform(
        X_train
    )

    X_test_pca = pca.transform(
        X_test
    )

    variance = np.sum(
        pca.explained_variance_ratio_
    )

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    )

    model.fit(
        X_train_pca,
        y_train
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test_pca
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    return {
        "components": n_components,
        "variance": variance,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# PLOT RESULTS
# ============================================================

def plot_results(results):

    df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        df["components"],
        df["accuracy"] * 100,
        marker="o"
    )

    plt.xlabel(
        "Number of PCA Components"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.title(
        "PCA Components vs SVM Accuracy"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        df["components"],
        df["f1"] * 100,
        marker="o"
    )

    plt.xlabel(
        "Number of PCA Components"
    )

    plt.ylabel(
        "Coal F1 Score (%)"
    )

    plt.title(
        "PCA Components vs Coal F1 Score"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("PCA COMPONENT EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = load_data()

    print(
        "\nOriginal features:",
        X_train.shape[1]
    )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    results = []

    for n_components in COMPONENTS:

        result = run_experiment(
            X_train,
            X_test,
            y_train,
            y_test,
            n_components
        )

        results.append(
            result
        )

        print(
            f"  Variance : "
            f"{result['variance'] * 100:.2f}%"
        )

        print(
            f"  Accuracy : "
            f"{result['accuracy'] * 100:.2f}%"
        )

        print(
            f"  Precision: "
            f"{result['precision'] * 100:.2f}%"
        )

        print(
            f"  Recall   : "
            f"{result['recall'] * 100:.2f}%"
        )

        print(
            f"  F1       : "
            f"{result['f1'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    print("\n")
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df.to_csv(
        RESULTS_DIR
        / "pca_experiment.csv",
        index=False
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_results(
        results
    )