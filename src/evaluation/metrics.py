
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# PRINT METRICS
# ============================================================

def print_metrics(
    y_true,
    y_pred
):

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    print("\n")
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(
        f"\nAccuracy  : "
        f"{metrics['accuracy']:.4f} "
        f"({metrics['accuracy'] * 100:.2f}%)"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f} "
        f"({metrics['precision'] * 100:.2f}%)"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f} "
        f"({metrics['recall'] * 100:.2f}%)"
    )

    print(
        f"F1 Score  : "
        f"{metrics['f1']:.4f} "
        f"({metrics['f1'] * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report")
    print("-" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "Non-Coal",
                "Coal"
            ],
            zero_division=0
        )
    )

    return metrics


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nConfusion Matrix")
    print("-" * 70)

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Non-Coal",
            "Coal"
        ]
    )

    display.plot()

    plt.title(
        "Coal vs Non-Coal Confusion Matrix"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Temporary example
    # --------------------------------------------------------

    y_true = np.array([
        1, 1, 1, 0, 0, 0, 1, 0
    ])

    y_pred = np.array([
        1, 1, 0, 0, 0, 1, 1, 0
    ])

    print_metrics(
        y_true,
        y_pred
    )

    plot_confusion_matrix(
        y_true,
        y_pred
    )