import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
EXPECTED_FEATURES = 1500


def load_model(model_dir=MODEL_DIR):
    """Load the fitted SVM pipeline components without refitting them."""
    return (
        joblib.load(model_dir / "scaler.pkl"),
        joblib.load(model_dir / "pca.pkl"),
        joblib.load(model_dir / "svm.pkl")
    )


def validate_spectra(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError(
            "Spectra must have shape (samples, wavelengths)."
        )
    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} wavelength features, "
            f"received {X.shape[1]}."
        )
    if not np.isfinite(X).all():
        raise ValueError("Spectra contain NaN or infinite values.")
    return X


def predict_spectra(X, model_dir=MODEL_DIR):
    """Predict coal/non-coal labels for one or more raw spectra."""
    X = validate_spectra(X)
    scaler, pca, svm = load_model(model_dir)

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    labels = svm.predict(X_pca).astype(np.int64)
    scores = svm.decision_function(X_pca)

    return pd.DataFrame({
        "predicted_label": labels,
        "predicted_class": np.where(labels == 1, "Coal", "Non-Coal"),
        "decision_score": scores
    })


def main():
    parser = argparse.ArgumentParser(
        description="Predict coal/non-coal labels from a .npy spectra file."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a .npy array shaped (1500,) or (samples, 1500)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path for predictions."
    )
    args = parser.parse_args()

    predictions = predict_spectra(np.load(args.input))
    if args.output is None:
        print(predictions.to_string(index=False))
    else:
        predictions.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()