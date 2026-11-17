import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from src.models.inference import (
    EXPECTED_FEATURES,
    load_model,
    predict_spectra,
    validate_spectra
)


DATA_PATH = (
    "data/processed/final_split/X_final_test.npy"
)


def load_sample_spectra():
    return np.load(DATA_PATH)[:8]


def test_validate_accepts_single_spectrum():
    spectra = validate_spectra(load_sample_spectra()[0])

    assert spectra.shape == (1, EXPECTED_FEATURES)
    assert spectra.dtype == np.float64


def test_predict_returns_expected_schema_for_batch():
    predictions = predict_spectra(load_sample_spectra())

    assert len(predictions) == 8
    assert list(predictions.columns) == [
        "predicted_label",
        "predicted_class",
        "decision_score"
    ]
    assert set(predictions["predicted_label"]).issubset({0, 1})
    assert set(predictions["predicted_class"]).issubset({
        "Coal",
        "Non-Coal"
    })


def test_predictions_match_saved_model_components():
    spectra = load_sample_spectra()
    scaler, pca, svm = load_model()
    expected = svm.predict(
        pca.transform(scaler.transform(spectra))
    )

    predictions = predict_spectra(spectra)

    assert np.array_equal(
        predictions["predicted_label"].to_numpy(),
        expected
    )
    assert accuracy_score(
        expected,
        predictions["predicted_label"]
    ) == 1.0


def test_validate_rejects_wrong_feature_count():
    with pytest.raises(ValueError, match="Expected 1500"):
        validate_spectra(np.zeros((2, EXPECTED_FEATURES - 1)))


def test_validate_rejects_non_finite_values():
    spectra = np.zeros((1, EXPECTED_FEATURES))
    spectra[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_spectra(spectra)