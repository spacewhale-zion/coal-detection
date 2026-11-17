import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

# spectral.py
#     ↓
# src/
#     ↓
# coal-detection/

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load processed spectroscopy data.

    Returns
    -------
    X : numpy.ndarray
        Shape:
        (number_of_samples, number_of_wavelengths)

    y : numpy.ndarray
        1 = Coal
        0 = Non-Coal

    wavelengths : numpy.ndarray
        Wavelength corresponding to each feature.
    """

    X_path = PROCESSED_DIR / "X.npy"
    y_path = PROCESSED_DIR / "y.npy"
    wavelengths_path = PROCESSED_DIR / "wavelengths.npy"

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not X_path.exists():

        raise FileNotFoundError(
            f"\nX.npy not found!\n"
            f"Expected location:\n{X_path}\n\n"
            "Run prepare_dataset.py first."
        )

    if not y_path.exists():

        raise FileNotFoundError(
            f"\ny.npy not found!\n"
            f"Expected location:\n{y_path}\n\n"
            "Run prepare_dataset.py first."
        )

    if not wavelengths_path.exists():

        raise FileNotFoundError(
            f"\nwavelengths.npy not found!\n"
            f"Expected location:\n{wavelengths_path}\n\n"
            "Run prepare_dataset.py first."
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X = np.load(X_path)
    y = np.load(y_path)
    wavelengths = np.load(wavelengths_path)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if X.ndim != 2:

        raise ValueError(
            f"X must be 2-dimensional.\n"
            f"Current shape: {X.shape}"
        )

    if y.ndim != 1:

        raise ValueError(
            f"y must be 1-dimensional.\n"
            f"Current shape: {y.shape}"
        )

    if X.shape[0] != y.shape[0]:

        raise ValueError(
            "X and y contain different numbers "
            "of samples.\n"
            f"X: {X.shape[0]}\n"
            f"y: {y.shape[0]}"
        )

    if X.shape[1] != len(wavelengths):

        raise ValueError(
            "Number of spectral features does not "
            "match number of wavelengths.\n"
            f"X features: {X.shape[1]}\n"
            f"Wavelengths: {len(wavelengths)}"
        )

    return X, y, wavelengths


# ============================================================
# STANDARD NORMAL VARIATE (SNV)
# ============================================================

def snv(X):
    """
    Apply Standard Normal Variate (SNV).

    SNV normalizes each individual spectrum:

        X_snv = (X - mean) / std

    The normalization is performed independently
    for every sample.

    Parameters
    ----------
    X : numpy.ndarray
        Shape:
        (samples, wavelengths)

    Returns
    -------
    X_snv : numpy.ndarray
        SNV-transformed spectra.
    """

    X = np.asarray(
        X,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Mean of each spectrum
    # --------------------------------------------------------

    mean = np.mean(
        X,
        axis=1,
        keepdims=True
    )

    # --------------------------------------------------------
    # Standard deviation of each spectrum
    # --------------------------------------------------------

    std = np.std(
        X,
        axis=1,
        keepdims=True
    )

    # --------------------------------------------------------
    # Avoid division by zero
    # --------------------------------------------------------

    std[std == 0] = 1.0

    # --------------------------------------------------------
    # SNV transformation
    # --------------------------------------------------------

    X_snv = (
        X - mean
    ) / std

    return X_snv


# ============================================================
# SAVITZKY-GOLAY FILTER
# ============================================================

def savitzky_golay(
    X,
    window_length=11,
    polyorder=2
):
    """
    Smooth spectra using Savitzky-Golay filtering.

    Parameters
    ----------
    X : numpy.ndarray
        Shape:
        (samples, wavelengths)

    window_length : int
        Number of points in smoothing window.

        Must be:
        - positive
        - odd
        - greater than polyorder

    polyorder : int
        Polynomial order.

    Returns
    -------
    X_smooth : numpy.ndarray
        Smoothed spectra.
    """

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if window_length <= polyorder:

        raise ValueError(
            "window_length must be greater "
            "than polyorder."
        )

    if window_length % 2 == 0:

        raise ValueError(
            "window_length must be odd."
        )

    if window_length > X.shape[1]:

        raise ValueError(
            f"window_length ({window_length}) "
            f"cannot be greater than the number "
            f"of wavelengths ({X.shape[1]})."
        )

    # --------------------------------------------------------
    # Apply filter
    # --------------------------------------------------------

    X_smooth = savgol_filter(
        X,
        window_length=window_length,
        polyorder=polyorder,
        axis=1
    )

    return X_smooth


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize(
    X_train,
    X_test
):
    """
    Standardize spectral features.

    IMPORTANT
    ---------
    The scaler is fitted ONLY on training data.

    This prevents test-data leakage.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training spectra.

    X_test : numpy.ndarray
        Test spectra.

    Returns
    -------
    X_train_scaled
    X_test_scaled
    scaler
    """

    scaler = StandardScaler()

    # --------------------------------------------------------
    # Fit ONLY on training data
    # --------------------------------------------------------

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # --------------------------------------------------------
    # Transform test data
    # --------------------------------------------------------

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )


# ============================================================
# PREPROCESS COMPLETE DATASET
# ============================================================

def preprocess_spectra(
    X,
    window_length=11,
    polyorder=2
):
    """
    Apply preprocessing that does not require
    train/test information.

    Pipeline:

        Raw spectra
             ↓
            SNV
             ↓
        Savitzky-Golay
             ↓
        Preprocessed spectra

    StandardScaler is NOT applied here because
    it must be fitted only on training data.
    """

    # --------------------------------------------------------
    # SNV
    # --------------------------------------------------------

    X_snv = snv(X)

    # --------------------------------------------------------
    # Savitzky-Golay
    # --------------------------------------------------------

    X_smooth = savitzky_golay(
        X_snv,
        window_length=window_length,
        polyorder=polyorder
    )

    return X_smooth


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SPECTRAL PREPROCESSING")
    print("=" * 70)

    print(
        f"\nProject root:\n{PROJECT_ROOT}"
    )

    print(
        f"\nProcessed data:\n{PROCESSED_DIR}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y, wavelengths = load_data()

    print("\nDataset")
    print("-" * 70)

    print(
        f"X shape          : {X.shape}"
    )

    print(
        f"y shape          : {y.shape}"
    )

    print(
        f"Wavelengths      : {len(wavelengths)}"
    )

    print(
        f"Wavelength range : "
        f"{wavelengths[0]} - "
        f"{wavelengths[-1]} nm"
    )

    # --------------------------------------------------------
    # SNV
    # --------------------------------------------------------

    print("\nApplying SNV...")

    X_snv = snv(X)

    print(
        f"After SNV        : {X_snv.shape}"
    )

    # --------------------------------------------------------
    # Savitzky-Golay
    # --------------------------------------------------------

    print(
        "\nApplying Savitzky-Golay filter..."
    )

    X_smooth = savitzky_golay(
        X_snv,
        window_length=11,
        polyorder=2
    )

    print(
        f"After smoothing  : {X_smooth.shape}"
    )

    # --------------------------------------------------------
    # Verify preprocessing
    # --------------------------------------------------------

    print("\nVerification")
    print("-" * 70)

    print(
        "Original sample mean:",
        np.mean(X[0])
    )

    print(
        "SNV sample mean:",
        np.mean(X_snv[0])
    )

    print(
        "SNV sample std:",
        np.std(X_snv[0])
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        "\nSNV + Savitzky-Golay preprocessing "
        "completed successfully."
    )

    print(
        "\nStandardScaler has NOT been applied yet."
    )

    print(
        "It should be fitted after train/test "
        "splitting to avoid data leakage."
    )