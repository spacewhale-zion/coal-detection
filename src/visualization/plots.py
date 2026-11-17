import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# plots.py
#   ↓
# src/
#   ↓
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
    Load processed spectroscopy dataset.

    Returns
    -------
    X : numpy.ndarray
        Spectral data.
        Shape = (number_of_samples, number_of_wavelengths)

    y : numpy.ndarray
        Labels.
        1 = Coal
        0 = Non-Coal

    wavelengths : numpy.ndarray
        Wavelength values corresponding to X columns.
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

    if len(X) != len(y):

        raise ValueError(
            f"Number of samples in X and y do not match.\n"
            f"X samples: {len(X)}\n"
            f"y samples: {len(y)}"
        )

    if X.shape[1] != len(wavelengths):

        raise ValueError(
            f"Number of wavelengths does not match "
            f"number of features.\n"
            f"X features: {X.shape[1]}\n"
            f"Wavelengths: {len(wavelengths)}"
        )

    return X, y, wavelengths


# ============================================================
# PLOT SAMPLE SPECTRA
# ============================================================

def plot_sample_spectra(
    X,
    y,
    wavelengths,
    number_of_spectra=10
):
    """
    Plot individual sample spectra for
    Coal and Non-Coal classes.
    """

    # --------------------------------------------------------
    # Separate classes
    # --------------------------------------------------------

    coal = X[y == 1]
    non_coal = X[y == 0]

    # ========================================================
    # COAL SPECTRA
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    number_of_coal = min(
        number_of_spectra,
        len(coal)
    )

    for i in range(number_of_coal):

        plt.plot(
            wavelengths,
            coal[i],
            alpha=0.5
        )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Reflectance"
    )

    plt.title(
        "Sample Coal Spectra"
    )

    plt.grid(
        True,
        alpha=0.3
    )
    plt.legend(
    [f"{number_of_coal} Coal Samples"]
)

    plt.tight_layout()

    plt.show()

    # ========================================================
    # NON-COAL SPECTRA
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    number_of_non_coal = min(
        number_of_spectra,
        len(non_coal)
    )

    for i in range(number_of_non_coal):

        plt.plot(
            wavelengths,
            non_coal[i],
            alpha=0.5
        )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Reflectance"
    )
    plt.legend(
    [f"{number_of_non_coal} Non-Coal Samples"]
)

    plt.title(
        "Sample Non-Coal Spectra"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# PLOT SPECTRAL DIFFERENCE
# ============================================================

def plot_spectral_difference(
    X,
    y,
    wavelengths
):
    """
    Plot the absolute difference between
    average Coal and Non-Coal spectra.
    """

    coal = X[y == 1]
    non_coal = X[y == 0]

    coal_mean = np.mean(
        coal,
        axis=0
    )

    non_coal_mean = np.mean(
        non_coal,
        axis=0
    )

    difference = np.abs(
        coal_mean - non_coal_mean
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        wavelengths,
        difference
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Absolute Spectral Difference"
    )

    plt.title(
        "Spectral Difference: Coal vs Non-Coal"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()

# ============================================================
# PLOT AVERAGE SPECTRA
# ============================================================

def plot_average_spectra(
    X,
    y,
    wavelengths
):
    """
    Plot average spectrum of Coal vs Non-Coal.
    """

    # --------------------------------------------------------
    # Separate classes
    # --------------------------------------------------------

    coal = X[y == 1]
    non_coal = X[y == 0]

    # --------------------------------------------------------
    # Check classes
    # --------------------------------------------------------

    if len(coal) == 0:

        raise ValueError(
            "No Coal samples found."
        )

    if len(non_coal) == 0:

        raise ValueError(
            "No Non-Coal samples found."
        )

    # --------------------------------------------------------
    # Calculate mean spectrum
    # --------------------------------------------------------

    coal_mean = np.mean(
        coal,
        axis=0
    )

    non_coal_mean = np.mean(
        non_coal,
        axis=0
    )

    # ========================================================
    # PLOT
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        wavelengths,
        coal_mean,
        label="Coal"
    )

    plt.plot(
        wavelengths,
        non_coal_mean,
        label="Non-Coal"
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Reflectance"
    )

    plt.title(
        "Average Coal vs Non-Coal Spectra"
    )

    plt.legend()

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

    print("=" * 60)
    print("SPECTRAL DATA VISUALIZATION")
    print("=" * 60)

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

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print("\nDataset information")
    print("-" * 60)

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
        f"{wavelengths[0]} - {wavelengths[-1]} nm"
    )

    print(
        f"Coal samples     : "
        f"{np.sum(y == 1)}"
    )

    print(
        f"Non-Coal samples : "
        f"{np.sum(y == 0)}"
    )

    # --------------------------------------------------------
    # Plot individual spectra
    # --------------------------------------------------------

    plot_sample_spectra(
        X,
        y,
        wavelengths,
        number_of_spectra=10
    )

    # --------------------------------------------------------
    # Plot average spectra
    # --------------------------------------------------------

    plot_average_spectra(
        X,
        y,
        wavelengths
    )
    plot_spectral_difference(
    X,
    y,
    wavelengths
)