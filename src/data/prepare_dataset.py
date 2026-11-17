import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# prepare_dataset.py
#     ↓
# src/
#     ↓
# coal-detection/   ← PROJECT_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Spectroscopy dataset"
    / "Spectrum"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)


# ============================================================
# COAL CLASSES
# ============================================================

COAL_CLASSES = {
    "0.3 coking coal",
    "Anthracite coal grade 1",
    "Anthracite coal grade 2",
    "Coking coal",
    "Fat coal",
    "Gas coal",
    "Gas-fat coal",
    "Lean coal",
    "Lean-thin coal",
    "Lignite coal grade 1",
    "Lignite coal grade 2",
    "Thin coal",
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD ONE SPECTRUM CSV
# ============================================================

def load_spectrum_file(file_path):
    """
    Load one spectroscopy CSV file.

    Expected structure:

        Rows 0-6  -> metadata/header
        Row 7+    -> wavelength + spectral measurements

    Column 0    -> wavelength
    Column 1+   -> individual spectra

    Returns
    -------
    wavelengths : numpy.ndarray
        Shape: (number_of_wavelengths,)

    spectra : numpy.ndarray
        Shape: (number_of_spectra, number_of_wavelengths)
    """

    print(f"\nReading: {file_path.name}")

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:
        df = pd.read_csv(
            file_path,
            header=None,
            low_memory=False
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not read file:\n{file_path}\n"
            f"Error: {e}"
        )

    print(f"  Raw CSV shape: {df.shape}")

    # --------------------------------------------------------
    # Check minimum size
    # --------------------------------------------------------

    if df.shape[0] <= 7:
        raise ValueError(
            f"{file_path.name} does not contain enough rows.\n"
            f"Expected wavelength data after row 7."
        )

    if df.shape[1] < 2:
        raise ValueError(
            f"{file_path.name} does not contain spectral columns."
        )

    # --------------------------------------------------------
    # Extract wavelength column
    # --------------------------------------------------------

    wavelength_column = pd.to_numeric(
        df.iloc[7:, 0],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Extract spectral data
    # --------------------------------------------------------

    spectral_data = df.iloc[7:, 1:].apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Find rows where wavelength is valid
    # --------------------------------------------------------

    valid_wavelength_rows = (
        wavelength_column.notna()
    )

    wavelengths = wavelength_column[
        valid_wavelength_rows
    ].to_numpy(dtype=np.float64)

    spectral_data = spectral_data[
        valid_wavelength_rows
    ]

    # --------------------------------------------------------
    # Convert spectral data to numpy
    # --------------------------------------------------------

    spectra = spectral_data.to_numpy(
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Transpose
    #
    # Current:
    #
    # wavelength × spectrum
    #
    # Required:
    #
    # spectrum × wavelength
    # --------------------------------------------------------

    spectra = spectra.T

    # --------------------------------------------------------
    # Remove spectra containing NaN
    # --------------------------------------------------------

    valid_spectra = ~np.isnan(
        spectra
    ).any(axis=1)

    spectra = spectra[
        valid_spectra
    ]

    # --------------------------------------------------------
    # Remove infinite values
    # --------------------------------------------------------

    valid_spectra = np.isfinite(
        spectra
    ).all(axis=1)

    spectra = spectra[
        valid_spectra
    ]

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if len(wavelengths) == 0:
        raise ValueError(
            f"No valid wavelengths found in "
            f"{file_path.name}"
        )

    if len(spectra) == 0:
        raise ValueError(
            f"No valid spectra found in "
            f"{file_path.name}"
        )

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if spectra.shape[1] != len(wavelengths):
        raise ValueError(
            f"Dimension mismatch in {file_path.name}\n"
            f"Wavelengths: {len(wavelengths)}\n"
            f"Spectral features: {spectra.shape[1]}"
        )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print(
        f"  Wavelengths: {len(wavelengths)}"
    )

    print(
        f"  Spectra: {len(spectra)}"
    )

    print(
        f"  Final shape: {spectra.shape}"
    )

    print(
        f"  Wavelength range: "
        f"{wavelengths[0]} - {wavelengths[-1]}"
    )

    return wavelengths, spectra


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset():

    print("=" * 70)
    print("COAL SPECTROSCOPY DATASET PREPARATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Print paths
    # --------------------------------------------------------

    print(f"\nProject root:")
    print(PROJECT_ROOT)

    print(f"\nData directory:")
    print(DATA_DIR)

    print(f"\nOutput directory:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Check dataset directory
    # --------------------------------------------------------

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            "\nDataset directory does not exist!\n\n"
            f"Expected:\n{DATA_DIR}\n\n"
            "Make sure your folder structure is:\n\n"
            "coal-detection/\n"
            "└── data/\n"
            "    └── raw/\n"
            "        └── Spectroscopy dataset/\n"
            "            └── Spectrum/\n"
            "                ├── file1.csv\n"
            "                ├── file2.csv\n"
            "                └── ..."
        )

    # --------------------------------------------------------
    # Find CSV files
    # --------------------------------------------------------

    files = sorted(
        DATA_DIR.glob("*.csv")
    )

    print(
        f"\nFound {len(files)} CSV files."
    )

    # --------------------------------------------------------
    # Check if files exist
    # --------------------------------------------------------

    if len(files) == 0:

        raise FileNotFoundError(
            "\nNo CSV files found!\n\n"
            f"Expected CSV files inside:\n{DATA_DIR}\n\n"
            "Check that your files have the .csv extension."
        )

    # --------------------------------------------------------
    # Lists for dataset
    # --------------------------------------------------------

    X = []
    y = []
    materials = []

    # Store wavelengths from first file
    reference_wavelengths = None

    # --------------------------------------------------------
    # Process every CSV
    # --------------------------------------------------------

    for file_path in files:

        material_name = file_path.stem

        print("\n" + "-" * 70)
        print(
            f"Processing: {material_name}"
        )

        # ----------------------------------------------------
        # Determine class
        # ----------------------------------------------------

        if material_name in COAL_CLASSES:

            label = 1
            class_name = "Coal"

        else:

            label = 0
            class_name = "Non-Coal"

        print(
            f"  Class: {class_name}"
        )

        # ----------------------------------------------------
        # Load spectra
        # ----------------------------------------------------

        try:

            wavelengths, spectra = (
                load_spectrum_file(
                    file_path
                )
            )

        except Exception as e:

            print(
                f"\nERROR processing "
                f"{file_path.name}:"
            )

            print(e)

            print(
                "\nSkipping this file..."
            )

            continue

        # ----------------------------------------------------
        # Check wavelength consistency
        # ----------------------------------------------------

        if reference_wavelengths is None:

            reference_wavelengths = (
                wavelengths.copy()
            )

        else:

            if len(wavelengths) != len(
                reference_wavelengths
            ):

                raise ValueError(
                    f"\nWavelength count mismatch!\n"
                    f"File: {file_path.name}\n"
                    f"Expected: "
                    f"{len(reference_wavelengths)}\n"
                    f"Found: "
                    f"{len(wavelengths)}"
                )

            if not np.allclose(
                wavelengths,
                reference_wavelengths,
                rtol=1e-5,
                atol=1e-8
            ):

                raise ValueError(
                    f"\nWavelength values differ "
                    f"in {file_path.name}"
                )

        # ----------------------------------------------------
        # Add spectra
        # ----------------------------------------------------

        X.extend(
            spectra
        )

        # ----------------------------------------------------
        # Add labels
        # ----------------------------------------------------

        y.extend(
            [label] * len(spectra)
        )

        # ----------------------------------------------------
        # Add material names
        # ----------------------------------------------------

        materials.extend(
            [material_name] * len(spectra)
        )

        print(
            f"  Added {len(spectra)} spectra."
        )

    # ========================================================
    # CHECK WHETHER DATA WAS LOADED
    # ========================================================

    if len(X) == 0:

        raise RuntimeError(
            "\nNo valid spectra were loaded!\n\n"
            "Possible reasons:\n"
            "1. CSV files have a different format.\n"
            "2. Wavelength data does not start at row 8.\n"
            "3. Spectral values contain invalid data.\n"
            "4. CSV files are empty."
        )

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

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
            f"X and y have different lengths!\n"
            f"X: {len(X)}\n"
            f"y: {len(y)}"
        )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(
        f"X shape              : {X.shape}"
    )

    print(
        f"y shape              : {y.shape}"
    )

    print(
        f"Total samples        : {len(X)}"
    )

    print(
        f"Number of wavelengths: {X.shape[1]}"
    )

    print(
        f"Coal samples         : "
        f"{np.sum(y == 1)}"
    )

    print(
        f"Non-Coal samples     : "
        f"{np.sum(y == 0)}"
    )

    print(
        f"Wavelength range     : "
        f"{reference_wavelengths[0]} - "
        f"{reference_wavelengths[-1]}"
    )

    # ========================================================
    # MATERIAL SUMMARY
    # ========================================================

    metadata = pd.DataFrame(
        {
            "material": materials,
            "label": y,
            "class": [
                "Coal" if label == 1
                else "Non-Coal"
                for label in y
            ]
        }
    )

    print("\n")
    print("=" * 70)
    print("SAMPLES PER MATERIAL")
    print("=" * 70)

    print(
        metadata["material"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # SAVE X
    # ========================================================

    X_path = (
        OUTPUT_DIR / "X.npy"
    )

    np.save(
        X_path,
        X
    )

    # ========================================================
    # SAVE y
    # ========================================================

    y_path = (
        OUTPUT_DIR / "y.npy"
    )

    np.save(
        y_path,
        y
    )

    # ========================================================
    # SAVE WAVELENGTHS
    # ========================================================

    wavelength_path = (
        OUTPUT_DIR / "wavelengths.npy"
    )

    np.save(
        wavelength_path,
        reference_wavelengths
    )

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_path = (
        OUTPUT_DIR / "metadata.csv"
    )

    metadata.to_csv(
        metadata_path,
        index=False
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DATASET SAVED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"X            → {X_path}"
    )

    print(
        f"y            → {y_path}"
    )

    print(
        f"Wavelengths   → {wavelength_path}"
    )

    print(
        f"Metadata      → {metadata_path}"
    )

    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    prepare_dataset()