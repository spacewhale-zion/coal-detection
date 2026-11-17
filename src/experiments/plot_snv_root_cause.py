import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Spectroscopy dataset"
    / "Spectrum"
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MATERIALS
# ============================================================

coal_file = DATA_DIR / "Lignite coal grade 2.csv"
noncoal_file = DATA_DIR / "Argillaceous limestone 2.csv"


# ============================================================
# LOAD SPECTRA
# ============================================================

def load_spectra(file_path):

    df = pd.read_csv(
        file_path,
        header=None
    )

    # Wavelengths are stored starting from row 7
    wavelengths = pd.to_numeric(
        df.iloc[7:, 0],
        errors="coerce"
    ).to_numpy()

    # Spectral measurements start from column 1
    spectra = df.iloc[7:, 1:].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy()

    # Remove invalid wavelength rows
    valid = ~np.isnan(wavelengths)

    wavelengths = wavelengths[valid]
    spectra = spectra[valid]

    return wavelengths, spectra


# ============================================================
# LOAD DATA
# ============================================================

coal_wavelengths, coal_spectra = load_spectra(
    coal_file
)

noncoal_wavelengths, noncoal_spectra = load_spectra(
    noncoal_file
)


# ============================================================
# CHECK
# ============================================================

print("=" * 60)
print("RAW SPECTRAL COMPARISON")
print("=" * 60)

print(
    f"Lignite coal grade 2 samples : "
    f"{coal_spectra.shape[1]}"
)

print(
    f"Argillaceous limestone 2 samples : "
    f"{noncoal_spectra.shape[1]}"
)

print(
    f"Wavelengths : "
    f"{len(coal_wavelengths)}"
)


# ============================================================
# AVERAGE SPECTRUM
# ============================================================

coal_mean = np.nanmean(
    coal_spectra,
    axis=1
)

noncoal_mean = np.nanmean(
    noncoal_spectra,
    axis=1
)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    coal_wavelengths,
    coal_mean,
    linewidth=2,
    label="Lignite coal grade 2"
)

plt.plot(
    noncoal_wavelengths,
    noncoal_mean,
    linewidth=2,
    label="Argillaceous limestone 2"
)

plt.xlabel(
    "Wavelength (nm)",
    fontsize=12
)

plt.ylabel(
    "Spectral Intensity",
    fontsize=12
)

plt.title(
    "Raw Spectra: Lignite Coal Grade 2 vs Argillaceous Limestone 2",
    fontsize=14
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_path = (
    FIGURES_DIR
    / "lignite_vs_argillaceous_limestone_raw.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPlot saved to:")
print(output_path)