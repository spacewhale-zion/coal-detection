import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(
    PROCESSED_DIR / "X.npy"
)

y = np.load(
    PROCESSED_DIR / "y.npy"
)

wavelengths = np.load(
    PROCESSED_DIR / "wavelengths.npy"
)

metadata = pd.read_csv(
    PROCESSED_DIR / "metadata.csv"
)


# ============================================================
# MATERIALS TO INVESTIGATE
# ============================================================

TARGET_MATERIALS = [
    "Lignite coal grade 2",
    "Argillaceous limestone 2"
]


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(12, 7)
)

for material in TARGET_MATERIALS:

    mask = (
        metadata["material"] == material
    )

    spectra = X[mask]

    mean_spectrum = spectra.mean(
        axis=0
    )

    std_spectrum = spectra.std(
        axis=0
    )

    plt.plot(
        wavelengths,
        mean_spectrum,
        linewidth=2,
        label=material
    )

    plt.fill_between(
        wavelengths,
        mean_spectrum - std_spectrum,
        mean_spectrum + std_spectrum,
        alpha=0.15
    )


# ============================================================
# FORMAT
# ============================================================

plt.xlabel(
    "Wavelength (nm)"
)

plt.ylabel(
    "Spectral Intensity"
)

plt.title(
    "Spectral Comparison of Difficult Material Groups"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

output_path = (
    FIGURES_DIR
    / "difficult_materials_comparison.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.show()

print(
    f"\nSaved figure to:\n{output_path}"
)

# ============================================================
# CLASS-LEVEL COMPARISON
# ============================================================

plt.figure(
    figsize=(12, 7)
)


# ------------------------------------------------------------
# Coal average
# ------------------------------------------------------------

coal_mask = (
    y == 1
)

coal_mean = X[coal_mask].mean(
    axis=0
)

plt.plot(
    wavelengths,
    coal_mean,
    linewidth=2,
    label="All Coal"
)


# ------------------------------------------------------------
# Non-Coal average
# ------------------------------------------------------------

noncoal_mask = (
    y == 0
)

noncoal_mean = X[noncoal_mask].mean(
    axis=0
)

plt.plot(
    wavelengths,
    noncoal_mean,
    linewidth=2,
    label="All Non-Coal"
)


# ------------------------------------------------------------
# Lignite grade 2
# ------------------------------------------------------------

lignite_mask = (
    metadata["material"]
    == "Lignite coal grade 2"
)

lignite_mean = X[lignite_mask].mean(
    axis=0
)

plt.plot(
    wavelengths,
    lignite_mean,
    linewidth=2,
    label="Lignite coal grade 2"
)


# ------------------------------------------------------------
# Argillaceous limestone 2
# ------------------------------------------------------------

limestone_mask = (
    metadata["material"]
    == "Argillaceous limestone 2"
)

limestone_mean = X[limestone_mask].mean(
    axis=0
)

plt.plot(
    wavelengths,
    limestone_mean,
    linewidth=2,
    label="Argillaceous limestone 2"
)


plt.xlabel(
    "Wavelength (nm)"
)

plt.ylabel(
    "Spectral Intensity"
)

plt.title(
    "Coal vs Non-Coal vs Difficult Material Groups"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

output_path = (
    FIGURES_DIR
    / "coal_noncoal_problem_materials.png"
)

plt.savefig(
    output_path,
    dpi=300
)

plt.show()

print(
    f"Saved figure to:\n{output_path}"
)