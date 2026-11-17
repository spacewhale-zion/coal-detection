import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA FROM PCA COMPONENT SENSITIVITY EXPERIMENT
# C = 10
# ============================================================

pca_components = [1, 2, 3, 5, 10, 20]

accuracy = [
    78.01,
    93.60,
    92.10,
    92.84,
    93.11,
    93.18
]


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    pca_components,
    accuracy,
    marker="o",
    linewidth=2
)

# Highlight PCA = 2
plt.scatter(
    2,
    93.60,
    s=100,
    zorder=5
)

plt.annotate(
    "PCA = 2\n93.60%",
    xy=(2, 93.60),
    xytext=(4, 89),
    arrowprops=dict(arrowstyle="->"),
    fontsize=11
)

plt.title(
    "Effect of PCA Components on SVM Accuracy",
    fontsize=15
)

plt.xlabel(
    "Number of PCA Components",
    fontsize=12
)

plt.ylabel(
    "Accuracy (%)",
    fontsize=12
)

plt.xticks(pca_components)

plt.ylim(75, 96)

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
    / "pca_components_vs_accuracy.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 60)
print("PLOT CREATED")
print("=" * 60)
print(f"Saved to:")
print(output_path)