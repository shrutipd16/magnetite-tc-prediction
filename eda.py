import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
df = pd.read_csv("data/manganites_dataset.csv")

X = df[["a_A", "b_A", "c_A"]].values
y = df["TC_experimental_K"].values
a, b, c = X[:, 0], X[:, 1], X[:, 2]

print("=== Exploratory Data Analysis ===")
print(f"  Total samples : {len(y)}")
print(f"  TC range      : {y.min():.1f} K — {y.max():.1f} K")
print(f"  TC mean       : {y.mean():.2f} K")
print(f"  TC std        : {y.std():.2f} K")

# ─────────────────────────────────────────────
# FIG 0 — 3D scatter: a, b, c coloured by TC
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")

sc = ax.scatter(a, b, c, c=y, cmap="plasma", s=60, edgecolors="k", linewidths=0.4)

cbar = fig.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
cbar.set_label("Curie Temperature TC (K)", fontsize=11)

ax.set_xlabel("a (Å)", fontsize=11, labelpad=8)
ax.set_ylabel("b (Å)", fontsize=11, labelpad=8)
ax.set_zlabel("c (Å)", fontsize=11, labelpad=8)
ax.set_title("Lattice Parameters vs Curie Temperature", fontsize=13, pad=12)

plt.tight_layout()
out = FIGURES_DIR / "fig0_3d_lattice_TC.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\nSaved: {out}")