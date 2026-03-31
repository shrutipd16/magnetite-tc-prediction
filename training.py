import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUTPUT_DIR  = Path("output")
FIGURES_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Load & scale data
# ─────────────────────────────────────────────
df = pd.read_csv("data/manganites_dataset.csv")

X = df[["a_A", "b_A", "c_A"]].values
y = df["TC_experimental_K"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────────────────────
# Define kernels
# ─────────────────────────────────────────────
rq_kernel = RationalQuadratic(length_scale=1.0, alpha=1.0,
                               length_scale_bounds=(1e-3, 1e3),
                               alpha_bounds=(1e-3, 1e3))
exp_kernel = Matern(length_scale=1.0, nu=0.5,
                    length_scale_bounds=(1e-3, 1e3))

# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────
def train_gpr(kernel, X_scaled, y):
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                   normalize_y=True)
    gpr.fit(X_scaled, y)
    return gpr

print("=== Model Training ===")
print("Training GPR — Rational Quadratic kernel...")
gpr_rq = train_gpr(rq_kernel, X_scaled, y)

print("Training GPR — Exponential kernel...")
gpr_exp = train_gpr(exp_kernel, X_scaled, y)

# ─────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────
def evaluate(gpr, X_scaled, y, name):
    y_pred = gpr.predict(X_scaled)
    cc   = np.corrcoef(y, y_pred)[0, 1]
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae  = mean_absolute_error(y, y_pred)
    print(f"\n  [{name}]")
    print(f"    CC:   {cc*100:.2f}%    (paper: 99.99%)")
    print(f"    RMSE: {rmse:.4f} K  (paper: 1.3453)")
    print(f"    MAE:  {mae:.4f} K  (paper: 0.7869)")
    return y_pred, cc, rmse, mae

y_pred_rq,  *_ = evaluate(gpr_rq,  X_scaled, y, "Rational Quadratic")
y_pred_exp, *_ = evaluate(gpr_exp, X_scaled, y, "Exponential")

# ─────────────────────────────────────────────
# Save models & scaler to output/
# ─────────────────────────────────────────────
joblib.dump(gpr_rq,  OUTPUT_DIR / "gpr_rq_model.joblib")
joblib.dump(gpr_exp, OUTPUT_DIR / "gpr_exp_model.joblib")
joblib.dump(scaler,  OUTPUT_DIR / "scaler.joblib")
print("\nSaved models:")
print(f"  {OUTPUT_DIR}/gpr_rq_model.joblib")
print(f"  {OUTPUT_DIR}/gpr_exp_model.joblib")
print(f"  {OUTPUT_DIR}/scaler.joblib")

# ─────────────────────────────────────────────
# FIG 3 — TC per compound: Experimental vs both predicted
# X axis = compound, Y axis = Temperature (K)
# ─────────────────────────────────────────────
compounds = df["sample"].values
x_idx = np.arange(len(compounds))

# Sort all by experimental TC for readability
order = np.argsort(y)
compounds_sorted = compounds[order]
y_sorted         = y[order]
y_pred_rq_sorted  = y_pred_rq[order]
y_pred_exp_sorted = y_pred_exp[order]

fig, ax = plt.subplots(figsize=(28, 7))

ax.plot(x_idx, y_sorted,          "o-", color="black",      linewidth=1.2, markersize=4, label="Experimental TC")
ax.plot(x_idx, y_pred_rq_sorted,  "s--", color="steelblue", linewidth=1.0, markersize=3, label="Predicted — Rational Quadratic", alpha=0.85)
ax.plot(x_idx, y_pred_exp_sorted, "^:",  color="orangered",  linewidth=1.0, markersize=3, label="Predicted — Exponential",        alpha=0.85)

ax.set_xticks(x_idx)
ax.set_xticklabels(compounds_sorted, rotation=90, fontsize=5.5)
ax.set_ylabel("Curie Temperature TC (K)", fontsize=12)
ax.set_xlabel("Compound", fontsize=12)
ax.set_title("Experimental vs Predicted Curie Temperature per Compound", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
out = FIGURES_DIR / "fig3_experimental_vs_predicted.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")