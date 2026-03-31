import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import resample
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OUTPUT_DIR  = Path("output")
FIGURES_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# Load data & saved scaler
# ─────────────────────────────────────────────
df = pd.read_csv("data/manganites_dataset.csv")
X = df[["a_A", "b_A", "c_A"]].values
y = df["TC_experimental_K"].values

scaler  = joblib.load(OUTPUT_DIR / "scaler.joblib")
X_scaled = scaler.transform(X)

# Re-use same kernel definition for bootstrap/sweep fits
rq_kernel = RationalQuadratic(length_scale=1.0, alpha=1.0,
                               length_scale_bounds=(1e-3, 1e3),
                               alpha_bounds=(1e-3, 1e3))

# ─────────────────────────────────────────────
# FIG 4 — Bootstrap stability (7500 samples)
# ─────────────────────────────────────────────
print("=== Bootstrap Stability Analysis ===")
N_BOOT = 7500
cc_boot, rmse_boot, mae_boot = [], [], []

for i in range(N_BOOT):
    if (i + 1) % 500 == 0:
        print(f"  Progress: {i+1}/{N_BOOT}")
    X_b, y_b = resample(X_scaled, y, random_state=i)
    gpr_b = GaussianProcessRegressor(kernel=rq_kernel, n_restarts_optimizer=3,
                                     normalize_y=True)
    gpr_b.fit(X_b, y_b)
    y_pred_b = gpr_b.predict(X_scaled)
    cc_boot.append(np.corrcoef(y, y_pred_b)[0, 1])
    rmse_boot.append(np.sqrt(mean_squared_error(y, y_pred_b)))
    mae_boot.append(mean_absolute_error(y, y_pred_b))

print(f"\nBootstrap averages over {N_BOOT} samples:")
print(f"  CC:   {np.mean(cc_boot)*100:.2f}%    (paper: 99.99%)")
print(f"  RMSE: {np.mean(rmse_boot):.4f} K  (paper: 0.6190)")
print(f"  MAE:  {np.mean(mae_boot):.4f} K  (paper: 0.1398)")

fig, axes = plt.subplots(3, 1, figsize=(7, 10))
for ax, data, label in zip(axes,
                            [cc_boot, rmse_boot, mae_boot],
                            ["CC", "RMSE (K)", "MAE (K)"]):
    ax.hist(data, bins=40, color="steelblue", edgecolor="k", linewidth=0.3)
    ax.axvline(np.mean(data), color="red", linestyle="--", label=f"Mean = {np.mean(data):.4f}")
    ax.set_xlabel(label)
    ax.set_ylabel("Frequency")
    ax.legend()
plt.suptitle("Bootstrap Stability Analysis (7500 samples)", fontsize=13)
plt.tight_layout()
out = FIGURES_DIR / "fig4_bootstrap_stability.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# ─────────────────────────────────────────────
# FIG 2 — Model performance vs training size
# ─────────────────────────────────────────────
print("\n=== Training Size Analysis ===")
sizes = list(range(30, 93)) + [93, 94]
cc_by_size, rmse_by_size, mae_by_size = [], [], []

for size in sizes:
    print(f"  Training size: {size}")
    cc_s, rmse_s, mae_s = [], [], []
    rng = np.random.default_rng(42)
    for _ in range(100):
        idx = rng.choice(len(X_scaled), size=size, replace=False)
        gpr_s = GaussianProcessRegressor(kernel=rq_kernel, n_restarts_optimizer=3,
                                         normalize_y=True)
        gpr_s.fit(X_scaled[idx], y[idx])
        yp = gpr_s.predict(X_scaled)
        cc_s.append(np.corrcoef(y, yp)[0, 1])
        rmse_s.append(np.sqrt(mean_squared_error(y, yp)))
        mae_s.append(mean_absolute_error(y, yp))

    cc_by_size.append(cc_s)
    rmse_by_size.append(rmse_s)
    mae_by_size.append(mae_s)

fig, axes = plt.subplots(3, 1, figsize=(12, 12))
for ax, data, label in zip(axes,
                            [cc_by_size, rmse_by_size, mae_by_size],
                            ["CC", "RMSE (K)", "MAE (K)"]):
    ax.boxplot(data, positions=sizes, widths=0.6, showfliers=True,
               flierprops=dict(marker="+", markersize=3))
    ax.set_xlabel("Training dataset size")
    ax.set_ylabel(label)
plt.suptitle("Model Performance vs Training Data Size", fontsize=13)
plt.tight_layout()
out = FIGURES_DIR / "fig2_training_size.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")