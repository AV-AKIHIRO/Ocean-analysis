"""
NA–IO CO₂ Relationship Analysis — Step 1
========================================
Explores the statistical and empirical relationship between North Atlantic (NA)
and Indian Ocean (IO) dissolved CO₂ concentrations from the physics ODE simulation.

Produces a 4-panel figure:
  Panel 1 — Time series (C_NA and C_IO, 1987–2014)
  Panel 2 — Scatter plot C_NA vs C_IO (colour = year, linear fit)
  Panel 3 — Time-lagged cross-correlation (±24 months)
  Panel 4 — Monthly increments ΔC_NA vs ΔC_IO (linear fit)

Outputs:
  output/na_io_co2_relationship.png
  output/na_io_co2_stats.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import pearsonr
from scipy.signal import correlate
from sklearn.linear_model import LinearRegression

# ── Paths ──────────────────────────────────────────────────────────────────────
CSV_PATH  = "output/co2_salinity_simulation_1987_2014.csv"
OUT_FIG   = "output/na_io_co2_relationship.png"
OUT_STATS = "output/na_io_co2_stats.csv"

# ── 1. Load and resample to monthly ────────────────────────────────────────────
print("Loading ODE simulation data...")
df_raw = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)

# The ODE uses adaptive timesteps → resample to one value per calendar month
# (take the last value in each month — closest to month-end steady state)
df = df_raw[["C_NA", "C_IO"]].resample("ME").last().dropna()
print(f"Monthly timesteps after resampling: {len(df)}")
print(df.head(3))
print(df.tail(3))

C_NA  = df["C_NA"].values        # mmol/m³
C_IO  = df["C_IO"].values
years = df.index.year.values
t     = np.arange(len(df))       # month index 0..329

# ── 2. Linear regression: C_IO = a·C_NA + b ────────────────────────────────────
reg = LinearRegression().fit(C_NA.reshape(-1, 1), C_IO)
a, b = reg.coef_[0], reg.intercept_
C_IO_pred = reg.predict(C_NA.reshape(-1, 1))
r_sq = reg.score(C_NA.reshape(-1, 1), C_IO)
r_pearson, p_val = pearsonr(C_NA, C_IO)

print(f"\nLinear fit: C_IO = {a:.4f} · C_NA + {b:.4f}")
print(f"  R² = {r_sq:.6f},  Pearson r = {r_pearson:.6f},  p = {p_val:.2e}")

# ── 3. Time-lagged cross-correlation (±24 months) ─────────────────────────────
max_lag = 24
C_NA_norm = (C_NA - C_NA.mean()) / C_NA.std()
C_IO_norm = (C_IO - C_IO.mean()) / C_IO.std()

# scipy correlate gives full cross-correlation; extract the ±max_lag window
full_corr  = correlate(C_IO_norm, C_NA_norm, mode="full") / len(C_NA)
mid        = len(full_corr) // 2
lags       = np.arange(-max_lag, max_lag + 1)
xcorr      = full_corr[mid - max_lag : mid + max_lag + 1]

optimal_lag = lags[np.argmax(xcorr)]
max_xcorr   = xcorr.max()
print(f"\nCross-correlation peak: r = {max_xcorr:.4f} at lag = {optimal_lag} months")

# ── 4. Monthly increments ΔC ──────────────────────────────────────────────────
dC_NA = np.diff(C_NA)
dC_IO = np.diff(C_IO)

reg_delta = LinearRegression().fit(dC_NA.reshape(-1, 1), dC_IO)
alpha     = reg_delta.coef_[0]
r_delta, p_delta = pearsonr(dC_NA, dC_IO)
r_sq_delta = reg_delta.score(dC_NA.reshape(-1, 1), dC_IO)

print(f"\nIncrement fit: ΔC_IO = {alpha:.4f} · ΔC_NA")
print(f"  R² = {r_sq_delta:.6f},  Pearson r = {r_delta:.6f},  p = {p_delta:.2e}")

# ── 5. Save statistics CSV ─────────────────────────────────────────────────────
stats = pd.DataFrame([{
    "Metric": "Linear fit slope (a)",
    "Value": round(a, 6), "Unit": "dimensionless"
}, {
    "Metric": "Linear fit intercept (b)",
    "Value": round(b, 6), "Unit": "mmol/m³"
}, {
    "Metric": "R² (C_IO vs C_NA)",
    "Value": round(r_sq, 6), "Unit": "—"
}, {
    "Metric": "Pearson r (C_IO vs C_NA)",
    "Value": round(r_pearson, 6), "Unit": "—"
}, {
    "Metric": "p-value (C_IO vs C_NA)",
    "Value": float(f"{p_val:.2e}"), "Unit": "—"
}, {
    "Metric": "Optimal lag (cross-correlation)",
    "Value": int(optimal_lag), "Unit": "months"
}, {
    "Metric": "Peak cross-correlation value",
    "Value": round(max_xcorr, 6), "Unit": "—"
}, {
    "Metric": "Increment ratio α (ΔC_IO/ΔC_NA)",
    "Value": round(alpha, 6), "Unit": "dimensionless"
}, {
    "Metric": "R² (ΔC_IO vs ΔC_NA)",
    "Value": round(r_sq_delta, 6), "Unit": "—"
}, {
    "Metric": "Pearson r (ΔC_IO vs ΔC_NA)",
    "Value": round(r_delta, 6), "Unit": "—"
}])
stats.to_csv(OUT_STATS, index=False)
print(f"\nStats saved → {OUT_STATS}")

# ── 6. Four-panel figure ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "North Atlantic ↔ Indian Ocean  CO₂ Relationship\n"
    "ODE Simulation  |  1987–2014  (330 monthly timesteps)",
    fontsize=14, fontweight="bold", y=1.01
)

cmap   = cm.plasma
y_norm = (years - years.min()) / (years.max() - years.min())
colors = cmap(y_norm)

# ── Panel 1: Time series ──────────────────────────────────────────────────────
ax1 = axes[0, 0]
color_na = "#2E86AB"
color_io = "#E84855"
ln1 = ax1.plot(df.index, C_NA * 1000, color=color_na, lw=1.6, label="C_NA")
ax1b = ax1.twinx()
ln2 = ax1b.plot(df.index, C_IO * 1000, color=color_io, lw=1.6, ls="--", label="C_IO")
ax1.set_ylabel("C_NA (µmol/m³)", color=color_na)
ax1b.set_ylabel("C_IO (µmol/m³)", color=color_io)
ax1.tick_params(axis="y", colors=color_na)
ax1b.tick_params(axis="y", colors=color_io)
ax1.set_title("Panel 1 — CO₂ Time Series (dual axis)", fontweight="bold")
ax1.set_xlabel("Year")
ax1.grid(True, alpha=0.3)
lns = ln1 + ln2
ax1.legend(lns, [l.get_label() for l in lns], loc="upper left", fontsize=9)

# ── Panel 2: Scatter C_NA vs C_IO ────────────────────────────────────────────
ax2 = axes[0, 1]
sc = ax2.scatter(C_NA * 1000, C_IO * 1000, c=years, cmap="plasma",
                 s=18, alpha=0.85, zorder=3)
x_line = np.linspace(C_NA.min(), C_NA.max(), 200)
ax2.plot(x_line * 1000, (a * x_line + b) * 1000,
         color="black", lw=1.8, ls="-",
         label=f"C_IO = {a:.3f}·C_NA + {b:.3f}\nR² = {r_sq:.4f}")
cbar = plt.colorbar(sc, ax=ax2)
cbar.set_label("Year", fontsize=9)
ax2.set_xlabel("C_NA (µmol/m³)")
ax2.set_ylabel("C_IO (µmol/m³)")
ax2.set_title("Panel 2 — Scatter: C_NA vs C_IO", fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Annotate Pearson r
ax2.annotate(f"Pearson r = {r_pearson:.4f}\np = {p_val:.1e}",
             xy=(0.04, 0.88), xycoords="axes fraction", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

# ── Panel 3: Cross-correlation ────────────────────────────────────────────────
ax3 = axes[1, 0]
bar_colors = ["#E84855" if x == optimal_lag else "#2E86AB" for x in lags]
ax3.bar(lags, xcorr, color=bar_colors, width=0.7, alpha=0.85)
ax3.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax3.axhline(0, color="black", lw=0.5)
ax3.set_xlabel("Lag (months)  [+ = IO lags NA]")
ax3.set_ylabel("Normalised cross-correlation")
ax3.set_title("Panel 3 — Time-Lagged Cross-Correlation", fontweight="bold")
ax3.annotate(f"Peak lag = {optimal_lag} months\nr = {max_xcorr:.4f}",
             xy=(0.04, 0.06), xycoords="axes fraction", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax3.grid(True, alpha=0.3)

# ── Panel 4: ΔC_NA vs ΔC_IO ──────────────────────────────────────────────────
ax4 = axes[1, 1]
ax4.scatter(dC_NA * 1000, dC_IO * 1000, color="#6A4C93", s=12, alpha=0.6, zorder=3)
x_d = np.linspace(dC_NA.min(), dC_NA.max(), 200)
ax4.plot(x_d * 1000, (alpha * x_d) * 1000,
         color="black", lw=1.8,
         label=f"ΔC_IO = {alpha:.3f}·ΔC_NA\nR² = {r_sq_delta:.4f}")
ax4.axhline(0, color="gray", lw=0.6, ls="--")
ax4.axvline(0, color="gray", lw=0.6, ls="--")
ax4.set_xlabel("ΔC_NA (µmol/m³ / month)")
ax4.set_ylabel("ΔC_IO (µmol/m³ / month)")
ax4.set_title("Panel 4 — Monthly Increments: ΔC_NA vs ΔC_IO", fontweight="bold")
ax4.legend(fontsize=9)
ax4.annotate(f"Pearson r = {r_delta:.4f}",
             xy=(0.04, 0.92), xycoords="axes fraction", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved → {OUT_FIG}")
print("\nDone.")
