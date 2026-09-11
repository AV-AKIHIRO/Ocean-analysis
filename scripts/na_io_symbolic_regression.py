#!/usr/bin/env python3
"""
scripts/na_io_symbolic_regression.py
====================================
Discovers algebraic and differential equations connecting North Atlantic (C_NA)
and Indian Ocean (C_IO) CO2 concentrations using Symbolic Regression techniques
(Scikit-Learn Polynomial & Non-linear Feature Selection + Ridge Symbolic Fitting).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

WORKSPACE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(WORKSPACE, 'output')

def main():
    print("="*70)
    print("SYMBOLIC REGRESSION: NA–IO CO2 RELATIONSHIP & DIFFERENTIAL EQUATION SEARCH")
    print("="*70)

    # 1. Load Simulation Output
    sim_path = os.path.join(OUTPUT_DIR, 'co2_salinity_simulation_1987_2014.csv')
    if not os.path.exists(sim_path):
        print(f"Error: {sim_path} not found.")
        return

    df = pd.read_csv(sim_path, index_col=0, parse_dates=True)
    
    # Load target and features from simulation + ML features
    C_IO = df['C_IO'].values
    C_NA = df['C_NA'].values
    S_NA = df['S_NA'].values
    S_IO = df['S_IO'].values
    S_diff = S_NA - S_IO  # Salinity gradient driving thermohaline flow
    
    # Load pCO2_air and F_IO from ml_features if available or construct Keeling
    ml_feat_path = os.path.join(OUTPUT_DIR, 'ml_features.csv')
    if os.path.exists(ml_feat_path):
        df_feat = pd.read_csv(ml_feat_path, index_col=0, parse_dates=True)
        if len(df_feat) == len(df):
            pCO2_air = df_feat['pCO2_air'].values if 'pCO2_air' in df_feat.columns else (367.41 + 0.18152 * np.arange(len(df)))
            F_IO = df_feat['F_IO'].values if 'F_IO' in df_feat.columns else np.zeros(len(df))
        else:
            pCO2_air = (367.41 + 0.18152 * np.linspace(0, len(df)/12.0, len(df)))
            F_IO = np.zeros(len(df))
    else:
        pCO2_air = (367.41 + 0.18152 * np.linspace(0, len(df)/12.0, len(df)))
        F_IO = np.zeros(len(df))
    
    # Monthly Increments (Differential Equation components)
    dC_IO = np.diff(C_IO, prepend=C_IO[0])
    dC_NA = np.diff(C_NA, prepend=C_NA[0])

    print(f"Loaded {len(df)} monthly timesteps (1987–2014).")

    # -------------------------------------------------------------------------
    # PART 1: Direct Algebraic Symbolic Fitting (C_IO = f(C_NA, S_diff, pCO2_air))
    # -------------------------------------------------------------------------
    print("\n--- PART 1: Algebraic Symbolic Regression ---")
    
    X_alg = pd.DataFrame({
        'C_NA': C_NA,
        'S_diff': S_diff,
        'pCO2_air': pCO2_air
    })

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_alg)
    feature_names = poly.get_feature_names_out(X_alg.columns)

    model_alg = Ridge(alpha=1e-3)
    model_alg.fit(X_poly, C_IO)
    pred_alg = model_alg.predict(X_poly)

    r2_alg = r2_score(C_IO, pred_alg)
    rmse_alg = np.sqrt(mean_squared_error(C_IO, pred_alg)) * 1000  # in mmol/m3

    print(f"Algebraic Fit R²: {r2_alg:.6f} | RMSE: {rmse_alg:.4f} mmol/m³")
    print("\nDiscovered Algebraic Equation:")
    eq_terms = [f"{model_alg.intercept_:.6f}"]
    for coef, name in zip(model_alg.coef_, feature_names):
        if abs(coef) > 1e-8:
            eq_terms.append(f"({coef:+.6e} * {name})")
    algebraic_eq_str = "C_IO = " + " ".join(eq_terms)
    print("  " + algebraic_eq_str)

    # -------------------------------------------------------------------------
    # PART 2: Differential Equation Symbolic Fitting (dC_IO/dt = f(dC_NA/dt, diseq_IO, S_diff))
    # -------------------------------------------------------------------------
    print("\n--- PART 2: Differential Equation Symbolic Regression ---")
    
    # Physics Disequilibrium term
    diseq_IO = (0.0348 * pCO2_air) - C_IO

    X_diff = pd.DataFrame({
        'dC_NA': dC_NA,
        'diseq_IO': diseq_IO,
        'S_diff': S_diff
    })

    poly_diff = PolynomialFeatures(degree=2, include_bias=False)
    X_diff_poly = poly_diff.fit_transform(X_diff)
    diff_feature_names = poly_diff.get_feature_names_out(X_diff.columns)

    model_diff = Ridge(alpha=1e-5)
    model_diff.fit(X_diff_poly, dC_IO)
    pred_dC_IO = model_diff.predict(X_diff_poly)

    r2_diff = r2_score(dC_IO, pred_dC_IO)
    rmse_diff = np.sqrt(mean_squared_error(dC_IO, pred_dC_IO)) * 1000

    print(f"Differential Fit R²: {r2_diff:.6f} | Increment RMSE: {rmse_diff:.4f} mmol/m³")
    print("\nDiscovered Governing Differential Equation:")
    diff_terms = [f"{model_diff.intercept_:.6e}"]
    for coef, name in zip(model_diff.coef_, diff_feature_names):
        if abs(coef) > 1e-8:
            diff_terms.append(f"({coef:+.6e} * {name})")
    diff_eq_str = "d(C_IO)/dt = " + " ".join(diff_terms)
    print("  " + diff_eq_str)

    # Reconstruct trajectory from integrated dC_IO
    C_IO_rec = np.zeros_like(C_IO)
    C_IO_rec[0] = C_IO[0]
    for t in range(1, len(C_IO)):
        C_IO_rec[t] = C_IO_rec[t-1] + pred_dC_IO[t]
    
    r2_rec = r2_score(C_IO, C_IO_rec)
    print(f"Reconstructed Trajectory R²: {r2_rec:.6f}")

    # -------------------------------------------------------------------------
    # PART 3: Visualization & Saving Output
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(df.index, C_IO, 'k-', lw=2.0, label='Actual ODE C_IO')
    axes[0].plot(df.index, pred_alg, 'r--', lw=1.5, label='Symbolic Algebraic Fit')
    axes[0].set_ylabel('C_IO [mol/m³]')
    axes[0].set_title(f'Symbolic Regression Algebraic Discovery (R² = {r2_alg:.5f})\n{algebraic_eq_str}', fontsize=10)
    axes[0].legend()
    axes[0].grid(True, ls='--', alpha=0.5)

    axes[1].plot(df.index, C_IO, 'k-', lw=2.0, label='Actual ODE C_IO')
    axes[1].plot(df.index, C_IO_rec, 'b-.', lw=1.5, label='Symbolic Differential Trajectory')
    axes[1].set_ylabel('C_IO [mol/m³]')
    axes[1].set_title(f'Symbolic Regression Differential Discovery (Reconstructed R² = {r2_rec:.5f})\n{diff_eq_str}', fontsize=10)
    axes[1].set_xlabel('Year')
    axes[1].legend()
    axes[1].grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, 'symbolic_regression_na_io.png')
    plt.savefig(out_img, dpi=130)
    plt.close()
    print(f"\nSaved visualization to {out_img}")

    # Save equation metrics summary to text file
    summary_txt = os.path.join(OUTPUT_DIR, 'symbolic_regression_summary.txt')
    with open(summary_txt, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SYMBOLIC REGRESSION RESULTS: NA–IO CO2 RELATIONSHIP\n")
        f.write("="*70 + "\n\n")
        f.write(f"1. ALGEBRAIC DISCOVERY (R2 = {r2_alg:.6f}):\n")
        f.write(f"   {algebraic_eq_str}\n\n")
        f.write(f"2. GOVERNING DIFFERENTIAL EQUATION DISCOVERY (R2 = {r2_diff:.6f}):\n")
        f.write(f"   {diff_eq_str}\n\n")
        f.write(f"3. RECONSTRUCTED TRAJECTORY R2: {r2_rec:.6f}\n")
    print(f"Saved summary to {summary_txt}")

if __name__ == '__main__':
    main()
