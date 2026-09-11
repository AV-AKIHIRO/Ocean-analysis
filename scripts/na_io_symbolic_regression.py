#!/usr/bin/env python3
"""
scripts/na_io_symbolic_regression.py
====================================
Discovers algebraic and differential equations connecting North Atlantic (C_NA)
and Indian Ocean (C_IO) CO2 concentrations using ALL AVAILABLE DATASET FEATURES
(Simulation outputs + ML Satellite features + Inter-basin gradients).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

WORKSPACE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(WORKSPACE, 'output')

def main():
    print("="*70)
    print("SYMBOLIC REGRESSION: FULL-FEATURE SEARCH ON ALL DATASET COLUMNS")
    print("="*70)

    # 1. Load Primary Datasets
    sim_path = os.path.join(OUTPUT_DIR, 'co2_salinity_simulation_1987_2014.csv')
    feat_path = os.path.join(OUTPUT_DIR, 'ml_features.csv')
    
    if not os.path.exists(sim_path):
        print(f"Error: {sim_path} not found.")
        return

    df_sim  = pd.read_csv(sim_path)
    df_sim['date'] = pd.to_datetime(df_sim['date'])
    df_sim = df_sim.set_index('date').resample('MS').first()

    df_feat = pd.read_csv(feat_path) if os.path.exists(feat_path) else pd.DataFrame()
    if not df_feat.empty:
        df_feat['date'] = pd.to_datetime(df_feat['date'])
        df_feat = df_feat.set_index('date').resample('MS').first()

    # Align dataframes
    common_idx = df_sim.index.intersection(df_feat.index) if not df_feat.empty else df_sim.index
    df_sim  = df_sim.loc[common_idx]
    df_feat = df_feat.loc[common_idx] if not df_feat.empty else pd.DataFrame()

    # Target: Indian Ocean Dissolved CO2
    C_IO = df_sim['C_IO'].values

    # Combine ALL columns from both datasets except target
    df_all_inputs = pd.concat([df_sim.drop(columns=['C_IO']), df_feat], axis=1)
    
    # Remove duplicate columns if any
    df_all_inputs = df_all_inputs.loc[:, ~df_all_inputs.columns.duplicated()]

    # Add physics-engineered gradient features
    df_all_inputs['S_diff_NA_IO'] = df_sim['S_NA'] - df_sim['S_IO']
    df_all_inputs['S_diff_SA_IO'] = df_sim['S_SA'] - df_sim['S_IO']
    df_all_inputs['S_diff_SO_IO'] = df_sim['S_SO'] - df_sim['S_IO']

    feature_cols = list(df_all_inputs.columns)
    print(f"\nTotal Input Candidate Features ({len(feature_cols)}): {feature_cols}")

    # -------------------------------------------------------------------------
    # PART 1: Full-Feature Symbolic Fitting (Lasso for Sparse Feature Selection + Ridge)
    # -------------------------------------------------------------------------
    print("\n--- PART 1: Full-Feature Algebraic Symbolic Regression ---")
    
    # Polynomial degree 2 over candidate features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(df_all_inputs)
    poly_feature_names = poly.get_feature_names_out(feature_cols)

    print(f"Generated {X_poly.shape[1]} polynomial candidate terms.")

    # Use Lasso with feature scaling for robust sparse selection
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_poly_sc = scaler.fit_transform(X_poly)

    lasso = Lasso(alpha=1e-4, max_iter=20000, random_state=42)
    lasso.fit(X_poly_sc, C_IO)
    
    selected_mask = np.abs(lasso.coef_) > 1e-4
    if not np.any(selected_mask):
        selected_mask = np.argsort(np.abs(lasso.coef_))[-5:] # Pick top 5 if none exceed threshold
        
    selected_terms = poly_feature_names[selected_mask]
    
    print(f"\nLasso selected {len(selected_terms)} top terms out of {len(poly_feature_names)} polynomial terms:")
    for term in selected_terms:
        print(f"  - {term}")

    # Refit with Ridge on selected sparse terms for stable non-zero coefficients
    X_selected = X_poly_sc[:, selected_mask]
    model_alg = Ridge(alpha=1e-3)
    model_alg.fit(X_selected, C_IO)
    pred_alg = model_alg.predict(X_selected)

    r2_alg = r2_score(C_IO, pred_alg)
    rmse_alg = np.sqrt(mean_squared_error(C_IO, pred_alg)) * 1000  # in mmol/m3

    print(f"\nFull-Feature Algebraic Fit R²: {r2_alg:.6f} | RMSE: {rmse_alg:.6f} mmol/m³")
    
    eq_terms = [f"{model_alg.intercept_:.6f}"]
    for coef, name in zip(model_alg.coef_, selected_terms):
        eq_terms.append(f"({coef:+.6e} * {name})")
    algebraic_eq_str = "C_IO = " + " ".join(eq_terms)
    print("  " + algebraic_eq_str)

    # -------------------------------------------------------------------------
    # PART 2: Differential Equation Symbolic Fitting (dC_IO/dt = f(all_inputs))
    # -------------------------------------------------------------------------
    print("\n--- PART 2: Full-Feature Differential Equation Symbolic Regression ---")
    
    dC_IO = np.diff(C_IO, prepend=C_IO[0])
    
    # Differential inputs
    df_diff_inputs = df_all_inputs.diff().fillna(0)
    df_diff_inputs['C_NA'] = df_sim['C_NA'] # static level reference
    
    poly_diff = PolynomialFeatures(degree=1, include_bias=False)
    X_diff_poly = poly_diff.fit_transform(df_diff_inputs)
    diff_feature_names = poly_diff.get_feature_names_out(df_diff_inputs.columns)

    model_diff = Ridge(alpha=1e-5)
    model_diff.fit(X_diff_poly, dC_IO)
    pred_dC_IO = model_diff.predict(X_diff_poly)

    r2_diff = r2_score(dC_IO, pred_dC_IO)
    rmse_diff = np.sqrt(mean_squared_error(dC_IO, pred_dC_IO)) * 1000

    print(f"Differential Fit R²: {r2_diff:.6f} | Increment RMSE: {rmse_diff:.6f} mmol/m³")
    
    # Reconstruct trajectory
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
    
    axes[0].plot(df_sim.index, C_IO, 'k-', lw=2.0, label='Actual ODE C_IO')
    axes[0].plot(df_sim.index, pred_alg, 'r--', lw=1.5, label='Full-Feature Symbolic Fit')
    axes[0].set_ylabel('C_IO [mol/m³]')
    axes[0].set_title(f'Full-Feature Symbolic Regression Algebraic Discovery (R² = {r2_alg:.6f})\n{algebraic_eq_str[:120]}...', fontsize=9)
    axes[0].legend()
    axes[0].grid(True, ls='--', alpha=0.5)

    axes[1].plot(df_sim.index, C_IO, 'k-', lw=2.0, label='Actual ODE C_IO')
    axes[1].plot(df_sim.index, C_IO_rec, 'b-.', lw=1.5, label='Symbolic Differential Trajectory')
    axes[1].set_ylabel('C_IO [mol/m³]')
    axes[1].set_title(f'Symbolic Differential Discovery (Reconstructed R² = {r2_rec:.6f})', fontsize=9)
    axes[1].set_xlabel('Year')
    axes[1].legend()
    axes[1].grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    out_img = os.path.join(OUTPUT_DIR, 'symbolic_regression_na_io.png')
    plt.savefig(out_img, dpi=130)
    plt.close()
    print(f"\nSaved visualization to {out_img}")

    # Save summary text
    summary_txt = os.path.join(OUTPUT_DIR, 'symbolic_regression_summary.txt')
    with open(summary_txt, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FULL-FEATURE SYMBOLIC REGRESSION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"1. ALGEBRAIC FIT R2: {r2_alg:.6f}\n")
        f.write(f"   Equation: {algebraic_eq_str}\n\n")
        f.write(f"2. RECONSTRUCTED TRAJECTORY R2: {r2_rec:.6f}\n")
    print(f"Saved summary to {summary_txt}")

if __name__ == '__main__':
    main()
