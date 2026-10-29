#!/usr/bin/env python3
"""
scripts/ocean_sindy_discovery.py
=================================
Automated Dynamic Differential Equation Discovery for Ocean Carbon & Salinity
using PySINDy (Sparse Identification of Non-linear Dynamics).

Discovers:
  1. Governing ODE system for dissolved CO2 across all 5 ocean basins (NA, SA, SO, PO, IO)
  2. Targeted NA-IO dynamic coupling ODE for Indian Ocean carbon uptake:
       dC_IO/dt = f(C_IO, C_NA, S_diff, pCO2_air)
  3. Reconstructs full dynamic trajectories from initial boundary conditions (1987-2014)
     to benchmark against numerical ODE truth.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pysindy as ps
from sklearn.metrics import r2_score, mean_squared_error

WORKSPACE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(WORKSPACE, 'output')
SIM_CSV    = os.path.join(OUTPUT_DIR, 'co2_salinity_simulation_1987_2014.csv')

BASINS = ['NA', 'SA', 'SO', 'PO', 'IO']
C_COLS = [f'C_{b}' for b in BASINS]
S_COLS = [f'S_{b}' for b in BASINS]

def load_data():
    print(f"Loading simulation dataset from: {SIM_CSV}")
    df_sim = pd.read_csv(SIM_CSV)
    df_sim['date'] = pd.to_datetime(df_sim['date'])
    
    # Time in years from start
    t0 = df_sim['date'].iloc[0]
    t_years = (df_sim['date'] - t0).dt.total_seconds().values / (365.25 * 86400.0)
    
    # Keeling atmospheric pCO2 [uatm]
    dates_dt = pd.DatetimeIndex(df_sim['date'])
    months_from_2000 = (dates_dt.year - 2000) * 12.0 + (dates_dt.month - 1) + (dates_dt.day / 30.0)
    pCO2_air = 367.41 + 0.18152 * months_from_2000
    df_sim['pCO2_air'] = pCO2_air
    df_sim['pCO2_dev'] = (pCO2_air - 367.41) / 50.0  # normalized deviation around year 2000
    
    # Salinity differences (driving thermohaline exchange)
    df_sim['S_diff_NA_IO'] = df_sim['S_NA'] - df_sim['S_IO']
    df_sim['S_diff_SA_IO'] = df_sim['S_SA'] - df_sim['S_IO']
    df_sim['S_diff_SO_IO'] = df_sim['S_SO'] - df_sim['S_IO']
    df_sim['S_diff_PO_IO'] = df_sim['S_PO'] - df_sim['S_IO']
    
    return df_sim, t_years

def run_multibasin_sindy(df_sim, t_years):
    print("\n" + "="*70)
    print("1. MULTI-BASIN CO2 DYNAMICAL SYSTEM DISCOVERY (5-BOX ODE)")
    print("="*70)
    
    # States in mmol/m3 (scaled by 1000 for standard numerical conditioning)
    X = df_sim[C_COLS].values * 1000.0
    
    # Exogenous control inputs: pCO2_dev, S_diff_NA_IO, S_diff_SO_IO
    U = np.column_stack([
        df_sim['pCO2_dev'].values,
        df_sim['S_diff_NA_IO'].values,
        df_sim['S_diff_SO_IO'].values
    ])
    u_names = ['pCO2_dev', 'Sdiff_NA_IO', 'Sdiff_SO_IO']
    
    # Feature library: polynomial degree 1 + selective degree 2
    feature_lib = ps.PolynomialLibrary(degree=1, include_bias=True)
    optimizer = ps.STLSQ(threshold=5e-5, alpha=1e-6, max_iter=100)
    
    sindy_model = ps.SINDy(feature_library=feature_lib, optimizer=optimizer)
    sindy_model.fit(X, u=U, t=t_years, feature_names=C_COLS + u_names)
    
    lhs_names = [f'd({c})/dt' for c in C_COLS]
    print("\nDiscovered 5-Basin CO2 Governing Differential Equations:")
    sindy_model.print(lhs=lhs_names)
    
    # Reconstruct trajectory via numerical simulation
    dt = np.diff(t_years)
    X_sim = np.zeros_like(X)
    X_sim[0] = X[0]
    for i in range(len(dt)):
        dxdt = sindy_model.predict(X_sim[i:i+1], u=U[i:i+1])[0]
        X_sim[i+1] = X_sim[i] + dxdt * dt[i]
        
    r2_scores = [r2_score(X[:, i], X_sim[:, i]) for i in range(5)]
    rmse_scores = [np.sqrt(mean_squared_error(X[:, i], X_sim[:, i])) for i in range(5)]
    
    print("\nReconstructed Trajectory Accuracy (1987-2014):")
    for i, b in enumerate(BASINS):
        print(f"  Basin {b}: R² = {r2_scores[i]:.6f}, RMSE = {rmse_scores[i]:.6e} mmol/m³")
        
    return sindy_model, X, X_sim, r2_scores, rmse_scores, U, u_names

def run_targeted_na_io_sindy(df_sim, t_years):
    print("\n" + "="*70)
    print("2. TARGETED NA–IO DYNAMICAL COUPLING DISCOVERY (dC_IO/dt)")
    print("="*70)
    
    # Target states in mmol/m3
    X_pair = df_sim[['C_IO', 'C_NA']].values * 1000.0
    
    # Driving physical forces:
    U_pair = np.column_stack([
        df_sim['pCO2_dev'].values,
        df_sim['S_diff_NA_IO'].values,
        df_sim['S_diff_SO_IO'].values
    ])
    feat_names = ['C_IO', 'C_NA', 'pCO2_dev', 'Sdiff_NA_IO', 'Sdiff_SO_IO']
    
    poly_lib = ps.PolynomialLibrary(degree=2, include_bias=True)
    optimizer = ps.STLSQ(threshold=1e-4, alpha=1e-6, max_iter=100)
    
    io_sindy = ps.SINDy(feature_library=poly_lib, optimizer=optimizer)
    io_sindy.fit(X_pair, u=U_pair, t=t_years, feature_names=feat_names)
    
    print("\nDiscovered NA-IO Coupled Dynamic Equations:")
    io_sindy.print(lhs=['d(C_IO)/dt', 'd(C_NA)/dt'])
    
    # Simulate trajectory
    dt = np.diff(t_years)
    X_sim_pair = np.zeros_like(X_pair)
    X_sim_pair[0] = X_pair[0]
    for i in range(len(dt)):
        dxdt = io_sindy.predict(X_sim_pair[i:i+1], u=U_pair[i:i+1])[0]
        X_sim_pair[i+1] = X_sim_pair[i] + dxdt * dt[i]
        
    r2_io = r2_score(X_pair[:, 0], X_sim_pair[:, 0])
    r2_na = r2_score(X_pair[:, 1], X_sim_pair[:, 1])
    rmse_io = np.sqrt(mean_squared_error(X_pair[:, 0], X_sim_pair[:, 0]))
    rmse_na = np.sqrt(mean_squared_error(X_pair[:, 1], X_sim_pair[:, 1]))
    
    print(f"\nTargeted Indian Ocean Trajectory: R² = {r2_io:.6f}, RMSE = {rmse_io:.6e} mmol/m³")
    print(f"Targeted North Atlantic Trajectory: R² = {r2_na:.6f}, RMSE = {rmse_na:.6e} mmol/m³")
    
    return io_sindy, X_pair, X_sim_pair, r2_io, rmse_io

def plot_sindy_results(df_sim, t_years, X, X_sim, X_pair, X_sim_pair, sindy_model):
    print("\nGenerating SINDy figures...")
    dates = df_sim['date'].values
    
    # Plot 1: Targeted NA-IO Reconstructed Trajectory
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    
    ax1.plot(dates, X_pair[:, 0], label='Physical ODE Truth ($C_{\\mathrm{IO}}$)', color='#9467bd', lw=2.2)
    ax1.plot(dates, X_sim_pair[:, 0], '--', label='SINDy Discovered Trajectory', color='#ff7f0e', lw=2.0)
    ax1.set_ylabel('Indian Ocean $C_{\\mathrm{IO}}$\n(mmol/m³)', fontsize=11, fontweight='bold')
    ax1.set_title('PySINDy Automated Differential Equation Discovery: Indian Ocean CO₂ Trajectory', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', frameon=True)
    
    ax2.plot(dates, X_pair[:, 1], label='Physical ODE Truth ($C_{\\mathrm{NA}}$)', color='#1f77b4', lw=2.2)
    ax2.plot(dates, X_sim_pair[:, 1], '--', label='SINDy Discovered Trajectory', color='#2ca02c', lw=2.0)
    ax2.set_ylabel('North Atlantic $C_{\\mathrm{NA}}$\n(mmol/m³)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Year (1987–2014)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    out_na_io = os.path.join(OUTPUT_DIR, 'sindy_na_io_co2_trajectory.png')
    plt.savefig(out_na_io, dpi=160)
    plt.close()
    print(f"Saved: {out_na_io}")
    
    # Plot 2: All 5 Basins Trajectory Comparison
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    for i, b in enumerate(BASINS):
        axes[i].plot(dates, X[:, i], label=f'Physical ODE ({b})', color=colors[i], lw=2.0)
        axes[i].plot(dates, X_sim[:, i], '--', label=f'SINDy Discovered ({b})', color='black', lw=1.5, alpha=0.85)
        axes[i].set_ylabel(f'{b} (mmol/m³)', fontsize=10, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper left', fontsize=9)
    axes[-1].set_xlabel('Year (1987–2014)', fontsize=11, fontweight='bold')
    axes[0].set_title('PySINDy Discovered 5-Basin Global Ocean Carbon Differential Equations vs Physics ODE', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_5b = os.path.join(OUTPUT_DIR, 'sindy_all_basins_co2.png')
    plt.savefig(out_5b, dpi=160)
    plt.close()
    print(f"Saved: {out_5b}")
    
    # Plot 3: SINDy Coefficient Matrix Heatmap
    coef_matrix = sindy_model.coefficients()
    feat_names = sindy_model.get_feature_names()
    
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(coef_matrix, cmap='coolwarm', aspect='auto')
    ax.set_xticks(np.arange(len(feat_names)))
    ax.set_yticks(np.arange(len(BASINS)))
    ax.set_xticklabels(feat_names, rotation=35, ha='right', fontsize=9)
    ax.set_yticklabels([f'd({b})/dt' for b in BASINS], fontsize=10, fontweight='bold')
    
    # Annotate values
    for i in range(len(BASINS)):
        for j in range(len(feat_names)):
            val = coef_matrix[i, j]
            if abs(val) > 1e-4:
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        color='white' if abs(val) > 0.05 else 'black', fontsize=8)
                        
    plt.colorbar(im, ax=ax, label='Discovered Differential Coupling Weight')
    ax.set_title('PySINDy Discovered Sparse Coefficient Matrix (Dynamic Coupling)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    out_heat = os.path.join(OUTPUT_DIR, 'sindy_coefficients_heatmap.png')
    plt.savefig(out_heat, dpi=160)
    plt.close()
    print(f"Saved: {out_heat}")

def save_summary(sindy_5b, io_sindy, r2_scores, r2_io, rmse_io):
    summary_path = os.path.join(OUTPUT_DIR, 'sindy_discovery_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PySINDy DYNAMIC DIFFERENTIAL EQUATION DISCOVERY SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write("1. 5-Basin Discovered ODE System (dC/dt [mmol/(m^3*year)]):\n")
        for i, b in enumerate(BASINS):
            f.write(f"  d({b})/dt = {sindy_5b.equations()[i]}\n")
            f.write(f"    --> Trajectory R²: {r2_scores[i]:.6f}\n")
        f.write("\n" + "-"*70 + "\n")
        f.write("2. Targeted NA-IO Dynamic Coupling ODE (Indian Ocean):\n")
        f.write(f"  d(C_IO)/dt = {io_sindy.equations()[0]}\n")
        f.write(f"    --> Trajectory R²: {r2_io:.6f}\n")
        f.write(f"    --> Trajectory RMSE: {rmse_io:.6e} mmol/m³\n")
        f.write(f"  d(C_NA)/dt = {io_sindy.equations()[1]}\n")
        f.write("="*70 + "\n")
    print(f"Saved summary to: {summary_path}")

def main():
    df_sim, t_years = load_data()
    sindy_5b, X, X_sim, r2_scores, rmse_scores, U, u_names = run_multibasin_sindy(df_sim, t_years)
    io_sindy, X_pair, X_sim_pair, r2_io, rmse_io = run_targeted_na_io_sindy(df_sim, t_years)
    plot_sindy_results(df_sim, t_years, X, X_sim, X_pair, X_sim_pair, sindy_5b)
    save_summary(sindy_5b, io_sindy, r2_scores, r2_io, rmse_io)
    print("\n[SUCCESS] PySINDy Dynamic Discovery Complete!")

if __name__ == '__main__':
    main()
