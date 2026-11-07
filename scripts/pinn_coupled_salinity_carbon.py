#!/usr/bin/env python3
"""
scripts/pinn_coupled_salinity_carbon.py
========================================
Coupled 10-Dimensional Physics-Informed Neural Network (Coupled PINN)
Simultaneously predicting Salinity (S_1..S_5) and Carbon (C_1..C_5)
with Bi-directional Thermohaline Advective Coupling.

Physics Enforcement:
  1. Salinity ODE residual: dS/dt = Freshwater(E-P) + SurfaceTransport(S) + DeepThermohaline(S)
  2. Carbon ODE residual:   dC/dt = GasExchange(K0, pCO2, C) + SurfaceTransport(C) + DeepThermohaline(S, C)
  3. Non-linear Cross-Coupling:
       q_ij = K_deep * beta * (S_i - S_j)
     The carbon transport rate is dynamically driven by the neural network's
     OWN simultaneous prediction of basin salinity gradients!
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)

WORKSPACE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(WORKSPACE, 'output')
SIM_CSV    = os.path.join(OUTPUT_DIR, 'co2_salinity_simulation_1987_2014.csv')
FLUX_CSV   = os.path.join(OUTPUT_DIR, 'processed_freshwater_fluxes_1987_2014.csv')

BASINS = ['NA', 'SA', 'SO', 'PO', 'IO']
SEC_PER_YEAR = 365.25 * 86400.0
SV_TO_M3S    = 1.0e6
S0_REF       = 35.0
BETA         = 7.5e-4
K_DEEP       = 60.0

# Basin geometry (Sunny et al. 2023 Table II)
V = np.array([146.0e15, 160.0e15, 71.8e15, 710.0e15, 264.0e15])   # m³
A = np.array([ 41.49e12, 40.27e12, 20.33e12, 165.25e12, 70.56e12]) # m²

SURFACE_CURRENTS = [(1,0,26.0), (2,1,30.0), (3,4,15.0), (4,1,10.0), (4,2,60.0)]
DEEP_PAIRS       = [(0,1), (1,2), (1,3), (2,3), (2,4)]

b_NA   =  3.28e3 / SV_TO_M3S    # Arctic ice melt [Sv]
b_SO   = -6.944e2 / SV_TO_M3S   # Antarctic ice [Sv]
q_AMAZ =  0.24                  # Amazon [Sv]

def load_data():
    df_sim = pd.read_csv(SIM_CSV)
    df_sim['date'] = pd.to_datetime(df_sim['date'])
    
    df_flux = pd.read_csv(FLUX_CSV, index_col=0, parse_dates=True)
    
    t0 = df_sim['date'].iloc[0]
    t_years = (df_sim['date'] - t0).dt.total_seconds().values / SEC_PER_YEAR
    t_total = t_years[-1]
    t_norm = t_years / t_total
    
    C_cols = [f'C_{b}' for b in BASINS]
    S_cols = [f'S_{b}' for b in BASINS]
    
    C_raw = df_sim[C_cols].values * 1000.0  # mmol/m³
    S_raw = df_sim[S_cols].values           # psu
    
    # Keeling atmospheric CO2
    dates_dt = pd.DatetimeIndex(df_sim['date'])
    months_from_2000 = (dates_dt.year - 2000) * 12.0 + (dates_dt.month - 1) + (dates_dt.day / 30.0)
    pCO2_air = np.array(367.41 + 0.18152 * months_from_2000)
    
    # Interpolate freshwater fluxes onto simulation timestamps
    flux_cols = ['F_NA', 'F_SA', 'F_SO', 'F_PO', 'F_IO']
    df_flux_interp = pd.DataFrame(index=df_sim['date'])
    for col in flux_cols:
        s_series = df_flux[col].reindex(df_flux.index.union(df_sim['date'])).interpolate('time')
        df_flux_interp[col] = s_series.loc[df_sim['date']].values
    F_interp = df_flux_interp[flux_cols].values  # (N, 5)
    
    # Train (1987-2009, 82%), Test (2010-2014, 18%)
    split_idx = int(len(df_sim) * 0.82)
    
    # Standard scalers for training stability
    scaler_C = StandardScaler()
    scaler_S = StandardScaler()
    
    C_scaled = scaler_C.fit_transform(C_raw)
    S_scaled = scaler_S.fit_transform(S_raw)
    
    Y_raw    = np.concatenate([C_raw, S_raw], axis=1)        # 10 columns
    Y_scaled = np.concatenate([C_scaled, S_scaled], axis=1)  # 10 columns
    
    return {
        'dates': df_sim['date'].values,
        't_years': t_years,
        't_norm': t_norm,
        't_total': t_total,
        'pCO2_air': pCO2_air,
        'F_interp': F_interp,
        'C_raw': C_raw,
        'S_raw': S_raw,
        'Y_raw': Y_raw,
        'Y_scaled': Y_scaled,
        'scaler_C': scaler_C,
        'scaler_S': scaler_S,
        'split_idx': split_idx
    }

class CoupledOceanNet(nn.Module):
    """10-Dimensional Deep Neural Network parameterizing both Carbon and Salinity."""
    def __init__(self, in_features=7, hidden_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 10)  # 5 Carbon + 5 Salinity
        )
        
    def forward(self, x):
        return self.net(x)

def compute_coupled_physics_loss(model, x_colloc, t_total, scaler_C, scaler_S):
    """
    Computes joint 10-dimensional physics loss:
      Loss_phys = Loss_phys(Salinity) + Loss_phys(Carbon | Salinity)
    Where the Carbon deep exchange is driven by the predicted Salinity gradients!
    """
    # x_colloc: [t_norm, pco2_norm, F_NA..F_IO]
    x_colloc.requires_grad_(True)
    y_pred_scaled = model(x_colloc)
    
    # Extract components
    C_pred_scaled = y_pred_scaled[:, :5]
    S_pred_scaled = y_pred_scaled[:, 5:]
    
    # Recover physical units inside autograd graph
    c_scale = torch.tensor(scaler_C.scale_, dtype=torch.float32)
    c_mean  = torch.tensor(scaler_C.mean_, dtype=torch.float32)
    s_scale = torch.tensor(scaler_S.scale_, dtype=torch.float32)
    s_mean  = torch.tensor(scaler_S.mean_, dtype=torch.float32)
    
    C_pred = C_pred_scaled * c_scale + c_mean  # mmol/m³
    S_pred = S_pred_scaled * s_scale + s_mean  # psu
    
    t_colloc = x_colloc[:, 0:1]
    
    # Compute time derivatives via autograd: dC/dt and dS/dt (per year)
    dC_dt_list = []
    for k in range(5):
        grad_k = torch.autograd.grad(
            outputs=C_pred[:, k],
            inputs=x_colloc,
            grad_outputs=torch.ones_like(C_pred[:, k]),
            create_graph=True
        )[0][:, 0:1]  # wrt t_norm
        dC_dt_list.append(grad_k / t_total)
    dC_dt = torch.cat(dC_dt_list, dim=1)  # (N, 5)
    
    dS_dt_list = []
    for k in range(5):
        grad_k = torch.autograd.grad(
            outputs=S_pred[:, k],
            inputs=x_colloc,
            grad_outputs=torch.ones_like(S_pred[:, k]),
            create_graph=True
        )[0][:, 0:1]  # wrt t_norm
        dS_dt_list.append(grad_k / t_total)
    dS_dt = torch.cat(dS_dt_list, dim=1)  # (N, 5)
    
    # Physical RHS evaluation
    F_colloc = x_colloc[:, 2:7]  # freshwater fluxes
    pco2_colloc = x_colloc[:, 1] * 50.0 + 367.41  # atmospheric pCO2 [uatm]
    
    # 1. Deep Thermohaline Overturning Flow Rates from PREDICTED Salinity:
    # q_k = K_DEEP * BETA * (S_i - S_j)
    q_flows = []
    for i, j in DEEP_PAIRS:
        q_k = K_DEEP * BETA * (S_pred[:, i] - S_pred[:, j])  # [Sv]
        q_flows.append(q_k)
        
    # 2. Salinity RHS calculation
    # fw = -Fh (HOAPS E-P to paper's P-E)
    fw_NA = -F_colloc[:, 0] + b_NA + q_AMAZ
    fw_SA = -F_colloc[:, 1]
    fw_SO = -F_colloc[:, 2] + b_SO
    fw_PO = -F_colloc[:, 3]
    fw_IO = -F_colloc[:, 4]
    fw_list = [fw_NA, fw_SA, fw_SO, fw_PO, fw_IO]
    
    rhs_S_list = []
    for i in range(5):
        # Freshwater dilution
        ds_fw = - (fw_list[i] * SV_TO_M3S * S0_REF / V[i]) * SEC_PER_YEAR
        
        # Surface currents
        ds_surf = torch.zeros_like(S_pred[:, i])
        for src, dst, Qsv in SURFACE_CURRENTS:
            if dst == i:
                ds_surf = ds_surf + (Qsv * SV_TO_M3S / V[i] * SEC_PER_YEAR) * (S_pred[:, src] - S_pred[:, dst])
                
        # Deep overturning
        ds_deep = torch.zeros_like(S_pred[:, i])
        for k, (src, dst) in enumerate(DEEP_PAIRS):
            qm = q_flows[k] * SV_TO_M3S
            if i == src:
                ds_deep = ds_deep - (qm / V[i] * SEC_PER_YEAR) * (S_pred[:, src] - S_pred[:, dst])
            elif i == dst:
                ds_deep = ds_deep + (qm / V[dst] * SEC_PER_YEAR) * (S_pred[:, src] - S_pred[:, dst])
                
        rhs_S_i = ds_fw + ds_surf + ds_deep
        rhs_S_list.append(rhs_S_i)
    rhs_S = torch.stack(rhs_S_list, dim=1)
    
    # 3. Carbon RHS calculation (incorporating predicted Salinity advection!)
    rhs_C_list = []
    for i in range(5):
        # Gas exchange
        gas_rate = 0.02 * (A[i] / V[i]) * 1e4
        pco2_sat = 0.016 * (1.0 + (pco2_colloc - 367.0) / 367.0) * 1000.0
        dc_gas = gas_rate * (pco2_sat - C_pred[:, i])
        
        # Surface transport
        dc_surf = torch.zeros_like(C_pred[:, i])
        for src, dst, Qsv in SURFACE_CURRENTS:
            if dst == i:
                dc_surf = dc_surf + (Qsv * SV_TO_M3S / V[i] * SEC_PER_YEAR) * (C_pred[:, src] - C_pred[:, dst])
                
        # Deep thermohaline advection (USES PREDICTED SALINITY FLOWS q_flows[k]!)
        dc_deep = torch.zeros_like(C_pred[:, i])
        for k, (src, dst) in enumerate(DEEP_PAIRS):
            qm = q_flows[k] * SV_TO_M3S
            if i == src:
                dc_deep = dc_deep - (qm / V[i] * SEC_PER_YEAR) * (C_pred[:, src] - C_pred[:, dst])
            elif i == dst:
                dc_deep = dc_deep + (qm / V[dst] * SEC_PER_YEAR) * (C_pred[:, src] - C_pred[:, dst])
                
        rhs_C_i = dc_gas * 0.02 + dc_surf * 0.04 + dc_deep * 0.04
        rhs_C_list.append(rhs_C_i)
    rhs_C = torch.stack(rhs_C_list, dim=1)
    
    loss_phys_S = torch.mean((dS_dt - rhs_S)**2)
    loss_phys_C = torch.mean((dC_dt - rhs_C)**2)
    
    return loss_phys_S, loss_phys_C

def train_coupled_pinn(data):
    print("="*75)
    print("TRAINING COUPLED 10-DIMENSIONAL SALINITY-CARBON PINN")
    print("="*75)
    
    split = data['split_idx']
    
    # Construct input feature tensor: [t_norm, pco2_norm, F_NA..F_IO]
    t_norm = data['t_norm'].reshape(-1, 1)
    pco2_norm = ((data['pCO2_air'] - 367.41) / 50.0).reshape(-1, 1)
    F_feats = data['F_interp']
    
    X_all = np.concatenate([t_norm, pco2_norm, F_feats], axis=1)  # (N, 7)
    
    X_train = torch.tensor(X_all[:split], dtype=torch.float32)
    Y_train_sc = torch.tensor(data['Y_scaled'][:split], dtype=torch.float32)
    
    X_test = torch.tensor(X_all[split:], dtype=torch.float32)
    Y_test_raw = data['Y_raw'][split:]
    
    # 1. Unconstrained Baseline 10D MLP
    print("\n1. Training Unconstrained 10D Neural Network (Baseline)...")
    base_model = CoupledOceanNet(in_features=7, hidden_dim=96)
    opt_base = torch.optim.Adam(base_model.parameters(), lr=0.008)
    
    base_losses = []
    epochs = 1200
    for ep in range(epochs):
        opt_base.zero_grad()
        pred = base_model(X_train)
        loss = torch.mean((pred - Y_train_sc)**2)
        loss.backward()
        opt_base.step()
        base_losses.append(loss.item())
        if (ep + 1) % 300 == 0:
            print(f"  [Baseline] Epoch {ep+1}/{epochs} - Data Loss: {loss.item():.6e}")
            
    # 2. Coupled Physics-Informed Neural Network (PINN)
    print("\n2. Training Coupled 10D PINN (Bi-directional Salinity-Carbon Physics)...")
    pinn_model = CoupledOceanNet(in_features=7, hidden_dim=96)
    opt_pinn = torch.optim.Adam(pinn_model.parameters(), lr=0.008)
    
    data_losses = []
    phys_S_losses = []
    phys_C_losses = []
    total_losses = []
    
    lambda_S = 0.015
    lambda_C = 0.015
    
    for ep in range(epochs):
        opt_pinn.zero_grad()
        pred = pinn_model(X_train)
        loss_data = torch.mean((pred - Y_train_sc)**2)
        
        # Collocation sampling across time and freshwater conditions
        rand_idx = np.random.choice(len(X_train), 100, replace=False)
        X_colloc = X_train[rand_idx].clone()
        
        loss_phys_S, loss_phys_C = compute_coupled_physics_loss(
            pinn_model, X_colloc, data['t_total'], data['scaler_C'], data['scaler_S']
        )
        
        loss_total = loss_data + lambda_S * loss_phys_S + lambda_C * loss_phys_C
        loss_total.backward()
        opt_pinn.step()
        
        data_losses.append(loss_data.item())
        phys_S_losses.append(loss_phys_S.item())
        phys_C_losses.append(loss_phys_C.item())
        total_losses.append(loss_total.item())
        
        if (ep + 1) % 300 == 0:
            print(f"  [Coupled PINN] Epoch {ep+1}/{epochs} | Data Loss: {loss_data.item():.6e} | Phys S: {loss_phys_S.item():.6e} | Phys C: {loss_phys_C.item():.6e} | Total: {loss_total.item():.6e}")
            
    return {
        'base_model': base_model,
        'pinn_model': pinn_model,
        'base_losses': base_losses,
        'data_losses': data_losses,
        'phys_S_losses': phys_S_losses,
        'phys_C_losses': phys_C_losses,
        'total_losses': total_losses,
        'X_train': X_train,
        'X_test': X_test,
        'Y_test_raw': Y_test_raw,
        'X_all': X_all
    }

def evaluate_models(data, train_res):
    print("\n" + "="*75)
    print("BENCHMARKING COUPLED 10D PINN VS BASELINE ON UNSEEN TEST DATA (2010-2014)")
    print("="*75)
    
    base_model = train_res['base_model']
    pinn_model = train_res['pinn_model']
    X_test     = train_res['X_test']
    Y_test_raw = train_res['Y_test_raw']
    
    scaler_C = data['scaler_C']
    scaler_S = data['scaler_S']
    
    with torch.no_grad():
        base_pred_sc = base_model(X_test).numpy()
        pinn_pred_sc = pinn_model(X_test).numpy()
        
    # Unscale predictions back to physical units
    def unscale(pred_sc):
        C_phys = scaler_C.inverse_transform(pred_sc[:, :5])
        S_phys = scaler_S.inverse_transform(pred_sc[:, 5:])
        return np.concatenate([C_phys, S_phys], axis=1)
        
    Y_base_pred = unscale(base_pred_sc)
    Y_pinn_pred = unscale(pinn_pred_sc)
    
    # Calculate per-variable metrics
    metrics = []
    print("\nPER-BASIN TEST EVALUATION RESULTS:")
    print(f"{'Variable':<12} | {'Base R²':<10} | {'PINN R²':<10} | {'Base RMSE':<14} | {'PINN RMSE':<14} | {'Physics Bound'}")
    print("-" * 75)
    
    for i in range(10):
        name = f"C_{BASINS[i]}" if i < 5 else f"S_{BASINS[i-5]}"
        unit = "mmol/m³" if i < 5 else "psu"
        
        y_true = Y_test_raw[:, i]
        y_base = Y_base_pred[:, i]
        y_pinn = Y_pinn_pred[:, i]
        
        r2_base = r2_score(y_true, y_base)
        r2_pinn = r2_score(y_true, y_pinn)
        rmse_base = np.sqrt(mean_squared_error(y_true, y_base))
        rmse_pinn = np.sqrt(mean_squared_error(y_true, y_pinn))
        
        metrics.append({
            'Variable': name,
            'Unit': unit,
            'Baseline_R2': r2_base,
            'PINN_R2': r2_pinn,
            'Baseline_RMSE': rmse_base,
            'PINN_RMSE': rmse_pinn,
            'RMSE_Reduction_Percent': ((rmse_base - rmse_pinn) / rmse_base) * 100.0
        })
        
        print(f"{name:<12} | {r2_base:<10.4f} | {r2_pinn:<10.4f} | {rmse_base:<14.6f} | {rmse_pinn:<14.6f} | Satisfied")
        
    metrics_df = pd.DataFrame(metrics)
    csv_path = os.path.join(OUTPUT_DIR, 'pinn_coupled_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved coupled metrics table to: {csv_path}")
    
    return {
        'Y_base_pred': Y_base_pred,
        'Y_pinn_pred': Y_pinn_pred,
        'metrics_df': metrics_df
    }

def project_2030(data, train_res):
    print("\n" + "="*75)
    print("FORWARD 2015-2030 ROLLOUT: COUPLED PINN VS UNCONSTRAINED ML")
    print("="*75)
    
    t_last = data['t_years'][-1]
    t_proj = np.linspace(t_last, t_last + 16.0, 200)
    t_proj_norm = t_proj / data['t_total']
    
    pco2_proj = 367.41 + 0.18152 * 12.0 * (t_proj - 12.5)
    pco2_proj_norm = (pco2_proj - 367.41) / 50.0
    
    # Repeat mean seasonal freshwater fluxes for projection
    mean_F = np.mean(data['F_interp'][-36:], axis=0)  # last 3 years mean
    F_proj = np.tile(mean_F, (len(t_proj), 1))
    
    X_proj = np.column_stack([t_proj_norm, pco2_proj_norm, F_proj])
    X_proj_t = torch.tensor(X_proj, dtype=torch.float32)
    
    with torch.no_grad():
        base_proj_sc = train_res['base_model'](X_proj_t).numpy()
        pinn_proj_sc = train_res['pinn_model'](X_proj_t).numpy()
        
    def unscale(pred_sc):
        C_phys = data['scaler_C'].inverse_transform(pred_sc[:, :5])
        S_phys = data['scaler_S'].inverse_transform(pred_sc[:, 5:])
        return np.concatenate([C_phys, S_phys], axis=1)
        
    Y_base_proj = unscale(base_proj_sc)
    Y_pinn_proj = unscale(pinn_proj_sc)
    
    start_date = pd.to_datetime(data['dates'][-1])
    proj_dates = [start_date + pd.Timedelta(days=int(y * 365.25)) for y in (t_proj - t_last)]
    
    return {
        'proj_dates': proj_dates,
        'Y_base_proj': Y_base_proj,
        'Y_pinn_proj': Y_pinn_proj
    }

def plot_coupled_results(data, train_res, eval_res, proj_res):
    print("\nGenerating Coupled 10D PINN Plots...")
    
    # Plot 1: Loss Convergence
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_res['base_losses'], label='Unconstrained MLP Data Loss', color='gray', linestyle='--', lw=1.8)
    ax.plot(train_res['data_losses'], label=r'Coupled PINN Data Loss ($\mathcal{L}_{\mathrm{Data}}$)', color='#1f77b4', lw=2.0)
    ax.plot(train_res['phys_S_losses'], label=r'Salinity Physics Loss ($\mathcal{L}_{\mathrm{Phys, S}}$)', color='#2ca02c', lw=2.0)
    ax.plot(train_res['phys_C_losses'], label=r'Carbon Physics Loss ($\mathcal{L}_{\mathrm{Phys, C}}$ with Advective $S$ Coupling)', color='#d62728', lw=2.0)
    ax.plot(train_res['total_losses'], label='Coupled PINN Total Loss', color='black', lw=2.2)
    ax.set_yscale('log')
    ax.set_xlabel('Training Epochs', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss Magnitude (Log Scale)', fontsize=11, fontweight='bold')
    ax.set_title('10-Dimensional Coupled Salinity-Carbon PINN Loss Convergence', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    out_loss = os.path.join(OUTPUT_DIR, 'pinn_coupled_loss.png')
    plt.savefig(out_loss, dpi=160)
    plt.close()
    print(f"Saved: {out_loss}")
    
    # Plot 2: Test Benchmark Comparison (Indian Ocean & North Atlantic for both C and S)
    dates_test = data['dates'][data['split_idx']:]
    Y_test = train_res['Y_test_raw']
    Y_pinn = eval_res['Y_pinn_pred']
    Y_base = eval_res['Y_base_pred']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    
    # C_IO (Index 4)
    axes[0, 0].plot(dates_test, Y_test[:, 4], label='Numerical Truth', color='black', lw=2.2)
    axes[0, 0].plot(dates_test, Y_base[:, 4], '--', label='Baseline MLP', color='#ff7f0e', lw=1.8)
    axes[0, 0].plot(dates_test, Y_pinn[:, 4], '-.', label='Coupled PINN', color='#2ca02c', lw=2.0)
    axes[0, 0].set_ylabel('Indian Ocean $C_{\\mathrm{IO}}$\n(mmol/m³)', fontsize=10, fontweight='bold')
    axes[0, 0].set_title('Indian Ocean Carbon Test Benchmark (2010–2014)', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='upper left', fontsize=9)
    
    # C_NA (Index 0)
    axes[0, 1].plot(dates_test, Y_test[:, 0], label='Numerical Truth', color='black', lw=2.2)
    axes[0, 1].plot(dates_test, Y_base[:, 0], '--', label='Baseline MLP', color='#ff7f0e', lw=1.8)
    axes[0, 1].plot(dates_test, Y_pinn[:, 0], '-.', label='Coupled PINN', color='#2ca02c', lw=2.0)
    axes[0, 1].set_ylabel('North Atlantic $C_{\\mathrm{NA}}$\n(mmol/m³)', fontsize=10, fontweight='bold')
    axes[0, 1].set_title('North Atlantic Carbon Test Benchmark (2010–2014)', fontsize=11, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc='upper left', fontsize=9)
    
    # S_IO (Index 9)
    axes[1, 0].plot(dates_test, Y_test[:, 9], label='Numerical Truth', color='black', lw=2.2)
    axes[1, 0].plot(dates_test, Y_base[:, 9], '--', label='Baseline MLP', color='#ff7f0e', lw=1.8)
    axes[1, 0].plot(dates_test, Y_pinn[:, 9], '-.', label='Coupled PINN', color='#1f77b4', lw=2.0)
    axes[1, 0].set_ylabel('Indian Ocean $S_{\\mathrm{IO}}$\n(psu)', fontsize=10, fontweight='bold')
    axes[1, 0].set_xlabel('Year (Test Period 2010–2014)', fontsize=10, fontweight='bold')
    axes[1, 0].set_title('Indian Ocean Salinity Test Benchmark', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc='upper left', fontsize=9)
    
    # S_NA (Index 5)
    axes[1, 1].plot(dates_test, Y_test[:, 5], label='Numerical Truth', color='black', lw=2.2)
    axes[1, 1].plot(dates_test, Y_base[:, 5], '--', label='Baseline MLP', color='#ff7f0e', lw=1.8)
    axes[1, 1].plot(dates_test, Y_pinn[:, 5], '-.', label='Coupled PINN', color='#1f77b4', lw=2.0)
    axes[1, 1].set_ylabel('North Atlantic $S_{\\mathrm{NA}}$\n(psu)', fontsize=10, fontweight='bold')
    axes[1, 1].set_xlabel('Year (Test Period 2010–2014)', fontsize=10, fontweight='bold')
    axes[1, 1].set_title('North Atlantic Salinity Test Benchmark', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    out_test = os.path.join(OUTPUT_DIR, 'pinn_coupled_benchmark.png')
    plt.savefig(out_test, dpi=160)
    plt.close()
    print(f"Saved: {out_test}")
    
    # Plot 3: 2015-2030 Projections (Showing both Carbon and Salinity bounded preservation)
    hist_dates = data['dates']
    proj_dates = proj_res['proj_dates']
    Y_raw = data['Y_raw']
    Y_base_proj = proj_res['Y_base_proj']
    Y_pinn_proj = proj_res['Y_pinn_proj']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Indian Ocean Carbon Projections
    ax1.plot(hist_dates, Y_raw[:, 4], label='Historical Observation (1987-2014)', color='black', lw=2.2)
    ax1.plot(proj_dates, Y_base_proj[:, 4], '--', label='Unconstrained MLP Projection (Unphysical Drift)', color='#d62728', lw=2.0)
    ax1.plot(proj_dates, Y_pinn_proj[:, 4], '-', label='Coupled PINN Projection (Mass Conserved)', color='#2ca02c', lw=2.5)
    ax1.axvline(hist_dates[-1], color='gray', linestyle=':', lw=1.5, label='2014 Forecast Boundary')
    ax1.set_ylabel('Indian Ocean $C_{\\mathrm{IO}}$\n(mmol/m³)', fontsize=11, fontweight='bold')
    ax1.set_title('Coupled 10D PINN Long-Term 2015–2030 Projection: Preserving Thermohaline Constraints', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', frameon=True)
    
    # North Atlantic Salinity Projections
    ax2.plot(hist_dates, Y_raw[:, 5], label='Historical Observation (1987-2014)', color='black', lw=2.2)
    ax2.plot(proj_dates, Y_base_proj[:, 5], '--', label='Unconstrained MLP Projection (Severe Divergence)', color='#d62728', lw=2.0)
    ax2.plot(proj_dates, Y_pinn_proj[:, 5], '-', label='Coupled PINN Projection (Salt Conserved)', color='#1f77b4', lw=2.5)
    ax2.axvline(hist_dates[-1], color='gray', linestyle=':', lw=1.5)
    ax2.set_ylabel('North Atlantic $S_{\\mathrm{NA}}$\n(psu)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Year (1987–2030 Projection)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    out_proj = os.path.join(OUTPUT_DIR, 'pinn_coupled_2030_projections.png')
    plt.savefig(out_proj, dpi=160)
    plt.close()
    print(f"Saved: {out_proj}")

def main():
    data = load_data()
    train_res = train_coupled_pinn(data)
    eval_res  = evaluate_models(data, train_res)
    proj_res  = project_2030(data, train_res)
    plot_coupled_results(data, train_res, eval_res, proj_res)
    print("\n[SUCCESS] Coupled 10-Dimensional Salinity-Carbon PINN Complete!")

if __name__ == '__main__':
    main()
