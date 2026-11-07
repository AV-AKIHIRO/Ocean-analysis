#!/usr/bin/env python3
"""
scripts/climate_scenarios_ipcc.py
=================================
IPCC AR6 Shared Socioeconomic Pathways (SSP1-2.6, SSP2-4.5, SSP3-7.0, SSP5-8.5)
Coupled Ocean Carbon Sink & Thermohaline Simulation (2015–2060).

Simulates:
  1. Atmospheric pCO2 trajectories under IPCC AR6 SSPs
  2. Sea Surface Temperature (SST) warming impact on Weiss K0 solubility
  3. Accelerating Arctic meltwater forcing (b_NA) on North Atlantic salinity
  4. Cumulative global and basin-scale carbon uptake (in Petagrams of Carbon, PgC)
  5. Climate feedback: Ocean carbon sponge saturation & AMOC weakening
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKSPACE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(WORKSPACE, 'output')
SIM_CSV    = os.path.join(OUTPUT_DIR, 'co2_salinity_simulation_1987_2014.csv')

BASINS = ['NA', 'SA', 'SO', 'PO', 'IO']
SEC_PER_YEAR = 365.25 * 86400.0
SV_TO_M3S    = 1.0e6
MOL_C_TO_PGC = 12.011 * 1.0e-15  # 1 mol C = 12.011 g = 12.011e-15 PgC

# Basin geometry (Sunny et al. 2023 Table II)
V = np.array([146.0e15, 160.0e15, 71.8e15, 710.0e15, 264.0e15])   # m³
A = np.array([ 41.49e12, 40.27e12, 20.33e12, 165.25e12, 70.56e12]) # m²

SURFACE_CURRENTS = [(1,0,26.0), (2,1,30.0), (3,4,15.0), (4,1,10.0), (4,2,60.0)]
DEEP_PAIRS       = [(0,1), (1,2), (1,3), (2,3), (2,4)]

S0_REF = 35.0; BETA = 7.5e-4; K_DEEP = 60.0
K_EQ   = 0.0011; ALPHA = 1.0 + K_EQ
q_AMAZ = 0.24  # Amazon [Sv]

# Weiss (1974) K0 solubility constants
A1_K0, A2_K0, A3_K0 = -58.0931, 90.5069, 22.294
B1_K0, B2_K0, B3_K0 =  0.027766, -0.025888, 0.0050578
U10 = 6.0  # wind speed [m/s]

# Pressure-correction constants
a_c=3.63579e-5; b_c=4.47782e-5; g_c=1.8833e-5
l_c=7.31010e-5; h_c=5.41469e-5; psi_c=7.49356e-5
M_SAL = 18.015 + 0.015 * 35.0
RHO0  = 1029.0

# Base SST intercepts at year 2000 [K]
SST_BASE = np.array([291.9, 290.3, 272.29, 292.88, 291.57])

# Base Arctic and Antarctic freshwater terms
b_NA_base =  3.28e3 / SV_TO_M3S   # 0.00328 Sv
b_SO_base = -6.944e2 / SV_TO_M3S  # -0.000694 Sv

# -------------------------------------------------------------
# IPCC AR6 Scenario Definitions (2015–2060)
# -------------------------------------------------------------
SCENARIOS = {
    'SSP1-2.6': {
        'name': 'SSP1-2.6 (Paris 1.5°C–2.0°C Target)',
        'color': '#2ca02c',
        # pCO2 peaks at ~440 uatm around 2045, then stabilizes
        'pco2_func': lambda t_yr: 400.0 + 40.0 * (1.0 - np.exp(- (t_yr - 2015) / 12.0)),
        'sst_warming_2060': 0.8,    # +0.8 K global warming
        'arctic_melt_mult': 1.15,   # +15% melt
    },
    'SSP2-4.5': {
        'name': 'SSP2-4.5 (Middle-of-the-Road)',
        'color': '#1f77b4',
        # Linear/mild quadratic climb to ~540 uatm by 2060
        'pco2_func': lambda t_yr: 400.0 + 3.11 * (t_yr - 2015),
        'sst_warming_2060': 1.8,    # +1.8 K global warming
        'arctic_melt_mult': 1.45,   # +45% melt
    },
    'SSP3-7.0': {
        'name': 'SSP3-7.0 (High Emissions / Regional Rivalry)',
        'color': '#ff7f0e',
        # Accelerating to ~630 uatm by 2060
        'pco2_func': lambda t_yr: 400.0 + 3.5 * (t_yr - 2015) + 0.035 * (t_yr - 2015)**2,
        'sst_warming_2060': 2.6,    # +2.6 K global warming
        'arctic_melt_mult': 1.75,   # +75% melt
    },
    'SSP5-8.5': {
        'name': 'SSP5-8.5 (Fossil-Fueled Development / Worst Case)',
        'color': '#d62728',
        # Intense exponential acceleration to ~750 uatm by 2060
        'pco2_func': lambda t_yr: 400.0 + 4.2 * (t_yr - 2015) + 0.08 * (t_yr - 2015)**2,
        'sst_warming_2060': 3.6,    # +3.6 K global warming
        'arctic_melt_mult': 2.20,   # +120% Greenland melt surge
    }
}

def k0_weiss(T_K, S):
    T100 = T_K / 100.0
    lnK0 = (A1_K0 + A2_K0/T100 + A3_K0*np.log(T100)
             + S*(B1_K0 + B2_K0*T100 + B3_K0*T100**2))
    return np.exp(lnK0) * 1e3  # mol/(m³·atm)

def schmidt_no(T_K):
    T = T_K - 273.15
    return 2116.8 - 136.5*T + 4.7353*T**2 - 0.092307*T**3 + 0.0007555*T**4

def gas_piston_velocity(T_K):
    g660 = 0.24 * U10**2 / 100.0 / SEC_PER_YEAR  # m/s
    return g660 * (schmidt_no(T_K)/660.0)**(-0.5)

def C_deep_corr(C, T_K):
    M_rho = M_SAL / (RHO0/1000.0)
    T = T_K - 273.15
    corr = ((1+a_c-b_c*T) + (g_c*T+l_c)*M_rho*C + (h_c*T-psi_c)*M_rho**2*C**2)
    return C * corr

def get_initial_2015_state():
    df_sim = pd.read_csv(SIM_CSV)
    # Extract final state at Dec 2014 / Jan 2015
    C_2015 = df_sim[[f'C_{b}' for b in BASINS]].iloc[-1].values  # mol/m³
    S_2015 = df_sim[[f'S_{b}' for b in BASINS]].iloc[-1].values  # psu
    T_2015 = ALPHA * C_2015
    Y_2015 = np.concatenate([T_2015, S_2015])
    return Y_2015, C_2015, S_2015

def run_scenario_simulation(scen_key, scen_cfg, Y_init):
    print(f"\n--> Running simulation: {scen_cfg['name']} (2015-2060)")
    
    t_span = (2015.0, 2060.0)
    t_eval = np.linspace(2015.0, 2060.0, 451)  # 0.1 year resolution
    
    pco2_f = scen_cfg['pco2_func']
    warming_rate = scen_cfg['sst_warming_2060'] / 45.0  # K / year
    melt_rate = (scen_cfg['arctic_melt_mult'] - 1.0) / 45.0
    
    # Baseline freshwater seasonal flux (Sv)
    F_mean = np.array([-0.05, 0.08, -0.12, 0.25, 0.10])
    
    def scenario_ode(t, y):
        T = y[0:5]
        S = y[5:10]
        C = T / ALPHA
        
        # Current atmospheric pCO2 [atm]
        pa = pco2_f(t) * 1e-6
        
        # Warming SST [K]
        delta_t = t - 2015.0
        Tk = SST_BASE + warming_rate * delta_t
        
        K0v = k0_weiss(Tk, S)
        gv  = gas_piston_velocity(Tk)
        C_pr = C_deep_corr(C, Tk)
        
        # Deep circulation flow rates [Sv]
        q = np.array([K_DEEP * BETA * (S[i] - S[j]) for i, j in DEEP_PAIRS])
        
        # 1. Total Carbon (T) Derivatives
        dT = np.zeros(5)
        # Air-Sea Gas Exchange
        for i in range(5):
            dT[i] += ALPHA * gv[i] * A[i] / V[i] * (K0v[i]*pa - C[i]) * SEC_PER_YEAR
            
        # Amazon Inflow
        dT[0] += ALPHA * q_AMAZ * SV_TO_M3S * C[0] / V[0] * SEC_PER_YEAR
        
        # Surface transport
        for src, dst, Qsv in SURFACE_CURRENTS:
            Q = Qsv * SV_TO_M3S
            dT[dst] += Q * (T[src] - T[dst]) / V[dst] * SEC_PER_YEAR
            
        # Deep transport
        Tp = ALPHA * C_pr
        for k, (i, j) in enumerate(DEEP_PAIRS):
            qm = q[k] * SV_TO_M3S
            dT[i] -= qm * (Tp[i] - Tp[j]) / V[i] * SEC_PER_YEAR
            dT[j] += qm * (Tp[i] - Tp[j]) / V[j] * SEC_PER_YEAR
            
        # 2. Salinity (S) Derivatives
        dS = np.zeros(5)
        # Freshwater forcing with climate melt anomaly
        b_NA_t = b_NA_base * (1.0 + melt_rate * delta_t)
        fw = - F_mean.copy()
        fw[0] += b_NA_t + q_AMAZ
        fw[2] += b_SO_base
        
        for i in range(5):
            dS[i] -= fw[i] * SV_TO_M3S * S0_REF / V[i] * SEC_PER_YEAR
            
        for src, dst, Qsv in SURFACE_CURRENTS:
            Q = Qsv * SV_TO_M3S
            dS[dst] += Q * (S[src] - S[dst]) / V[dst] * SEC_PER_YEAR
            
        for k, (i, j) in enumerate(DEEP_PAIRS):
            qm = q[k] * SV_TO_M3S
            dS[i] -= qm * (S[i] - S[j]) / V[i] * SEC_PER_YEAR
            dS[j] += qm * (S[i] - S[j]) / V[j] * SEC_PER_YEAR
            
        return np.concatenate([dT, dS])
        
    sol = solve_ivp(scenario_ode, t_span, Y_init, t_eval=t_eval, method='RK45', rtol=1e-7, atol=1e-9)
    
    T_sol = sol.y[0:5]
    S_sol = sol.y[5:10]
    C_sol = T_sol / ALPHA
    
    # Calculate cumulative carbon uptake in Petagrams of Carbon (PgC)
    # Uptake_i(t) = (C_i(t) - C_i(2015)) * V_i * 12.011e-15 [PgC]
    delta_C = C_sol - C_sol[:, 0:1]
    uptake_basin_pgc = np.zeros_like(delta_C)
    for i in range(5):
        uptake_basin_pgc[i] = delta_C[i] * V[i] * MOL_C_TO_PGC
    total_uptake_pgc = np.sum(uptake_basin_pgc, axis=0)
    
    # AMOC Overturning Flow (NA to SA deep pair: index 0)
    amoc_sv = K_DEEP * BETA * (S_sol[0] - S_sol[1])
    
    pco2_vals = [pco2_f(t) for t in t_eval]
    
    return {
        't': t_eval,
        'C': C_sol,
        'S': S_sol,
        'pco2': np.array(pco2_vals),
        'uptake_basin': uptake_basin_pgc,
        'total_uptake': total_uptake_pgc,
        'amoc': amoc_sv
    }

def plot_climate_scenarios(results):
    print("\nGenerating IPCC Climate Scenario Visualizations...")
    
    # Plot 1: Atmospheric Forcing Pathways
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, res in results.items():
        cfg = SCENARIOS[k]
        ax.plot(res['t'], res['pco2'], label=cfg['name'], color=cfg['color'], lw=2.4)
    ax.set_ylabel('Atmospheric $p\\mathrm{CO}_2$ (µatm / ppm)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Year (2015–2060)', fontsize=11, fontweight='bold')
    ax.set_title('IPCC AR6 Atmospheric Carbon Forcing Pathways (2015–2060)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    out1 = os.path.join(OUTPUT_DIR, 'climate_scenarios_pco2_pathways.png')
    plt.savefig(out1, dpi=160)
    plt.close()
    print(f"Saved: {out1}")
    
    # Plot 2: Total Cumulative Ocean Carbon Uptake (PgC)
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, res in results.items():
        cfg = SCENARIOS[k]
        final_pgc = res['total_uptake'][-1]
        ax.plot(res['t'], res['total_uptake'], label=f"{cfg['name']} (Final: +{final_pgc:.1f} PgC)", color=cfg['color'], lw=2.6)
    ax.set_ylabel('Cumulative Global Ocean Carbon Sink (PgC)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Year (2015–2060)', fontsize=11, fontweight='bold')
    ax.set_title('Cumulative Oceanic Carbon Sequestration Under IPCC Climate Scenarios', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, 'climate_scenarios_cumulative_sink.png')
    plt.savefig(out2, dpi=160)
    plt.close()
    print(f"Saved: {out2}")
    
    # Plot 3: Basin-Scale Carbon Uptake Breakdown (2x2: NA, SO, PO, IO)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_basins = [(0, 'North Atlantic', axes[0, 0]),
                   (2, 'Southern Ocean', axes[0, 1]),
                   (3, 'Pacific Ocean', axes[1, 0]),
                   (4, 'Indian Ocean', axes[1, 1])]
                   
    for b_idx, b_name, ax in plot_basins:
        for k, res in results.items():
            cfg = SCENARIOS[k]
            ax.plot(res['t'], res['uptake_basin'][b_idx], label=k, color=cfg['color'], lw=2.2)
        ax.set_ylabel(f'{b_name} Sink (PgC)', fontsize=10, fontweight='bold')
        ax.set_title(f'{b_name} Carbon Accumulation', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
    axes[1, 0].set_xlabel('Year (2015–2060)', fontsize=10, fontweight='bold')
    axes[1, 1].set_xlabel('Year (2015–2060)', fontsize=10, fontweight='bold')
    plt.suptitle('Basin-by-Basin Carbon Sequestration Under IPCC Shared Socioeconomic Pathways', fontsize=13, fontweight='bold', y=0.99)
    plt.tight_layout()
    out3 = os.path.join(OUTPUT_DIR, 'climate_scenarios_carbon_uptake.png')
    plt.savefig(out3, dpi=160)
    plt.close()
    print(f"Saved: {out3}")
    
    # Plot 4: AMOC Thermohaline Circulation Weakening
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, res in results.items():
        cfg = SCENARIOS[k]
        ax.plot(res['t'], res['amoc'], label=cfg['name'], color=cfg['color'], lw=2.4)
    ax.set_ylabel('North Atlantic Overturning $q_{\\mathrm{AMOC}}$ (Sv)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Year (2015–2060)', fontsize=11, fontweight='bold')
    ax.set_title('AMOC Overturning Strength Decline Under Accelerating Arctic Freshwater Melt', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', frameon=True)
    plt.tight_layout()
    out4 = os.path.join(OUTPUT_DIR, 'climate_scenarios_amoc_weakening.png')
    plt.savefig(out4, dpi=160)
    plt.close()
    print(f"Saved: {out4}")

def save_summary_table(results):
    rows = []
    for k, res in results.items():
        cfg = SCENARIOS[k]
        rows.append({
            'Scenario': k,
            'Description': cfg['name'],
            'pCO2_2060_ppm': res['pco2'][-1],
            'Global_Cumulative_Sink_2060_PgC': res['total_uptake'][-1],
            'North_Atlantic_Sink_PgC': res['uptake_basin'][0][-1],
            'Southern_Ocean_Sink_PgC': res['uptake_basin'][2][-1],
            'Pacific_Ocean_Sink_PgC': res['uptake_basin'][3][-1],
            'Indian_Ocean_Sink_PgC': res['uptake_basin'][4][-1],
            'AMOC_2015_Sv': res['amoc'][0],
            'AMOC_2060_Sv': res['amoc'][-1],
            'AMOC_Decline_Percent': ((res['amoc'][0] - res['amoc'][-1]) / res['amoc'][0]) * 100.0
        })
    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, 'climate_scenarios_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved climate scenario summary table: {csv_path}")
    print("\n" + "="*80)
    print("CLIMATE SCENARIO SIMULATION SUMMARY (2015-2060)")
    print("="*80)
    print(df_summary[['Scenario', 'pCO2_2060_ppm', 'Global_Cumulative_Sink_2060_PgC', 'AMOC_Decline_Percent']].to_string(index=False))

def main():
    Y_init, C_init, S_init = get_initial_2015_state()
    results = {}
    for scen_key, scen_cfg in SCENARIOS.items():
        results[scen_key] = run_scenario_simulation(scen_key, scen_cfg, Y_init)
    plot_climate_scenarios(results)
    save_summary_table(results)
    print("\n[SUCCESS] IPCC Climate Scenarios Simulation Complete!")

if __name__ == '__main__':
    main()
