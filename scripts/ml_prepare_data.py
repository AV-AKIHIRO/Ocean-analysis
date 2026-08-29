#!/usr/bin/env python3
"""
ml_prepare_data.py  (v2)
========================
Builds feature/target matrices for ML training.

Features (19 per timestep after adding disequilibrium):
  F_NA..F_IO          — HOAPS freshwater flux [Sv]
  SST_NA..SST_IO      — SST from Appendix D fits [K]
  pCO2_air            — Keeling curve [µatm]
  month_sin, month_cos
  year_norm
  DISEQ_NA..DISEQ_IO  — Air-sea CO₂ disequilibrium: K₀(T,S)*pCO₂ − C_i(t)
                        (the dominant driver of ΔC, computed from ODE C values)
"""

import os, pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

WORKSPACE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(WORKSPACE, 'output')
BASINS = ['NA', 'SA', 'SO', 'PO', 'IO']

# SST fits (Appendix D), t = months since Jan 2000
T_REF = 2000.0 + 15.0/365.25
SST_SLOPES     = np.array([0.0017, -0.00015, -0.00056, 0.00155, 0.00026])
SST_INTERCEPTS = np.array([291.9,   290.3,    272.29,   292.88,  291.57])
PCO2_SLOPE, PCO2_OFFSET = 0.18152, 367.41

# Weiss (1974) K₀ constants [mol/(L·atm)]
A1 = np.array([-58.0931, -58.0931, -58.0931, -58.0931, -58.0931])
A2 = np.array([90.5069,  90.5069,  90.5069,  90.5069,  90.5069])
A3 = np.array([22.2940,  22.2940,  22.2940,  22.2940,  22.2940])
B1 = np.array([0.027766, 0.027766, 0.027766, 0.027766, 0.027766])
B2 = np.array([-0.025888]*5)
B3 = np.array([0.0050578]*5)
S_REF = np.array([34.912, 35.435, 34.427, 34.668, 34.538])  # mean salinity per basin

def k0(T_K, S=None):
    """Weiss K₀ [mol/(L·atm)] → convert to [mol/m³/µatm] = K₀*1000/1e6"""
    if S is None: S = S_REF
    T = T_K / 100.0
    ln_K0 = A1 + A2/T + A3*np.log(T) + S*(B1 + B2*T + B3*T**2)
    # K0 in mol/(L·atm) → mol/(m³·µatm) = *1000 / 1e6 = *1e-3
    return np.exp(ln_K0) * 1e-3

def timestamps_to_decimal(ts):
    out = []
    for t in pd.DatetimeIndex(ts):
        y = t.year; s = pd.Timestamp(f'{y}-01-01'); e = pd.Timestamp(f'{y+1}-01-01')
        out.append(y + (t-s).total_seconds()/(e-s).total_seconds())
    return np.array(out)

def build():
    hoaps = pd.read_csv(os.path.join(OUTPUT_DIR,'processed_freshwater_fluxes_1987_2014.csv'),
                        index_col=0, parse_dates=True)
    hoaps = hoaps[['F_NA','F_SA','F_SO','F_PO','F_IO']].resample('MS').mean().dropna()

    ode = pd.read_csv(os.path.join(OUTPUT_DIR,'co2_salinity_simulation_1987_2014.csv'),
                      index_col=0, parse_dates=True).resample('MS').mean().dropna()

    common = hoaps.index.intersection(ode.index)
    hoaps  = hoaps.loc[common]; ode = ode.loc[common]

    t_dec  = timestamps_to_decimal(common)
    t_mo   = (t_dec - T_REF) * 12.0

    # SST
    sst = np.outer(t_mo, SST_SLOPES) + SST_INTERCEPTS  # (N,5) in K
    # pCO2
    pco2 = PCO2_SLOPE * t_mo + PCO2_OFFSET  # µatm

    # K₀ per basin per month (using mean S per basin, good approx)
    K0_mat = np.zeros((len(common), 5))
    for i in range(len(common)):
        K0_mat[i] = k0(sst[i])  # mol/(m³·µatm)

    # CO₂(aq) from ODE [mol/m³]
    C_cols = ['C_NA','C_SA','C_SO','C_PO','C_IO']
    C_mat  = ode[C_cols].values  # (N,5) mol/m³

    # Air-sea disequilibrium = K₀ * pCO₂_air - C_i(t-1)  [mol/m³]
    # Use LAGGED C to avoid data leakage (C_i(t) would encode ΔC_i(t)).
    # This is also physically correct: gas exchange at start of month t
    # is driven by the concentration at end of month t-1.
    C_lag  = np.roll(C_mat, 1, axis=0); C_lag[0] = C_mat[0]
    diseq  = K0_mat * pco2[:, None] - C_lag  # positive → ocean absorbs CO₂

    # Targets
    sal_cols = ['S_NA','S_SA','S_SO','S_PO','S_IO']
    y_sal = ode[sal_cols].copy()
    y_co2 = ode[C_cols].copy() * 1e3  # mol → mmol/m³

    # Feature matrix
    months = pd.DatetimeIndex(common).month
    year_norm = (t_dec - t_dec.min()) / (t_dec.max() - t_dec.min())

    X = pd.DataFrame(index=common)
    for c in hoaps.columns:
        X[c] = hoaps[c].values
    for j, b in enumerate(BASINS):
        X[f'SST_{b}'] = sst[:, j]
    X['pCO2_air'] = pco2
    X['month_sin'] = np.sin(2*np.pi*months/12)
    X['month_cos'] = np.cos(2*np.pi*months/12)
    X['year_norm'] = year_norm
    for j, b in enumerate(BASINS):
        X[f'DISEQ_{b}'] = diseq[:, j]  # key physics-informed feature

    split_date = pd.Timestamp('2010-01-01')
    tr = np.array(X.index < split_date)
    te = np.array(X.index >= split_date)


    scaler_X = StandardScaler()
    scaler_X.fit(X.values[tr].astype(float))

    print(f"Dataset: {len(X)} months | X={X.shape} | features={list(X.columns)}")
    print(f"Train: {tr.sum()} | Test: {te.sum()}")

    X.to_csv(os.path.join(OUTPUT_DIR,'ml_features.csv'))
    y_sal.to_csv(os.path.join(OUTPUT_DIR,'ml_targets_salinity.csv'))
    y_co2.to_csv(os.path.join(OUTPUT_DIR,'ml_targets_co2.csv'))

    meta = dict(common_index=common, train_mask=tr, test_mask=te,
                scaler_X=scaler_X, t_decimal=t_dec)
    with open(os.path.join(OUTPUT_DIR,'ml_meta.pkl'),'wb') as f:
        pickle.dump(meta, f)
    print("Saved ml_features.csv, ml_targets_salinity.csv, ml_targets_co2.csv, ml_meta.pkl")

if __name__ == '__main__':
    build()
