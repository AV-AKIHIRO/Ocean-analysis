#!/usr/bin/env python3
"""
ml_forecast.py  (v2)
====================
Autoregressively forecasts salinity and CO₂ from 2015 to 2030.

Method:
  - Best SALINITY model: Ridge Regression (R²=0.984)
  - Best CO₂ model:      Linear Regression (R²=0.950)
  - Feature extrapolation: linear trend on HOAPS F_i, SST from Appendix D,
    Keeling curve for pCO₂_air. DISEQ updated autoregressively at each step.
  - ODE physics: linear-trend extrapolation of last 3 years of simulation.
"""

import os, sys, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

WORKSPACE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR   = os.path.join(WORKSPACE, 'output')
ARTIFACT_DIR = '/home/areen/.gemini/antigravity-ide/brain/6befa23f-0f3a-4ee5-a72c-d811e3201d63'
BASINS = ['NA', 'SA', 'SO', 'PO', 'IO']
LOOK_BACK = 12

T_REF = 2000.0 + 15.0/365.25
SST_SLOPES     = np.array([0.0017, -0.00015, -0.00056, 0.00155, 0.00026])
SST_INTERCEPTS = np.array([291.9,   290.3,    272.29,   292.88,  291.57])
PCO2_SLOPE, PCO2_OFFSET = 0.18152, 367.41
A1 = np.full(5, -58.0931); A2 = np.full(5, 90.5069); A3 = np.full(5, 22.2940)
B1 = np.full(5, 0.027766); B2 = np.full(5,-0.025888); B3 = np.full(5, 0.0050578)
S_REF = np.array([34.912, 35.435, 34.427, 34.668, 34.538])

MODEL_COLORS = {'ODE': '#000000', 'Ridge (Salinity)': '#2ca02c',
                'Linear (CO2)': '#1f77b4'}
COLORS = ['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def k0_vec(T_K):
    T = T_K / 100.0
    ln_K0 = A1 + A2/T + A3*np.log(T) + S_REF*(B1 + B2*T + B3*T**2)
    return np.exp(ln_K0) * 1e-3  # mol/(m³·µatm)

def timestamps_to_decimal(ts):
    out = []
    for t in pd.DatetimeIndex(ts):
        y = t.year; s = pd.Timestamp(f'{y}-01-01'); e = pd.Timestamp(f'{y+1}-01-01')
        out.append(y + (t-s).total_seconds()/(e-s).total_seconds())
    return np.array(out)

def build_base_row(t_dec, dt, f_vals):
    """Build base feature row (no DISEQ) at time t."""
    t_mo = (t_dec - T_REF) * 12.0
    sst  = SST_INTERCEPTS + SST_SLOPES * t_mo
    pco2 = PCO2_SLOPE * t_mo + PCO2_OFFSET
    row = list(f_vals) + list(sst) + [pco2,
          np.sin(2*np.pi*dt.month/12),
          np.cos(2*np.pi*dt.month/12)]
    return np.array(row), sst, pco2  # row has 13 entries (no year_norm, no DISEQ yet)

def main():
    print('='*65)
    print('Forecast 2015–2030: Best ML Models vs ODE Physics')
    print('='*65)

    # ── Load training data ────────────────────────────────────────
    X_hist = pd.read_csv(os.path.join(OUTPUT_DIR,'ml_features.csv'),
                         index_col=0, parse_dates=True)
    y_sal  = pd.read_csv(os.path.join(OUTPUT_DIR,'ml_targets_salinity.csv'),
                         index_col=0, parse_dates=True)
    y_co2  = pd.read_csv(os.path.join(OUTPUT_DIR,'ml_targets_co2.csv'),
                         index_col=0, parse_dates=True)
    with open(os.path.join(OUTPUT_DIR,'ml_meta.pkl'),'rb') as f:
        meta = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR,'ml_results_Salinity.pkl'),'rb') as f:
        res_sal = pickle.load(f)
    with open(os.path.join(OUTPUT_DIR,'ml_results_CO2.pkl'),'rb') as f:
        res_co2 = pickle.load(f)

    t_hist    = timestamps_to_decimal(X_hist.index)
    t_min, t_max = t_hist.min(), t_hist.max()
    tr = meta['train_mask']; te = meta['test_mask']

    sal_cols = list(y_sal.columns)
    co2_cols = list(y_co2.columns)
    n_sal    = len(sal_cols); n_co2 = len(co2_cols)

    X_arr   = X_hist.values.astype(float)
    y_s_arr = y_sal.values.astype(float)
    y_c_arr = y_co2.values.astype(float)

    feature_cols = list(X_hist.columns)
    hoaps_cols   = ['F_NA','F_SA','F_SO','F_PO','F_IO']

    # ── Fit linear trend to HOAPS F_i for extrapolation ──────────
    f_slopes = []; f_intercepts = []
    for col in hoaps_cols:
        sl, ic, *_ = linregress(t_hist, X_hist[col].values)
        f_slopes.append(sl); f_intercepts.append(ic)

    def get_f(t_dec):
        return np.array([f_slopes[i]*t_dec + f_intercepts[i] for i in range(5)])

    # ── Re-train best models on ALL available data (1987–2014) ────
    # AR differencing: augment X with y_lag
    print('\n[1/4] Re-training best models on full 1987–2014 data...')

    # Salinity → Ridge
    y_s_lag  = np.roll(y_s_arr,1,axis=0); y_s_lag[0] = y_s_arr[0]
    Xa_s     = np.hstack([X_arr, y_s_lag])
    dy_s     = np.diff(y_s_arr, axis=0, prepend=y_s_arr[[0]])
    scaler_Xs = StandardScaler().fit(Xa_s)
    Xa_s_sc   = scaler_Xs.transform(Xa_s)
    scalers_s = [StandardScaler().fit(dy_s[:, i:i+1]) for i in range(n_sal)]
    dy_s_sc   = np.column_stack([scalers_s[i].transform(dy_s[:,i:i+1]).ravel() for i in range(n_sal)])
    ridge_sal = Ridge(alpha=1.0).fit(Xa_s_sc, dy_s_sc)
    print('  Ridge (Salinity) fitted on full data.')

    # CO₂ → Linear
    y_c_lag  = np.roll(y_c_arr,1,axis=0); y_c_lag[0] = y_c_arr[0]
    Xa_c     = np.hstack([X_arr, y_c_lag])
    dy_c     = np.diff(y_c_arr, axis=0, prepend=y_c_arr[[0]])
    scaler_Xc = StandardScaler().fit(Xa_c)
    Xa_c_sc   = scaler_Xc.transform(Xa_c)
    scalers_c = [StandardScaler().fit(dy_c[:,i:i+1]) for i in range(n_co2)]
    dy_c_sc   = np.column_stack([scalers_c[i].transform(dy_c[:,i:i+1]).ravel() for i in range(n_co2)])
    lr_co2   = LinearRegression().fit(Xa_c_sc, dy_c_sc)
    print('  Linear (CO₂) fitted on full data.')

    # ── ODE physics: linear-trend extrapolation from last 36 months ─
    print('\n[2/4] Extrapolating ODE physics trend (2015–2030)...')
    n_trend = 36
    future_dates = pd.date_range('2015-01-01','2030-12-01', freq='MS')
    t_future     = timestamps_to_decimal(future_dates)

    def extrapolate_ode(y_df, n_trend):
        y_a = y_df.values; t_a = t_hist
        ode_f = np.zeros((len(future_dates), y_df.shape[1]))
        for i in range(y_df.shape[1]):
            sl, ic, *_ = linregress(t_a[-n_trend:], y_a[-n_trend:, i])
            ode_f[:, i] = sl * t_future + ic
        return ode_f

    ode_sal_f = extrapolate_ode(y_sal, n_trend)
    ode_co2_f = extrapolate_ode(y_co2, n_trend)

    # ── AR forward rollout for ML models ──────────────────────────
    print('\n[3/4] Running autoregressive ML forecast (2015–2030)...')

    # Start states: last known value from ODE simulation (end of 2014)
    s_state = y_s_arr[-1].copy()   # psu
    c_state = y_c_arr[-1].copy()   # mmol/m³

    n_fut       = len(future_dates)
    ridge_s_f   = np.zeros((n_fut, n_sal))
    lr_co2_f    = np.zeros((n_fut, n_co2))

    # year_norm index: extend beyond training max
    def year_norm(t_dec):
        return (t_dec - t_min) / (t_max - t_min)

    for i, (dt, t_dec) in enumerate(zip(future_dates, t_future)):
        t_mo = (t_dec - T_REF) * 12.0
        sst  = SST_INTERCEPTS + SST_SLOPES * t_mo
        pco2 = PCO2_SLOPE * t_mo + PCO2_OFFSET
        f_v  = get_f(t_dec)
        K0   = k0_vec(sst)

        # DISEQ using current state (lagged by 1 step in rollout = s_state/c_state)
        # CO₂ state is in mmol/m³ → convert to mol/m³ for K₀
        c_mol  = c_state * 1e-3
        diseq  = K0 * pco2 - c_mol   # [mol/m³]

        base_row = np.array(list(f_v) + list(sst) +
                            [pco2, np.sin(2*np.pi*dt.month/12),
                             np.cos(2*np.pi*dt.month/12),
                             year_norm(t_dec)] + list(diseq))  # length=19

        # Salinity step
        xs_row = np.hstack([base_row, s_state])                  # 19+5=24
        xs_sc  = scaler_Xs.transform(xs_row[np.newaxis])
        ds_sc  = ridge_sal.predict(xs_sc)[0]                     # (5,) scaled
        ds     = np.array([scalers_s[j].inverse_transform([[ds_sc[j]]])[0,0]
                           for j in range(n_sal)])
        s_state = s_state + ds
        ridge_s_f[i] = s_state

        # CO₂ step (state updated after salinity for consistency)
        xc_row = np.hstack([base_row, c_state])                  # 19+5=24
        xc_sc  = scaler_Xc.transform(xc_row[np.newaxis])
        dc_sc  = lr_co2.predict(xc_sc)[0]
        dc     = np.array([scalers_c[j].inverse_transform([[dc_sc[j]]])[0,0]
                           for j in range(n_co2)])
        c_state = c_state + dc
        lr_co2_f[i] = c_state

    # ── PLOTTING ──────────────────────────────────────────────────
    print('\n[4/4] Generating forecast plots...')

    def plot_forecast(y_hist, ode_f, ml_f, ml_label, future_dates,
                      label, unit, col_names):
        dt_hist   = pd.DatetimeIndex(y_hist.index)
        dt_future = pd.DatetimeIndex(future_dates)
        split_dt  = pd.Timestamp('2015-01-01')
        n_b = len(col_names)
        fig, axes = plt.subplots(n_b, 1, figsize=(16, 4*n_b), sharex=True)
        fig.suptitle(
            f'2015–2030 Forecast: {label}\n'
            f'ODE Physics (trend extrapolation) vs {ml_label} (AR rollout)',
            fontsize=12, fontweight='bold', y=1.001)
        for bi, (ax, col, color) in enumerate(zip(axes, col_names, COLORS)):
            basin = col.split('_')[1]
            ax.plot(dt_hist, y_hist.values[:,bi], color='black',
                    lw=2.0, label='ODE historical', alpha=0.95, zorder=10)
            ax.axvline(split_dt, color='gray', ls=':', lw=1.2, alpha=0.7)
            ax.axvspan(split_dt, dt_future[-1], alpha=0.05, color='blue')
            ax.plot(dt_future, ode_f[:,bi], color='black', ls='--',
                    lw=1.6, label='ODE extrapolated', alpha=0.85)
            ax.plot(dt_future, ml_f[:,bi], color=color, ls='-',
                    lw=1.8, label=f'{ml_label}', alpha=0.90)
            # 2030 endpoint annotation
            ax.annotate(f"ODE: {ode_f[-1,bi]:.4f}\nML: {ml_f[-1,bi]:.4f}",
                        xy=(dt_future[-1], ml_f[-1,bi]),
                        xytext=(-50, 10), textcoords='offset points',
                        fontsize=7.5, color=color,
                        arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
            ax.set_ylabel(f'{label} [{unit}]', fontsize=9)
            ax.set_title(basin, fontsize=9, fontweight='bold')
            ax.grid(True, ls='--', alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(3))
            if bi == 0:
                ax.legend(fontsize=9, ncol=2, loc='upper left')
        axes[-1].set_xlabel('Year', fontsize=11)
        plt.tight_layout()
        fname = f'forecast_2030_{label.lower()}.png'
        for path in [os.path.join(OUTPUT_DIR, fname),
                     os.path.join(ARTIFACT_DIR, fname)]:
            plt.savefig(path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f'  Saved {fname}')

    plot_forecast(y_sal, ode_sal_f, ridge_s_f, 'Ridge Regression',
                  future_dates, 'Salinity', 'psu', sal_cols)
    plot_forecast(y_co2, ode_co2_f, lr_co2_f, 'Linear Regression',
                  future_dates, 'CO2', 'mmol/m³', co2_cols)

    # ── Print 2030 summary table ──────────────────────────────────
    print('\n' + '='*65)
    print('2030 PROJECTION SUMMARY')
    print('='*65)
    print(f'\n{"Basin":<8} {"Sal ODE(2030)":>15} {"Sal ML(2030)":>14}'
          f' {"Δ(ML-ODE)":>10}')
    print('-'*50)
    for bi, col in enumerate(sal_cols):
        basin = col.split('_')[1]
        ode_v = ode_sal_f[-1,bi]; ml_v = ridge_s_f[-1,bi]
        print(f'{basin:<8} {ode_v:>15.5f} {ml_v:>14.5f} {ml_v-ode_v:>10.5f}')

    print(f'\n{"Basin":<8} {"CO2 ODE(2030)":>15} {"CO2 ML(2030)":>14}'
          f' {"Δ(ML-ODE)":>10}')
    print('-'*50)
    for bi, col in enumerate(co2_cols):
        basin = col.split('_')[1]
        ode_v = ode_co2_f[-1,bi]; ml_v = lr_co2_f[-1,bi]
        print(f'{basin:<8} {ode_v:>15.4f} {ml_v:>14.4f} {ml_v-ode_v:>10.4f}')

    print('\nForecast complete.')

if __name__ == '__main__':
    main()
