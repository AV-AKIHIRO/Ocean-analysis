#!/usr/bin/env python3
"""
Five-Box Ocean CO₂ & Salinity Coupled ODE Simulation
=====================================================
Based on: Sunny et al., Chaos 33, 103134 (2023)

Extension: Non-autonomous 15-dimensional ODE (C₁..C₅, N₁..N₅, S₁..S₅)
integrated over 1987–2014 using time-varying HOAPS freshwater forcing.

State vector layout (0-indexed):
  y[0:5]   = C_i   — aqueous CO₂ concentration  [mol/m³]
  y[5:10]  = N_i   — carbonic acid / reaction product [mol/m³]
  y[10:15] = S_i   — salinity [psu]

  Index 0=NA, 1=SA, 2=SO, 3=PO, 4=IO  (paper 1..5 shifted by 1)
"""

import os, sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
WORKSPACE    = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR   = os.path.join(WORKSPACE, 'output')
ARTIFACT_DIR = '/home/areen/.gemini/antigravity-ide/brain/6befa23f-0f3a-4ee5-a72c-d811e3201d63'
CSV_PATH     = os.path.join(OUTPUT_DIR, 'processed_freshwater_fluxes_1987_2014.csv')

# ============================================================
# UNIVERSAL CONSTANTS
# ============================================================
SEC_PER_YEAR  = 365.25 * 24.0 * 3600.0
SV_TO_M3S     = 1.0e6

# ============================================================
# MODEL GEOMETRY  (Appendix A)
# ============================================================
BASIN_NAMES = ['North Atlantic', 'South Atlantic', 'Southern Ocean',
               'Pacific Ocean',  'Indian Ocean']
BASIN_CODES = ['NA', 'SA', 'SO', 'PO', 'IO']

V = np.array([146.0e15, 160.0e15, 71.8e15, 710.0e15, 264.0e15])   # [m³]
A = np.array([ 41.49e12, 40.27e12, 20.33e12, 165.25e12, 70.56e12]) # [m²]

# ============================================================
# INITIAL CONDITIONS  (Table I, year 2000)
# ============================================================
C_INIT = np.array([0.01413, 0.01579, 0.01400, 0.01834, 0.01671])   # mol/m³
K_EQ   = 0.0011
N_INIT = K_EQ * C_INIT
S_INIT = np.array([34.912, 35.435, 34.427, 34.668, 34.538])        # psu

# ----------------------------------------------------------------
# QSS treatment of fast chemistry (paper Eq. 9: dN/dt = K_eq*dC/dt)
# ----------------------------------------------------------------
# Since chemistry equilibrates instantly:  N_i = K_eq * C_i  always.
# Total DIC: T_i = C_i + N_i = C_i*(1 + K_eq)
# Adding Eq.7 + Eq.8 eliminates all K_c / K_n terms:
#   dT_i/dt = (1+K_eq) * [gas_exchange(C_i) + transport(C_i, N_i)]
# where C_i = T_i / (1+K_eq)  and  N_i = K_eq*T_i/(1+K_eq)
#
# State vector: y[0:5] = T_i [mol/m³],  y[5:10] = S_i [psu]
ALPHA  = 1.0 + K_EQ          # = 1.0011
T_INIT = ALPHA * C_INIT       # total DIC initial values [mol/m³]
Y_INIT = np.concatenate([T_INIT, S_INIT])
C_AMAZ = C_INIT[0]           # Amazon CO₂ inflow concentration [mol/m³]

# ============================================================
# PARAMETERS  (Table II)
# ============================================================
S0_REF = 35.0; BETA = 7.5e-4; K_DEEP = 60.0
K_C = 0.037; K_N = 23.0                       # chemical rates [s⁻¹]

# Weiss (1974) K₀ constants (Appendix B)
A1_K0, A2_K0, A3_K0 =  -58.0931, 90.5069, 22.294
B1_K0, B2_K0, B3_K0 =  0.027766, -0.025888, 0.0050578

# Deep CO₂ pressure-correction constants (Ref 27)
a_c=3.63579e-5; b_c=4.47782e-5; g_c=1.8833e-5
l_c=7.31010e-5; h_c=5.41469e-5; psi_c=7.49356e-5
M_SAL = 18.015 + 0.015 * 35.0   # mean molar mass [g/mol]
RHO0  = 1029.0                   # [kg/m³]

U10 = 6.0   # 10-m wind speed [m/s]

# Keeling curve: pCO₂_air(t) = slope*t + offset  (t in months from Jan 2000)
PCO2_SLOPE = 0.18152; PCO2_OFFSET = 367.41   # [µatm/month, µatm]

# SST linear fits: T_SS_i(t) = intercept + slope*t  [K], t in months from Jan 2000
SST_SLOPES     = np.array([0.0017, -0.00015, -0.00056, 0.00155, 0.00026])
SST_INTERCEPTS = np.array([291.9,   290.3,    272.29,   292.88,  291.57])

T_REF_DECIMAL = 2000.0 + 15.0/365.25   # Jan 2000 in decimal years

# ============================================================
# FRESHWATER TERMS  (§II.C)
# ============================================================
b_NA   =  3.28e3 / SV_TO_M3S    # Arctic ice melt [Sv], P-E positive
b_SO   = -6.944e2 / SV_TO_M3S   # Antarctic ice [Sv], negative = forming
q_AMAZ =  0.24                   # Amazon [Sv]

# ============================================================
# CIRCULATION
# ============================================================
SURFACE_CURRENTS = [(1,0,26.),(2,1,30.),(3,4,15.),(4,1,10.),(4,2,60.)]
DEEP_PAIRS       = [(0,1),(1,2),(1,3),(2,3),(2,4)]

# ============================================================
# TIME HELPERS
# ============================================================
def dec2months(t): return (t - T_REF_DECIMAL) * 12.0
def timestamps_to_decimal_years(ts):
    out = []
    for t in pd.DatetimeIndex(ts):
        y = t.year; s = pd.Timestamp(f'{y}-01-01'); e = pd.Timestamp(f'{y+1}-01-01')
        out.append(y + (t-s).total_seconds()/(e-s).total_seconds())
    return np.array(out)
def dy_to_ts(dy):
    y=int(dy); f=dy-y; s=pd.Timestamp(f'{y}-01-01'); e=pd.Timestamp(f'{y+1}-01-01')
    return s + pd.Timedelta(seconds=f*(e-s).total_seconds())

# ============================================================
# CO₂ CHEMISTRY
# ============================================================
def sst_K(t):
    return SST_INTERCEPTS + SST_SLOPES * dec2months(t)

def pco2_atm(t):
    return (PCO2_SLOPE * dec2months(t) + PCO2_OFFSET) * 1e-6  # [atm]

def k0(T_K, S):
    T100 = T_K / 100.0
    lnK0 = (A1_K0 + A2_K0/T100 + A3_K0*np.log(T100)
             + S*(B1_K0 + B2_K0*T100 + B3_K0*T100**2))
    return np.exp(lnK0) * 1e3   # mol/(m³·atm)

def schmidt(T_K):
    T = T_K - 273.15
    return 2116.8 - 136.5*T + 4.7353*T**2 - 0.092307*T**3 + 0.0007555*T**4

def gamma(T_K):
    g660 = 0.24 * U10**2 / 100.0 / SEC_PER_YEAR   # m/s
    return g660 * (schmidt(T_K)/660.0)**(-0.5)

def C_prime(C, T_K):
    M_rho = M_SAL / (RHO0/1000.0)
    T = T_K - 273.15
    corr = ((1+a_c-b_c*T) + (g_c*T+l_c)*M_rho*C + (h_c*T-psi_c)*M_rho**2*C**2)
    return C * corr

# ============================================================
# FORCING LOAD
# ============================================================
def load_forcing(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    t  = timestamps_to_decimal_years(df.index)
    F  = df[['F_NA','F_SA','F_SO','F_PO','F_IO']].values
    sp = [CubicSpline(t, F[:,i], extrapolate=True) for i in range(5)]
    return sp, t, F

# ============================================================
# 15-DIM ODE
# ============================================================
def ode(t, y, F_sp):
    """
    10-dimensional ODE using total DIC T_i = C_i*(1+K_eq) and S_i.

    QSS treatment (paper Eq.9: N_i = K_eq*C_i always):
      d(T_i)/dt = d(C_i+N_i)/dt = (1+K_eq)*dC_i/dt  [chemistry cancels]
    C_i = T_i / ALPHA  is recovered at every step.

    y[0:5]  = T_i  [mol/m³]  total dissolved inorganic carbon
    y[5:10] = S_i  [psu]
    """
    T = y[0:5]; S = y[5:10]
    C = T / ALPHA        # aqueous CO₂ [mol/m³]
    C_pr = C_prime(C, sst_K(t))   # deep-sea pressure-corrected

    dT = np.zeros(5); dS = np.zeros(5)

    Tk  = sst_K(t)
    K0v = k0(Tk, S)
    gv  = gamma(Tk)
    pa  = pco2_atm(t)
    Fh  = np.array([float(sp(t)) for sp in F_sp])
    q   = np.array([K_DEEP*BETA*(S[i]-S[j]) for i,j in DEEP_PAIRS])

    # ----------------------------------------------------------------
    # T equation  = (1+K_eq) * [gas_exchange + Amazon + transport]
    # Chemistry terms vanish: dC/dt(-K_c*C+K_n*N) + dN/dt(K_c*C-K_n*N) = 0
    # ----------------------------------------------------------------
    # Gas exchange acts on C_i, but dT = ALPHA*dC so multiply through
    for i in range(5):
        dT[i] += ALPHA * gv[i]*A[i]/V[i] * (K0v[i]*pa - C[i]) * SEC_PER_YEAR

    # Amazon inflow (C concentration, ALPHA factor for T)
    dT[0] += ALPHA * q_AMAZ*SV_TO_M3S*C_AMAZ/V[0] * SEC_PER_YEAR

    # Surface transport (carries T_i, not C_i, since T tracks C)
    for src, dst, Qsv in SURFACE_CURRENTS:
        Q = Qsv * SV_TO_M3S
        dT[dst] += Q * (T[src] - T[dst]) / V[dst] * SEC_PER_YEAR

    # Deep transport (with deep-sea pressure correction on C')
    # T' = ALPHA * C'  for the corrected concentration
    Tp = ALPHA * C_pr
    for k, (i, j) in enumerate(DEEP_PAIRS):
        qm = q[k] * SV_TO_M3S
        dT[i] -= qm * (Tp[i] - Tp[j]) / V[i] * SEC_PER_YEAR
        dT[j] += qm * (Tp[i] - Tp[j]) / V[j] * SEC_PER_YEAR

    # ----------------------------------------------------------------
    # S equation  (Eq. 4, exchange form)
    # ----------------------------------------------------------------
    fw = -Fh.copy()           # convert HOAPS E-P → paper's P-E
    fw[0] += b_NA + q_AMAZ
    fw[2] += b_SO
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


# ============================================================
# PLOTTING
# ============================================================
COLORS = ['#1f77b4','#d62728','#2ca02c','#ff7f0e','#9467bd']

def _dt(t_eval):
    return pd.DatetimeIndex([dy_to_ts(t) for t in t_eval])

def plot_co2_overview(t_eval, C_out):
    """Single-panel CO₂(aq) overview — mirrors paper Fig.4."""
    dt = _dt(t_eval)
    fig, ax = plt.subplots(figsize=(14,6))
    for i,(code,color) in enumerate(zip(BASIN_CODES,COLORS)):
        ax.plot(dt, C_out[:,i]*1e3, color=color, lw=1.5, label=code)
    ax.set_ylabel('CO₂(aq)  [mmol/m³]', fontsize=11)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_title('Five-Box Ocean CO₂(aq) Time-Evolution  (1987–2014)\n'
                 'Time-Varying HOAPS Freshwater Forcing  |  Sunny et al. (2023) Extended',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, ncol=5)
    ax.grid(True, ls='--', alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    plt.tight_layout()
    for p in [os.path.join(OUTPUT_DIR,'co2_overview_1987_2014.png'),
              os.path.join(ARTIFACT_DIR,'co2_overview_1987_2014.png')]:
        plt.savefig(p, dpi=150, bbox_inches='tight'); print(f'  Saved → {p}')
    plt.close()

def plot_co2_per_basin(t_eval, C_out):
    dt = _dt(t_eval)
    fig, axes = plt.subplots(5, 1, figsize=(14,18), sharex=True)
    for i,(ax,name,code,color) in enumerate(zip(axes,BASIN_NAMES,BASIN_CODES,COLORS)):
        Cm = C_out[:,i]*1e3
        ax.plot(dt, Cm, color=color, lw=1.4, label=f'$C_{{\\mathrm{{{code}}}}}(t)$')
        ax.axhline(C_INIT[i]*1e3, color='k', ls='--', lw=0.9, alpha=0.6,
                   label=f'Initial ({C_INIT[i]*1e3:.3f} mmol/m³)')
        dc = C_out[-1,i]-C_INIT[i]; pct = dc/C_INIT[i]*100; sign='+' if dc>=0 else ''
        ax.set_title(f'{name}  —  ΔC={sign}{dc*1e3:.4f} mmol/m³  ({sign}{pct:.2f}%)',
                     fontsize=10, fontweight='bold')
        ax.set_ylabel('CO₂(aq)  [mmol/m³]', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, ls='--', alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
    axes[-1].set_xlabel('Year', fontsize=11)
    fig.suptitle('Five-Box CO₂(aq) Evolution Per Basin (1987–2014)\n'
                 'Sunny et al. (2023) Extended  |  HOAPS Forcing',
                 fontsize=13, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0,0,1,0.975])
    for p in [os.path.join(OUTPUT_DIR,'co2_per_basin_1987_2014.png'),
              os.path.join(ARTIFACT_DIR,'co2_per_basin_1987_2014.png')]:
        plt.savefig(p, dpi=150, bbox_inches='tight'); print(f'  Saved → {p}')
    plt.close()

def plot_salinity(t_eval, S_out):
    dt = _dt(t_eval)
    fig, axes = plt.subplots(5, 1, figsize=(14,18), sharex=True)
    for i,(ax,name,code,color) in enumerate(zip(axes,BASIN_NAMES,BASIN_CODES,COLORS)):
        ax.plot(dt, S_out[:,i], color=color, lw=1.4, label=f'$S_{{\\mathrm{{{code}}}}}(t)$')
        ax.axhline(S_INIT[i], color='k', ls='--', lw=0.9, alpha=0.6,
                   label=f'Initial ({S_INIT[i]:.3f} psu)')
        ds = S_out[-1,i]-S_INIT[i]; sign='+' if ds>=0 else ''
        ax.set_title(f'{name}  —  Δ={sign}{ds:.4f} psu  '
                     f'({S_INIT[i]:.3f}→{S_out[-1,i]:.3f} psu)',
                     fontsize=10, fontweight='bold')
        ax.set_ylabel('Salinity (psu)', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, ls='--', alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
    axes[-1].set_xlabel('Year', fontsize=11)
    fig.suptitle('Five-Box Salinity Evolution (1987–2014)  —  Coupled System\n'
                 'Sunny et al. (2023) Extended  |  HOAPS Forcing',
                 fontsize=13, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0,0,1,0.975])
    for p in [os.path.join(OUTPUT_DIR,'salinity_coupled_1987_2014.png'),
              os.path.join(ARTIFACT_DIR,'salinity_coupled_1987_2014.png')]:
        plt.savefig(p, dpi=150, bbox_inches='tight'); print(f'  Saved → {p}')
    plt.close()

def plot_sink_source(t_eval, C_out, S_out):
    """Net air-sea CO₂ flux and cumulative uptake."""
    dt = _dt(t_eval)
    flux = np.zeros((len(t_eval), 5))
    for k, t in enumerate(t_eval):
        T  = sst_K(t); K0v = k0(T, S_out[k]); gv = gamma(T); pa = pco2_atm(t)
        for i in range(5):
            flux[k,i] = gv[i]*(K0v[i]*pa - C_out[k,i]) * SEC_PER_YEAR   # mol/(m²·yr)

    fig, axes = plt.subplots(2, 1, figsize=(14,9), sharex=True)
    ax = axes[0]
    for i,(code,color) in enumerate(zip(BASIN_CODES,COLORS)):
        ax.plot(dt, flux[:,i]*1e3, color=color, lw=1.4, label=code)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.fill_between(dt, flux[:,0]*1e3, 0, where=flux[:,0]>0,
                    color=COLORS[0], alpha=0.12)  # shade NA sink
    ax.set_ylabel('Net CO₂ flux  [mmol/(m²·yr)]', fontsize=11)
    ax.set_title('Air-Sea CO₂ Flux  (positive = ocean absorbs = carbon sink)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10, ncol=5)
    ax.grid(True, ls='--', alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))

    ax2 = axes[1]
    dt_yr = np.gradient(t_eval)
    for i,(code,color) in enumerate(zip(BASIN_CODES,COLORS)):
        uptake = flux[:,i] * A[i]                          # mol/yr
        cum    = np.cumsum(uptake * dt_yr) / 1e15           # Pmol
        ax2.plot(dt, cum, color=color, lw=1.4, label=code)
    ax2.axhline(0, color='k', lw=0.8, ls='--')
    ax2.set_ylabel('Cumulative CO₂ uptake  [Pmol]', fontsize=11)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_title('Cumulative Ocean CO₂ Uptake (1987–2014)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10, ncol=5)
    ax2.grid(True, ls='--', alpha=0.4)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(3))

    fig.suptitle('Carbon Sink/Source Dynamics (1987–2014)\n'
                 'Sunny et al. (2023) Extended  |  Time-Varying HOAPS Forcing',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    for p in [os.path.join(OUTPUT_DIR,'co2_sink_source_1987_2014.png'),
              os.path.join(ARTIFACT_DIR,'co2_sink_source_1987_2014.png')]:
        plt.savefig(p, dpi=150, bbox_inches='tight'); print(f'  Saved → {p}')
    plt.close()

# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('='*65)
    print('Five-Box Ocean Coupled CO₂–Salinity Simulation')
    print('Sunny et al. (2023)  —  Extended HOAPS 1987–2014')
    print('='*65)

    print('\n[1/5] Loading HOAPS freshwater forcing...')
    F_sp, t_data, F_data = load_forcing(CSV_PATH)
    t0, t1 = t_data[0], t_data[-1]
    print(f'      Range: {t0:.3f} – {t1:.3f}  ({len(t_data)} months)')

    print('\n[2/5] Integrating 10-dim QSS ODE (C+S only, N=K_eq*C, RK45)...')
    t_eval = np.linspace(t0, t1, 5000)
    sol = solve_ivp(lambda t,y: ode(t, y, F_sp), (t0,t1), Y_INIT,
                    method='RK45', t_eval=t_eval, rtol=1e-8, atol=1e-10)
    if not sol.success:
        print(f'ERROR: {sol.message}'); sys.exit(1)
    print(f'  OK  ({sol.nfev} evaluations, {len(sol.t)} pts)')

    T_out = sol.y[0:5].T          # total DIC [mol/m³]
    S_out = sol.y[5:10].T
    C_out = T_out / ALPHA         # aqueous CO₂ [mol/m³]
    N_out = K_EQ * C_out          # H₂CO₃ proxy [mol/m³]


    print('\n[3/5] Saving CSV...')
    idx   = pd.DatetimeIndex([dy_to_ts(t) for t in sol.t], name='date')
    # Build full output including diagnosed N
    df_data = np.hstack([C_out, N_out, S_out])
    cols  = ([f'C_{c}' for c in BASIN_CODES]
             + [f'N_{c}' for c in BASIN_CODES]
             + [f'S_{c}' for c in BASIN_CODES])
    pd.DataFrame(df_data, index=idx, columns=cols).to_csv(
        os.path.join(OUTPUT_DIR, 'co2_salinity_simulation_1987_2014.csv'))
    print(f'  Saved → {os.path.join(OUTPUT_DIR, "co2_salinity_simulation_1987_2014.csv")}')

    n_yr = t1 - t0
    print('\n'+'='*65)
    print('CO₂(aq) EVOLUTION SUMMARY')
    print('='*65)
    print(f"{'Basin':<22} {'Init (mmol/m³)':<17} {'Final':<17} {'Δ (%)'}")
    print('-'*65)
    for i,name in enumerate(BASIN_NAMES):
        c0=C_INIT[i]*1e3; cf=C_out[-1,i]*1e3; pct=(cf-c0)/c0*100
        print(f'  {name:<20} {c0:<17.4f} {cf:<17.4f} {pct:+.2f}%')

    print('\n'+'='*65)
    print('SALINITY EVOLUTION SUMMARY')
    print('='*65)
    print(f"{'Basin':<22} {'Init (psu)':<14} {'Final':<14} {'Δ (psu)':<12} {'mpsu/yr'}")
    print('-'*74)
    for i,name in enumerate(BASIN_NAMES):
        s0=S_INIT[i]; sf=S_out[-1,i]; ds=sf-s0
        print(f'  {name:<20} {s0:<14.4f} {sf:<14.4f} {ds:<12.5f} {ds/n_yr*1000:+.3f}')

    print('\n[4/5] Generating time-series plots...')
    plot_co2_overview(sol.t, C_out)
    plot_co2_per_basin(sol.t, C_out)
    plot_salinity(sol.t, S_out)

    print('\n[5/5] Generating sink/source analysis...')
    plot_sink_source(sol.t, C_out, S_out)

    print('\nDone.')

if __name__ == '__main__':
    main()
