#!/usr/bin/env python3
"""
Five-Box Ocean Salinity ODE Simulation
=======================================
Based on: Sunny et al., Chaos 33, 103134 (2023)

Extension: Non-autonomous ODE integration using time-varying freshwater
forcing F_i(t) from CM SAF HOAPS satellite data (1987-2014) instead of
the original model's constant year-2000 values.

Box labelling (0-indexed throughout):
    0 = North Atlantic (NA)
    1 = South Atlantic (SA)
    2 = Southern Ocean (SO)
    3 = Pacific Ocean  (PO)
    4 = Indian Ocean   (IO)

Sign convention for freshwater fluxes:
    Positive F_i = net evaporation (E > P) → ocean loses water → S_i increases.
    Consistent with HOAPS 'budg' variable and paper's Table I values.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for headless environments
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
# PHYSICAL CONSTANTS
# ============================================================
SEC_PER_YEAR = 365.25 * 24.0 * 3600.0   # seconds per year
SV_TO_M3S    = 1.0e6                      # 1 Sv = 1e6 m³/s

# ============================================================
# MODEL PARAMETERS  (Table I & Appendix A, Sunny et al. 2023)
# ============================================================

BASIN_NAMES = ['North Atlantic', 'South Atlantic', 'Southern Ocean',
               'Pacific Ocean', 'Indian Ocean']
BASIN_CODES = ['NA', 'SA', 'SO', 'PO', 'IO']

# Reference salinity [psu]
S0_REF = 35.0

# Haline contraction coefficient [psu⁻¹]
BETA = 7.5e-4

# Hydraulic conductivity for deep thermohaline currents.
# Paper (line 265): "In calculating qij we use K_ij ≈ 60 Sv, β = 7.5×10⁻⁴ psu⁻¹"
# Formula (Eq 3): q_ij = K_ij * β * (S_i − S_j)   [Sv]
K_DEEP = 60.0   # [Sv]


# Box volumes [m³]
V = np.array([
    146.0e15,   # 0: North Atlantic
    160.0e15,   # 1: South Atlantic
     71.8e15,   # 2: Southern Ocean
    710.0e15,   # 3: Pacific Ocean
    264.0e15,   # 4: Indian Ocean
])

# Initial salinities [psu] – Table I of the paper
S_INIT = np.array([34.912, 35.435, 34.427, 34.668, 34.538])

# ---- Extra freshwater boundary terms (converted to Sv) ----
#
# Paper (lines 276-283): F_i = P - E (net precipitation from atmosphere to ocean).
# HOAPS data: F_hoaps = E - P  →  F_paper = -F_hoaps.
#
# b_NA > 0: Arctic ice melt adds freshwater to NA (positive P-E contribution to NA)
b_NA   =  3.28e3 / SV_TO_M3S    # [Sv]  (P-E convention, positive = inflow)

# b_SO < 0: Antarctic ice FORMATION removes freshwater from SO (net outflow)
b_SO   = -6.944e2 / SV_TO_M3S   # [Sv]  (negative = outflow from SO)

# Amazon discharge: freshwater inflow to NA (positive P-E contribution to NA)
q_AMAZ = 0.24                    # [Sv]

# ============================================================
# SURFACE CURRENTS  (Figure 2, red solid arrows – fixed)
# Format: (source_idx, dest_idx, magnitude_Sv)
# ============================================================
SURFACE_CURRENTS = [
    (1, 0, 26.0),   # SA → NA   Q_21
    (2, 1, 30.0),   # SO → SA   Q_32
    (3, 4, 15.0),   # PO → IO   Q_45
    (4, 1, 10.0),   # IO → SA   Q_52
    (4, 2, 60.0),   # IO → SO   Q_53
]

# ============================================================
# DEEP THERMOHALINE CURRENT ADJACENCY LIST (Figure 2, blue dashed)
# Flow direction is dynamic (sign of q_ij evolves with salinity).
# Format: (i, j) → q_ij = K_DEEP * BETA * (S[i] - S[j])  [Sv]
#   q > 0 → flow from i to j     q < 0 → flow from j to i
# ============================================================
DEEP_PAIRS = [
    (0, 1),   # NA ↔ SA  (q_12)
    (1, 2),   # SA ↔ SO  (q_23)
    (1, 3),   # SA ↔ PO  (q_24)
    (2, 3),   # SO ↔ PO  (q_34)
    (2, 4),   # SO ↔ IO  (q_35)
]

# ============================================================
# HELPER: DECIMAL-YEAR CONVERSION
# ============================================================

def timestamps_to_decimal_years(timestamps):
    result = []
    for ts in pd.DatetimeIndex(timestamps):
        y = ts.year
        start = pd.Timestamp(f'{y}-01-01')
        end   = pd.Timestamp(f'{y + 1}-01-01')
        frac  = (ts - start).total_seconds() / (end - start).total_seconds()
        result.append(y + frac)
    return np.array(result)


def decimal_year_to_timestamp(dy):
    y    = int(dy)
    frac = dy - y
    start = pd.Timestamp(f'{y}-01-01')
    end   = pd.Timestamp(f'{y + 1}-01-01')
    return start + pd.Timedelta(seconds=frac * (end - start).total_seconds())

# ============================================================
# LOAD & INTERPOLATE FRESHWATER FORCING
# ============================================================

def load_forcing(csv_path):
    """
    Returns
    -------
    splines : list of 5 CubicSpline objects   F_i(t) [Sv, E-P convention]
    t_data  : ndarray   decimal years of data points
    F_data  : ndarray (N, 5)  raw flux values [Sv]
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    t_data = timestamps_to_decimal_years(df.index)
    col_order = ['F_NA', 'F_SA', 'F_SO', 'F_PO', 'F_IO']
    F_data = df[col_order].values          # shape (330, 5)

    # CubicSpline with boundary-value extrapolation clamped (not=True keeps last slope)
    splines = [CubicSpline(t_data, F_data[:, i], extrapolate=True) for i in range(5)]
    return splines, t_data, F_data

# ============================================================
# SALINITY ODE
# ============================================================

def salinity_ode(t, S, F_splines):
    """
    Right-hand side of the five-box salinity equation.

    Parameters
    ----------
    t         : float     current time [decimal year]
    S         : array(5)  salinity of each box [psu]
    F_splines : list      F_i(t) interpolants [Sv]

    Returns
    -------
    dS : array(5)   rate of change [psu / year]
    """
    dS = np.zeros(5)

    # ----------------------------------------------------------
    # 1. Atmospheric freshwater flux  [psu/year]  — Paper Eq. (4)
    #
    #    Paper ODE: dS/dt = -(F_paper + b_NA + b_SO + q_Amaz)*S0/V + transport
    #    Paper defines F_paper = P-E (positive = net precipitation = ocean freshens)
    #
    #    HOAPS delivers E-P  →  F_paper = -F_hoaps
    #
    #    Substituting: -(−F_hoaps + b_NA + b_SO + q_Amaz)*S0/V
    #                = +(F_hoaps − b_NA − b_SO − q_Amaz)*S0/V
    #
    #    b_NA > 0: Arctic melt flows INTO NA (P-E positive) → freshens NA
    #    b_SO < 0: Antarctic ice formation takes water FROM SO (P-E negative)
    #             → b_SO term in -(... + b_SO ...) → -(negative) = +term = saltier SO
    #    q_Amaz > 0: Amazon freshwater INTO NA (P-E positive) → freshens NA
    # ----------------------------------------------------------
    F_hoaps = np.array([float(sp(t)) for sp in F_splines])  # E-P [Sv]

    # Build the combined freshwater flux in P-E convention:
    # F_paper_i = -F_hoaps_i (negate) then add the extra terms
    # Final ODE term: -(F_paper + ...) * S0/V = (F_hoaps - b_NA - q_Amaz_NA - b_SO_SO) * S0/V
    fw_effective = -F_hoaps.copy()          # start with P-E: positive = inflow = freshens
    fw_effective[0] += b_NA + q_AMAZ        # NA: add ice melt + Amazon inflow (both positive P-E)
    fw_effective[2] += b_SO                 # SO: add ice term (b_SO < 0 = ice formation = saltier)

    for i in range(5):
        # ODE: -fw_effective * S0/V  (the minus sign from Eq.4)
        dS[i] -= fw_effective[i] * SV_TO_M3S * S0_REF / V[i] * SEC_PER_YEAR

    # ----------------------------------------------------------
    # 2. Surface currents (fixed direction) — difference form
    #    Net contribution of Q_ji (src→dst) to each box:
    #      dst gains:  +Q * (S_src - S_dst) / V_dst  [net exchange]
    #      src loses:  -Q * (S_src - S_src) / V_src = 0  [no net at src? No...]
    #
    #    Standard Stommel form: dS_i/dt includes +Q_ji*S_j/V_i and -Q_ij*S_i/V_i
    #    For volume conservation (inflow = outflow per box), this reduces to:
    #      dS_i/dt = sum_j Q_ji * (S_j - S_i) / V_i  (exchange form)
    #
    #    This form is numerically stable and prevents drift from volume imbalance.
    # ----------------------------------------------------------

    # First, build net surface flow into each box [Sv] and salinity flux
    for (src, dst, Q_sv) in SURFACE_CURRENTS:
        Q = Q_sv * SV_TO_M3S                               # [m³/s]
        delta_S = S[src] - S[dst]                          # salinity difference
        dS[dst] += Q * delta_S / V[dst] * SEC_PER_YEAR    # dst receives salt deficit/excess

    # ----------------------------------------------------------
    # 3. Deep thermohaline currents (dynamic direction) — difference form
    #    q_ij = K·β·(S_i - S_j); positive → flow from i to j
    #    Net salinity exchange between pair (i,j):
    #      Box i: -q_ij * (S_i - S_j) / V_i  = -q²/(K·β·V_i)  (always removes gradient)
    #      Box j: +q_ij * (S_i - S_j) / V_j
    # ----------------------------------------------------------
    for (i, j) in DEEP_PAIRS:
        q_sv = K_DEEP * BETA * (S[i] - S[j])    # [Sv]; Eq.(3): q = K·β·ΔS
        q    = q_sv * SV_TO_M3S                   # [m³/s]
        delta_S = S[i] - S[j]                     # salinity difference

        dS[i] -= q * delta_S / V[i] * SEC_PER_YEAR
        dS[j] += q * delta_S / V[j] * SEC_PER_YEAR

    return dS

# ============================================================
# PLOTTING
# ============================================================

def plot_salinity(t_eval, S_out, t_data, F_data, timestamps):
    """Generate the main 5-panel salinity time-series figure."""
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

    dt_eval = pd.DatetimeIndex([decimal_year_to_timestamp(t) for t in t_eval])

    for i, (ax, name, code, color) in enumerate(
        zip(axes, BASIN_NAMES, BASIN_CODES, colors)
    ):
        ax.plot(dt_eval, S_out[:, i], color=color, linewidth=1.5,
                label=f'Simulated $S_{{\\mathrm{{{code}}}}}(t)$')
        ax.axhline(S_INIT[i], color='black', linestyle='--', linewidth=0.9,
                   alpha=0.6, label=f'Initial value ({S_INIT[i]:.3f} psu)')

        final_val = S_out[-1, i]
        delta     = final_val - S_INIT[i]
        sign      = '+' if delta >= 0 else ''
        ax.set_ylabel('Salinity (psu)', fontsize=10)
        ax.set_title(
            f'{name}  —  Δ = {sign}{delta:.4f} psu  '
            f'(init={S_INIT[i]:.3f} → final={final_val:.3f} psu)',
            fontsize=10, fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator(3))

    axes[-1].set_xlabel('Year', fontsize=11)
    fig.suptitle(
        'Five-Box Ocean Salinity Evolution (1987–2014)\n'
        'Time-Varying HOAPS Freshwater Forcing   |   Sunny et al. (2023) Extended',
        fontsize=13, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.975])

    for path in [
        os.path.join(OUTPUT_DIR, 'salinity_simulation_1987_2014.png'),
        os.path.join(ARTIFACT_DIR, 'salinity_simulation_1987_2014.png'),
    ]:
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {path}")
    plt.close()


def plot_deep_flows(t_eval, S_out):
    """Plot the deep thermohaline flow rates q_ij(t)."""
    colors_q = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    labels_q  = ['$q_{12}$ (NA–SA)', '$q_{23}$ (SA–SO)',
                  '$q_{24}$ (SA–PO)', '$q_{34}$ (SO–PO)', '$q_{35}$ (SO–IO)']

    q_vals = np.zeros((len(t_eval), 5))
    for k, (i, j) in enumerate(DEEP_PAIRS):
        q_vals[:, k] = K_DEEP * BETA * (S_out[:, i] - S_out[:, j])

    dt_eval = pd.DatetimeIndex([decimal_year_to_timestamp(t) for t in t_eval])

    fig, ax = plt.subplots(figsize=(14, 5))
    for k in range(5):
        ax.plot(dt_eval, q_vals[:, k], linewidth=1.4,
                color=colors_q[k], label=labels_q[k])
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax.set_ylabel('Flow rate (Sv)', fontsize=11)
    ax.set_xlabel('Year', fontsize=11)
    ax.set_title('Deep Thermohaline Flow Rates $q_{ij}(t)$ (1987–2014)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    plt.tight_layout()

    for path in [
        os.path.join(OUTPUT_DIR, 'deep_flow_rates_1987_2014.png'),
        os.path.join(ARTIFACT_DIR, 'deep_flow_rates_1987_2014.png'),
    ]:
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved → {path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Five-Box Ocean Salinity ODE Simulation")
    print("Sunny et al. (2023) — Extended with HOAPS 1987-2014")
    print("=" * 60)

    # ---- Load forcing ----
    print("\n[1/4] Loading freshwater forcing data...")
    F_splines, t_data, F_data = load_forcing(CSV_PATH)
    t_start, t_end = t_data[0], t_data[-1]
    print(f"      Data range: {t_start:.3f} – {t_end:.3f} decimal years")
    print(f"      Data points: {len(t_data)} months")

    # ---- Integrate ----
    print("\n[2/4] Integrating salinity ODE (RK45, tol=1e-8)...")
    t_eval = np.linspace(t_start, t_end, 5000)   # ~15 pts/month

    sol = solve_ivp(
        fun     = lambda t, S: salinity_ode(t, S, F_splines),
        t_span  = (t_start, t_end),
        y0      = S_INIT,
        method  = 'RK45',
        t_eval  = t_eval,
        rtol    = 1e-8,
        atol    = 1e-10,
    )

    if not sol.success:
        print(f"  WARNING: ODE solver issue — {sol.message}")
    else:
        print(f"  Integration successful ({sol.nfev} evaluations, {len(sol.t)} output pts)")

    S_out = sol.y.T    # shape (n_eval, 5)

    # ---- Save CSV ----
    print("\n[3/4] Saving output CSV...")
    timestamps = [decimal_year_to_timestamp(t) for t in sol.t]
    df_out = pd.DataFrame(
        S_out,
        index   = pd.DatetimeIndex(timestamps, name='date'),
        columns = ['S_NA', 'S_SA', 'S_SO', 'S_PO', 'S_IO']
    )
    csv_out = os.path.join(OUTPUT_DIR, 'salinity_simulation_1987_2014.csv')
    df_out.to_csv(csv_out)
    print(f"  Saved → {csv_out}")

    # ---- Print summary statistics ----
    print("\n" + "=" * 60)
    print("SALINITY EVOLUTION SUMMARY")
    print("=" * 60)
    hdr = f"{'Basin':<22} {'Init (psu)':<14} {'Final (psu)':<14} {'Δ (psu)':<12} {'Δ/yr (mpsu)'}"
    print(hdr)
    print("-" * 74)
    n_years = t_end - t_start
    for i, (name, code) in enumerate(zip(BASIN_NAMES, BASIN_CODES)):
        s0 = S_INIT[i]
        sf = S_out[-1, i]
        ds = sf - s0
        ds_yr = ds / n_years * 1000   # milli-psu per year
        print(f"  {name:<20} {s0:<14.4f} {sf:<14.4f} {ds:<12.5f} {ds_yr:+.3f}")

    print("\n[4/4] Generating plots...")
    plot_salinity(sol.t, S_out, t_data, F_data, timestamps)
    plot_deep_flows(sol.t, S_out)

    print("\nDone.")

if __name__ == '__main__':
    main()
