"""
Validation and Improvement of the SST-to-ODE Pipeline
Five-Ocean Carbon Sink Box Model based on Sunny et al. (2023)

Runs three diagnostic SST cases through the 15-variable coupled stiff ODE:
  - Case A: Legacy / Approximate masks (280-360 Atlantic, 50S Southern Ocean)
  - Case B: Paper-boundary masks + newly fitted SST linear equations
  - Case C: Paper-boundary masks + exact paper Appendix D SST equations

Reproduces and validates against Figure 4 and Figure 5 in Sunny et al. (2023).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==============================================================================
# 1. DIRECTORY CONFIGURATION
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "sst_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECONDS_PER_MONTH = 365.25 * 86400.0 / 12.0  # 2,629,800 seconds
TOTAL_MONTHS = 228  # Jan 2000 to Dec 2018
SV_TO_M3_S = 1.0e6   # 1 Sverdrup = 10^6 m^3/s

OCEANS = ["NA", "SA", "SO", "PO", "IO"]
OCEAN_NAMES = {
    "NA": "North Atlantic",
    "SA": "South Atlantic",
    "SO": "Southern Ocean",
    "PO": "Pacific Ocean",
    "IO": "Indian Ocean"
}
OCEAN_COLORS = {
    "NA": "#1f77b4",  # blue
    "SA": "#ff7f0e",  # orange
    "SO": "#2ca02c",  # green
    "PO": "#d62728",  # red
    "IO": "#9467bd"   # purple
}

# ==============================================================================
# 2. PHYSICAL PARAMETERS (Sunny et al. 2023, Appendices A, B, F, Table I & II)
# ==============================================================================
# Index mapping: 0: NA (1), 1: SA (2), 2: SO (3), 3: PO (4), 4: IO (5)
INDEX_MAP = {"NA": 0, "SA": 1, "SO": 2, "PO": 3, "IO": 4}

# Volumes [m^3] (Appendix A)
VOLUMES = {
    "NA": 146.0e15,
    "SA": 160.0e15,
    "SO": 71.8e15,
    "PO": 710.0e15,
    "IO": 264.0e15
}

# Surface Areas [m^2] (Appendix A)
AREAS = {
    "NA": 41.49e12,
    "SA": 40.27e12,
    "SO": 20.33e12,
    "PO": 165.25e12,
    "IO": 70.56e12
}

# Initial conditions for Jan 2000 (Table I)
# C: mol/m^3, S: psu
INITIAL_C = {
    "NA": 0.01413,
    "SA": 0.01579,
    "SO": 0.01400,
    "PO": 0.01834,
    "IO": 0.01671
}

INITIAL_S = {
    "NA": 34.912,
    "SA": 35.435,
    "SO": 34.427,
    "PO": 34.668,
    "IO": 34.538
}

K_EQ = 0.0011  # Hydration equilibrium constant for initial N

# Atmospheric Keeling curve linear fit: pCO2(t) in uatm, t in months (Section II F)
PCO2_SLOPE = 0.18152
PCO2_INTERCEPT = 367.41

# Reaction rate constants (Table II) [1/s]
K_C = 0.037
K_N = 23.0

# Gas transfer velocity base parameterization (Appendix B)
U10 = 6.0  # m/s
GAMMA_660_CM_YR = 0.24 * (U10 ** 2)  # 8.64 cm/yr
GAMMA_660_M_S = (GAMMA_660_CM_YR * 0.01) / (365.25 * 86400.0)  # ~2.738e-9 m/s

# Weiss solubility constants (Appendix B, Eq. B2)
A1 = -58.0931
A2 = 90.5069
A3 = 22.2940
B1 = 0.027766
B2 = -0.025888
B3 = 0.0050578

# Physical coefficients (Table II)
ALPHA_THERMAL = 2.07e-4  # 1/°C
BETA_HALINE = 7.5e-4     # 1/psu
RHO_0 = 1029.0           # kg/m^3
K_IJ = 60.0 * SV_TO_M3_S # 60 Sv in m^3/s
S_REF = 35.0             # Reference salinity (psu)

# Surface Currents Q_ij [m^3/s] (Table II)
# Q45 (PO->IO)=15 Sv, Q21 (SA->NA)=26 Sv, Q53 (IO->SO)=60 Sv, Q52 (IO->SA)=10 Sv, Q32 (SO->SA)=30 Sv
Q_SURFACE = {
    ("PO", "IO"): 15.0 * SV_TO_M3_S,
    ("SA", "NA"): 26.0 * SV_TO_M3_S,
    ("IO", "SO"): 60.0 * SV_TO_M3_S,
    ("IO", "SA"): 10.0 * SV_TO_M3_S,
    ("SO", "SA"): 30.0 * SV_TO_M3_S,
}

# Connected deep-water ocean pairs (undirected topology from Fig. 2)
DEEP_PAIRS = [
    ("NA", "SA"),
    ("SA", "SO"),
    ("SO", "PO"),
    ("SO", "IO"),
    ("PO", "IO"),
    ("SA", "IO"),
]

# Atmospheric Freshwater Flux F_i [m^3/s] (Table II)
FRESHWATER_F = {
    "NA": 0.475 * SV_TO_M3_S,
    "SA": 0.526 * SV_TO_M3_S,
    "SO": -0.209 * SV_TO_M3_S,
    "PO": 0.064 * SV_TO_M3_S,
    "IO": 2.3495 * SV_TO_M3_S,
}

# Polar ice melt flux b_i [m^3/s] (Table II)
ICE_MELT_B = {
    "NA": 3.28e3,
    "SA": 0.0,
    "SO": -6.944e2,
    "PO": 0.0,
    "IO": 0.0,
}

# Amazon River freshwater flux [m^3/s] (Table II)
Q_AMAZON = 0.24 * SV_TO_M3_S

# Constants for deep water CO2 correction C'_i (Eq. 10 & Table II)
A_CONST = 3.63579e-5
B_CONST = 4.477820e-5
G_CONST = 1.8833e-5
L_CONST = 7.31010e-5
H_CONST = 5.41469e-5
PSI_CONST = 7.49356e-5
M_BAR = 18.0  # g/mol average molar mass of water

# ==============================================================================
# 3. SST EQUATIONS FOR THE 3 CASES
# ==============================================================================
# Case C: Exact Paper Appendix D Equations
PAPER_SST = {
    "NA": {"slope": 0.00170, "intercept": 291.90},
    "SA": {"slope": -0.00015, "intercept": 290.30},
    "SO": {"slope": -0.00056, "intercept": 272.29},
    "PO": {"slope": 0.00155, "intercept": 292.88},
    "IO": {"slope": 0.00026, "intercept": 291.57},
}

# Case B: Corrected paper boundaries + our fitted SST (from sst_trend_comparison.csv)
FITTED_SST_PAPER_MASKS = {
    "NA": {"slope": 0.00151986, "intercept": 286.8443},
    "SA": {"slope": 6.9188e-7,  "intercept": 288.3090},
    "SO": {"slope": -0.00018704, "intercept": 272.2622},
    "PO": {"slope": 0.00032146, "intercept": 291.5633},
    "IO": {"slope": 0.00054785, "intercept": 289.9148},
}

# Case A: Legacy approximate masks SST fits (computed from sst_previous_masks.csv)
LEGACY_SST_MASKS = {
    "NA": {"slope": 0.001620, "intercept": 287.120},
    "SA": {"slope": 0.000105, "intercept": 288.750},
    "SO": {"slope": -0.000310, "intercept": 274.850},
    "PO": {"slope": 0.000450, "intercept": 291.800},
    "IO": {"slope": 0.000610, "intercept": 290.200},
}


def get_sst_kelvin(ocean: str, t_months: float, case: str) -> float:
    """Return SST in Kelvin at time t (months from Jan 2000)."""
    if case == "A":
        p = LEGACY_SST_MASKS[ocean]
    elif case == "B":
        p = FITTED_SST_PAPER_MASKS[ocean]
    elif case == "C":
        p = PAPER_SST[ocean]
    else:
        raise ValueError(f"Unknown case: {case}")
    return p["slope"] * t_months + p["intercept"]


def compute_schmidt_number(t_celsius: float) -> float:
    """Schmidt number for CO2 in seawater (Appendix B, Eq. B1)."""
    T = t_celsius
    return 2116.8 - 136.5 * T + 4.7353 * (T**2) - 0.092307 * (T**3) + 0.0007555 * (T**4)


def compute_solubility_k0(t_kelvin: float, salinity: float) -> float:
    """
    Weiss (1974) CO2 solubility K0 in mol/(L*atm) (Appendix B, Eq. B2).
    """
    T_100 = t_kelvin / 100.0
    ln_k0 = (
        A1
        + A2 * (1.0 / T_100)
        + A3 * np.log(T_100)
        + salinity * (B1 + B2 * T_100 + B3 * (T_100**2))
    )
    return float(np.exp(ln_k0))


def compute_density(t_celsius: float, salinity: float) -> float:
    """Water density rho in kg/m^3 (Eq. 1)."""
    return RHO_0 * (1.0 - ALPHA_THERMAL * t_celsius + BETA_HALINE * salinity)


def compute_c_prime(c_i: float, t_celsius: float, salinity: float) -> float:
    """Deep-flow CO2 corrected concentration C'_i (Eq. 10)."""
    rho = compute_density(t_celsius, salinity)
    t = t_celsius
    c_corr = (
        c_i * (1.0 + A_CONST - B_CONST * t)
        + (G_CONST * t + L_CONST) * (M_BAR / rho) * c_i
        + (H_CONST * t - PSI_CONST) * ((M_BAR / rho) ** 2) * (c_i**2)
    )
    return c_corr


# ==============================================================================
# 4. COUPLED 15-VARIABLE ODE RHS FUNCTION
# ==============================================================================
def box_model_rhs(t_sec: float, y: np.ndarray, sst_case: str) -> np.ndarray:
    """
    Computes derivatives [dC/dt, dN/dt, dS/dt] for all 5 oceans (15 state variables).
    t_sec: time in seconds.
    y: array of shape (15,) containing [C_1..5, N_1..5, S_1..5].
    """
    t_months = t_sec / SECONDS_PER_MONTH

    # Atmospheric pCO2 in uatm
    pco2_atm_uatm = PCO2_SLOPE * t_months + PCO2_INTERCEPT

    C = y[0:5]
    N = y[5:10]
    S = y[10:15]

    dC_dt = np.zeros(5)
    dN_dt = np.zeros(5)
    dS_dt = np.zeros(5)

    # 1. Compute temperatures, Schmidt numbers, K0, and air-sea equilibrium for each ocean
    T_K = np.zeros(5)
    T_C = np.zeros(5)
    gamma = np.zeros(5)
    K0 = np.zeros(5)
    C_star = np.zeros(5)

    for i, ocean in enumerate(OCEANS):
        t_k = get_sst_kelvin(ocean, t_months, sst_case)
        t_c = t_k - 273.15
        T_K[i] = t_k
        T_C[i] = t_c

        sc = compute_schmidt_number(t_c)
        sc = max(sc, 10.0)  # Numerical floor
        gamma[i] = GAMMA_660_M_S * ((sc / 660.0) ** -0.5)

        k0_val = compute_solubility_k0(t_k, S[i])
        K0[i] = k0_val

        # C_star in mol/m^3: K0 [mol/(L*atm)] * 1000 [L/m^3] * (pCO2 [uatm] * 1e-6 [atm/uatm])
        C_star[i] = k0_val * 1000.0 * (pco2_atm_uatm * 1e-6)

    # 2. Deep-water flows q_ij based on salinity differences
    q_flows = {}
    for o1, o2 in DEEP_PAIRS:
        i1, i2 = INDEX_MAP[o1], INDEX_MAP[o2]
        s1, s2 = S[i1], S[i2]
        if s1 > s2:
            q_flows[(o1, o2)] = K_IJ * BETA_HALINE * (s1 - s2)
            q_flows[(o2, o1)] = 0.0
        else:
            q_flows[(o1, o2)] = 0.0
            q_flows[(o2, o1)] = K_IJ * BETA_HALINE * (s2 - s1)

    # Deep flow corrected CO2 C'_i
    C_prime = np.zeros(5)
    for i, ocean in enumerate(OCEANS):
        C_prime[i] = compute_c_prime(C[i], T_C[i], S[i])

    # 3. Assemble ODEs for each basin
    for i, ocean in enumerate(OCEANS):
        vol = VOLUMES[ocean]
        area = AREAS[ocean]

        # --- A. Air-Sea Exchange ---
        flux_air_sea = (gamma[i] * area / vol) * (C_star[i] - C[i])

        # --- B. Chemical Kinetics ---
        chem_c = -K_C * C[i] + K_N * N[i]
        chem_n = K_C * C[i] - K_N * N[i]

        # --- C. Surface Currents Inflow / Outflow ---
        surf_c_in, surf_c_out = 0.0, 0.0
        surf_n_in, surf_n_out = 0.0, 0.0
        surf_s_in, surf_s_out = 0.0, 0.0

        for (src, dst), q_s in Q_SURFACE.items():
            s_idx, d_idx = INDEX_MAP[src], INDEX_MAP[dst]
            if dst == ocean:
                surf_c_in += (q_s / vol) * C[s_idx]
                surf_n_in += (q_s / vol) * N[s_idx]
                surf_s_in += (q_s / vol) * S[s_idx]
            if src == ocean:
                surf_c_out += (q_s / vol) * C[s_idx]
                surf_n_out += (q_s / vol) * N[s_idx]
                surf_s_out += (q_s / vol) * S[s_idx]

        # --- D. Deep Ocean Flows Inflow / Outflow ---
        deep_c_in, deep_c_out = 0.0, 0.0
        deep_n_in, deep_n_out = 0.0, 0.0
        deep_s_in, deep_s_out = 0.0, 0.0

        for (o1, o2) in DEEP_PAIRS:
            q_12 = q_flows[(o1, o2)]
            q_21 = q_flows[(o2, o1)]

            idx1, idx2 = INDEX_MAP[o1], INDEX_MAP[o2]

            if o2 == ocean and q_12 > 0:
                deep_c_in += (q_12 / vol) * C_prime[idx1]
                deep_n_in += (q_12 / vol) * N[idx1]
                deep_s_in += (q_12 / vol) * S[idx1]
            if o1 == ocean and q_12 > 0:
                deep_c_out += (q_12 / vol) * C_prime[idx1]
                deep_n_out += (q_12 / vol) * N[idx1]
                deep_s_out += (q_12 / vol) * S[idx1]

            if o1 == ocean and q_21 > 0:
                deep_c_in += (q_21 / vol) * C_prime[idx2]
                deep_n_in += (q_21 / vol) * N[idx2]
                deep_s_in += (q_21 / vol) * S[idx2]
            if o2 == ocean and q_21 > 0:
                deep_c_out += (q_21 / vol) * C_prime[idx2]
                deep_n_out += (q_21 / vol) * N[idx2]
                deep_s_out += (q_21 / vol) * S[idx2]

        # --- E. Freshwater & Fluvial Fluxes (Salinity & Amazon) ---
        f_i = FRESHWATER_F[ocean]
        b_i = ICE_MELT_B[ocean]
        q_amaz = Q_AMAZON if ocean == "NA" else 0.0

        salinity_freshwater_loss = -((f_i + b_i + q_amaz) * S_REF) / vol

        # Rate of change equations (Eqs. 4, 7, 8)
        dC_dt[i] = flux_air_sea + chem_c + (surf_c_in - surf_c_out) + (deep_c_in - deep_c_out)
        dN_dt[i] = chem_n + (surf_n_in - surf_n_out) + (deep_n_in - deep_n_out)
        dS_dt[i] = salinity_freshwater_loss + (surf_s_in - surf_s_out) + (deep_s_in - deep_s_out)

    return np.concatenate([dC_dt, dN_dt, dS_dt])


# ==============================================================================
# 5. SIMULATION RUNNER
# ==============================================================================
def run_simulation(case_name: str) -> dict:
    """Integrate the 15-variable box model ODE for 228 months."""
    print(f"--> Running simulation for Case {case_name}...")

    # Initial condition vector: [C_0..4, N_0..4, S_0..4]
    c0 = np.array([INITIAL_C[o] for o in OCEANS])
    n0 = c0 * K_EQ
    s0 = np.array([INITIAL_S[o] for o in OCEANS])
    y0 = np.concatenate([c0, n0, s0])

    t_span = (0.0, TOTAL_MONTHS * SECONDS_PER_MONTH)
    t_eval = np.linspace(0.0, TOTAL_MONTHS * SECONDS_PER_MONTH, TOTAL_MONTHS + 1)

    sol = solve_ivp(
        fun=lambda t, y: box_model_rhs(t, y, case_name),
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        method="Radau",
        rtol=1e-6,
        atol=1e-10
    )

    if not sol.success:
        print(f"Warning: solver returned with message: {sol.message}")

    months = sol.t / SECONDS_PER_MONTH
    c_trajectories = {OCEANS[i]: sol.y[i, :] for i in range(5)}
    n_trajectories = {OCEANS[i]: sol.y[5 + i, :] for i in range(5)}
    s_trajectories = {OCEANS[i]: sol.y[10 + i, :] for i in range(5)}

    return {
        "case": case_name,
        "months": months,
        "C": c_trajectories,
        "N": n_trajectories,
        "S": s_trajectories,
    }


# ==============================================================================
# 6. VALIDATION & ANALYSIS
# ==============================================================================
def analyze_and_plot_results(res_a: dict, res_b: dict, res_c: dict):
    """Generate publication figures, metrics tables, and comparisons."""
    months = res_c["months"]
    t_years = 2000.0 + months / 12.0

    # -------------------------------------------------------------------------
    # 1. Plot Figure 4 Reproduction: 3-Case Comparison of C_i(t)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Left: Main Figure 4 Reproduction (All Basins for Case C vs Case B vs Case A)
    ax = axes[0]
    for ocean in OCEANS:
        col = OCEAN_COLORS[ocean]
        lbl_name = OCEAN_NAMES[ocean]
        ax.plot(t_years, res_c["C"][ocean], color=col, linestyle="-", linewidth=2.2, label=f"{lbl_name} (Paper SST)")
        ax.plot(t_years, res_b["C"][ocean], color=col, linestyle="--", linewidth=1.5, alpha=0.85)
        ax.plot(t_years, res_a["C"][ocean], color=col, linestyle=":", linewidth=1.2, alpha=0.7)

    ax.set_title("Figure 4 Reproduction: Evolution of Aqueous $CO_2$ ($C_i$) (2000–2018)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Aqueous $CO_2$ Concentration $C_i$ ($mol/m^3$)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # Right: Detailed Inset / Zoom on Lower Basins (NA, SA, PO, IO)
    ax_in = axes[1]
    for ocean in ["NA", "SA", "PO", "IO"]:
        col = OCEAN_COLORS[ocean]
        ax_in.plot(t_years, res_c["C"][ocean], color=col, linestyle="-", linewidth=2.0, label=f"{ocean} (Paper SST)")
        ax_in.plot(t_years, res_b["C"][ocean], color=col, linestyle="--", linewidth=1.4)
        ax_in.plot(t_years, res_a["C"][ocean], color=col, linestyle=":", linewidth=1.2)

    ax_in.set_title("Lower Basins Zoom (NA, SA, PO, IO)", fontsize=12, fontweight="bold")
    ax_in.set_xlabel("Year", fontsize=11)
    ax_in.set_ylabel("$C_i$ ($mol/m^3$)", fontsize=11)
    ax_in.grid(True, linestyle="--", alpha=0.4)
    ax_in.legend(loc="upper left", fontsize=9)

    fig.suptitle("Five-Ocean CO2 Trajectories Across SST Preprocessing Regimes\n"
                 "[Solid: Case C (Paper Reference) | Dashed: Case B (Corrected Masks+Fits) | Dotted: Case A (Legacy Baseline)]",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure4_co2_trajectory_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure4_co2_trajectory_validation.png")

    # -------------------------------------------------------------------------
    # 2. Plot Salinity Trajectories (Figure 5 Reproduction)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    for ocean in OCEANS:
        col = OCEAN_COLORS[ocean]
        lbl_name = OCEAN_NAMES[ocean]
        ax.plot(t_years, res_c["S"][ocean], color=col, linewidth=2.0, label=lbl_name)
    ax.set_title("Figure 5 Reproduction: Ocean Salinity Evolution (2000–2018)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Salinity $S_i$ (psu)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure5_salinity_evolution.png", dpi=300)
    plt.close(fig)
    print("Saved figure5_salinity_evolution.png")

    # -------------------------------------------------------------------------
    # 3. Discrepancy & Sensitivity Decomposition Plot
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Difference (Case B - Case C): Effect of SST regression slope & intercept error
    ax = axes[0]
    for ocean in OCEANS:
        diff_bc = (res_b["C"][ocean] - res_c["C"][ocean]) * 1000.0  # mmol/m^3
        ax.plot(t_years, diff_bc, color=OCEAN_COLORS[ocean], linewidth=1.8, label=ocean)
    ax.set_title("Discrepancy from SST Linear Fits\n[Case B (Fitted SST) − Case C (Paper SST)]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(r"$\Delta C_i$ ($mmol/m^3$)", fontsize=10)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)

    # Difference (Case A - Case C): Total effect of boundary masks + SST errors
    ax = axes[1]
    for ocean in OCEANS:
        diff_ac = (res_a["C"][ocean] - res_c["C"][ocean]) * 1000.0  # mmol/m^3
        ax.plot(t_years, diff_ac, color=OCEAN_COLORS[ocean], linewidth=1.8, label=ocean)
    ax.set_title("Total Downstream Error from Legacy Preprocessing\n[Case A (Legacy Masks) − Case C (Paper Ref)]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(r"$\Delta C_i$ ($mmol/m^3$)", fontsize=10)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sst_to_ode_error_decomposition.png", dpi=300)
    plt.close(fig)
    print("Saved sst_to_ode_error_decomposition.png")

    # -------------------------------------------------------------------------
    # 4. Generate Quantitative Summary Tables
    # -------------------------------------------------------------------------
    summary_rows = []
    for ocean in OCEANS:
        init_c = res_c["C"][ocean][0]
        final_c_c = res_c["C"][ocean][-1]
        final_c_b = res_b["C"][ocean][-1]
        final_c_a = res_a["C"][ocean][-1]

        delta_c_c = final_c_c - init_c
        delta_c_b = final_c_b - init_c
        delta_c_a = final_c_a - init_c

        mae_bc = np.mean(np.abs(res_b["C"][ocean] - res_c["C"][ocean]))
        rmse_bc = np.sqrt(np.mean((res_b["C"][ocean] - res_c["C"][ocean]) ** 2))
        max_diff_bc = np.max(np.abs(res_b["C"][ocean] - res_c["C"][ocean]))

        mae_ac = np.mean(np.abs(res_a["C"][ocean] - res_c["C"][ocean]))
        rmse_ac = np.sqrt(np.mean((res_a["C"][ocean] - res_c["C"][ocean]) ** 2))

        summary_rows.append({
            "Ocean": ocean,
            "Name": OCEAN_NAMES[ocean],
            "C_init (mol/m3)": init_c,
            "Case C Final (mol/m3)": final_c_c,
            "Case B Final (mol/m3)": final_c_b,
            "Case A Final (mol/m3)": final_c_a,
            "Case C Net Delta (mol/m3)": delta_c_c,
            "Case B Net Delta (mol/m3)": delta_c_b,
            "Case A Net Delta (mol/m3)": delta_c_a,
            "Error_BC_MAE (mol/m3)": mae_bc,
            "Error_BC_RMSE (mol/m3)": rmse_bc,
            "Error_AC_RMSE (mol/m3)": rmse_ac,
            "Error_BC_Final_Pct": 100.0 * (final_c_b - final_c_c) / final_c_c,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "figure4_validation_metrics.csv", index=False)
    print("Saved figure4_validation_metrics.csv")

    print("\n==================== FIGURE 4 VALIDATION METRICS ====================")
    print(summary_df[["Ocean", "C_init (mol/m3)", "Case C Final (mol/m3)", "Case B Final (mol/m3)", "Error_BC_RMSE (mol/m3)", "Error_BC_Final_Pct"]].to_string(index=False))

    return summary_df


def main():
    print("Starting SST -> ODE Validation Pipeline...")
    res_a = run_simulation("A")
    res_b = run_simulation("B")
    res_c = run_simulation("C")

    summary_df = analyze_and_plot_results(res_a, res_b, res_c)
    print("\nValidation complete! All outputs generated successfully in sst_outputs/.")


if __name__ == "__main__":
    main()

