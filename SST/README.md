# Five-Ocean Carbon Sink Box Model: Reproduction & Validation

Computational reproduction, validation, and sensitivity analysis of the five-ocean conceptual carbon sink box model introduced by **Sunny et al. (2023)**.

> **Reference Paper:**  
> Sunny, E. M., Ashok, B., Balakrishnan, J., & Kurths, J. (2023). *The ocean carbon sinks and climate change.* **Chaos: An Interdisciplinary Journal of Nonlinear Science**, 33(10), 103134. [doi:10.1063/5.0164196](https://doi.org/10.1063/5.0164196)

---

## 📌 Project Overview

This repository implements the end-to-end physics pipeline of the five-ocean box model:
1. **Sea Surface Temperature (SST) Preprocessing**: Reconstructs five-ocean spatial masks from NOAA COBE-2 monthly gridded SST data ($1^\circ \times 1^\circ$, Jan 2000 – Dec 2018) using the paper's exact geographic boundaries ($70^\circ\text{W}$, $20^\circ\text{E}$, $140^\circ\text{E}$, $60^\circ\text{S}$, and Atlantic wrap).
2. **15-Variable Coupled ODE Solver**: Integrates the coupled non-linear differential equations governing dissolved aqueous $\text{CO}_2$ ($C_i$), reaction intermediate products ($N_i$), and salinity ($S_i$) across the five major ocean basins:
   - **NA**: North Atlantic
   - **SA**: South Atlantic
   - **SO**: Southern Ocean
   - **PO**: Pacific Ocean
   - **IO**: Indian Ocean
3. **Physics Validation (Figure 4 & 5 Reproduction)**: Runs a 3-case diagnostic simulation to isolate error propagation from SST preprocessing vs. ODE dynamics and verifies the physical dominance of the Southern Ocean as the primary global carbon sink.

---

## 🌐 Datasets & Data Sources

The model and preprocessing rely on publicly available climate and oceanographic datasets:

### 1. Sea Surface Temperature (SST) — COBE-2
- **Description:** Monthly mean Sea Surface Temperature on a $1.0^\circ \times 1.0^\circ$ global grid (1850–present; January 2000 – December 2018 used).
- **Source:** NOAA Physical Sciences Laboratory (PSL), Boulder, Colorado, USA.
- **Dataset Page:** [NOAA PSL COBE-2 SST](https://psl.noaa.gov/data/gridded/data.cobe2.html)
- **Direct Download (`sst.mon.mean.nc`):**
  ```bash
  wget https://downloads.psl.noaa.gov/Datasets/COBE2/sst.mon.mean.nc
  ```
- **Kaggle Mirror:** [cobe2-sst-dataset](https://www.kaggle.com/datasets/aryanvaghasiya/cobe2-sst-dataset)
- **Citation:** Ishii, M., et al. (2005). *Objective Analyses of Sea-Surface Temperature and Marine Meteorological Variables for the 20th Century using ICOADS and the Kobe Collection.* Int. J. Climatol., 25: 865–879.

### 2. Atmospheric $\text{CO}_2$ — Mauna Loa Record (Keeling Curve)
- **Description:** Monthly mean atmospheric $\text{CO}_2$ mole fraction and partial pressure ($p\text{CO}_2$).
- **Source:** NOAA Global Monitoring Laboratory (GML) & Scripps Institution of Oceanography.
- **Data Access:** [NOAA GML Carbon Cycle Trends](https://gml.noaa.gov/ccgg/trends/) | [Scripps $\text{CO}_2$ Data](https://scrippsco2.ucsd.edu/data/atmospheric_co2/mlo.html)
- **Model Representation:** Parameterized via the linear fit $p\text{CO}_2(t) = 0.18152 t + 367.41\text{ }\mu\text{atm}$ ($t$ in months from Jan 2000, Sunny et al. 2023 Sec. II F).

### 3. Global Ocean Hydrography & Transport Constants
- **Initial Salinity & Aqueous $\text{CO}_2$:** Taken from **Table I** of Sunny et al. (2023), compiled from the Global Ocean Data Analysis Project ([GLODAPv2](https://www.ncei.noaa.gov/access/ocean-carbon-acidification-data-system/oceans/GLODAPv2/)) and World Ocean Atlas ([WOA](https://www.ncei.noaa.gov/products/world-ocean-atlas)).
- **Freshwater & River Discharges:** Fluvial input from the Amazon River ($0.24\text{ Sv}$) and net freshwater flux ($P - E$) from Fennig et al. (2012).

---

## 🏗️ Repository Structure

```text
├── README.md                          # Project overview and documentation
├── requirements.txt                   # Python environment dependencies
├── .gitignore                         # Git ignore definitions
├── docs/                              # Project reports and reference papers
│   ├── ba_chaos_X23.pdf               # Reference paper (Sunny et al. 2023)
│   ├── ocean_carbon_progress_report.pdf
│   ├── ocean_carbon_sink_implementation_report.pdf
│   ├── ocean_carbon_full_report.pdf
│   ├── SST_processedReport.pdf
│   ├── physics_vs_ml_ocean_prediction_report.pdf
│   └── sst_work_plan.md               # Workstream roadmap and methodology
├── src/                               # Core Python source code
│   ├── phy_sst_improved.py            # SST mask generation, basin averaging & trend fitting
│   └── run_validation_pipeline.py     # 15-variable stiff ODE solver & 3-case validation
└── sst_outputs/                       # Generated data tables, plots & validation reports
    ├── README_SST_VALIDATION.md       # Detailed SST-to-ODE validation report
    ├── sst_monthly_paper_boundaries_C.csv
    ├── sst_monthly_paper_boundaries_K.csv
    ├── sst_previous_masks.csv
    ├── sst_trend_comparison.csv
    ├── sst_actual_vs_trend_paper_boundaries.png
    ├── sst_basin_timeseries.png
    ├── figure4_co2_trajectory_validation.png
    ├── figure4_validation_metrics.csv
    ├── figure5_salinity_evolution.png
    └── sst_to_ode_error_decomposition.png
```

---

## ⚙️ Installation & Requirements

Ensure you have Python 3.10+ installed:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Run SST Preprocessing & Trend Fitting
Processes the gridded SST netCDF dataset, applies the five-basin geographic masks, fits monthly linear equations $T_i(t) = a_i t + b_i$, and saves the results into `sst_outputs/`:

```bash
python src/phy_sst_improved.py
```

### 2. Run the 3-Case ODE Simulation & Validation
Integrates the 15-variable stiff ODE system using the implicit Radau solver and compares:
- **Case A**: Legacy approximate masks ($280^\circ - 360^\circ$ Atlantic, $-50^\circ\text{S}$ Southern Ocean)
- **Case B**: Corrected paper boundaries + newly fitted linear SST equations
- **Case C**: Corrected paper boundaries + exact Appendix D linear SST equations

```bash
python src/run_validation_pipeline.py
```

---

## 📊 Key Results

### 1. SST Trend Equations vs. Paper Appendix D

| Basin | Paper Slope [$\text{K/mo}$] | Fitted Slope [$\text{K/mo}$] | % Diff | Paper Intercept [$\text{K}$] | Fitted Intercept [$\text{K}$] | Residual RMSE [$\text{K}$] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **North Atlantic (NA)** | $+0.001700$ | $+0.001520$ | $-10.60\%$ | $291.90$ | $286.84$ | $1.65$ |
| **South Atlantic (SA)** | $-0.000150$ | $+0.0000007$ | $+100.46\%$ | $290.30$ | $288.31$ | $1.54$ |
| **Southern Ocean (SO)** | $-0.000560$ | $-0.000187$ | $+66.60\%$ | $272.29$ | $272.26$ | **$0.51$** |
| **Pacific Ocean (PO)** | $+0.001550$ | $+0.000321$ | $-79.26\%$ | $292.88$ | $291.56$ | $1.30$ |
| **Indian Ocean (IO)** | $+0.000260$ | $+0.000548$ | $+110.71\%$ | $291.57$ | $289.91$ | $1.18$ |

### 2. Aqueous $\text{CO}_2$ Trajectories (Figure 4 Reproduction)

| Ocean Basin | $C_{\text{initial}}$ [$\text{mol/m}^3$] | Case C Final [$\text{mol/m}^3$] | Case B Final [$\text{mol/m}^3$] | Net Uptake $\Delta C$ [$\text{mol/m}^3$] | Final Discrepancy % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Southern Ocean (SO)** | $0.01400$ | $0.017790$ | $0.017791$ | **$+0.003791$** | **$+0.0004\%$** |
| **North Atlantic (NA)** | $0.01413$ | $0.015847$ | $0.015848$ | $+0.001717$ | $+0.0061\%$ |
| **South Atlantic (SA)** | $0.01579$ | $0.016597$ | $0.016597$ | $+0.000807$ | $+0.0021\%$ |
| **Pacific Ocean (PO)** | $0.01834$ | $0.018099$ | $0.018099$ | $-0.000241$ | $+0.0015\%$ |
| **Indian Ocean (IO)** | $0.01671$ | $0.014838$ | $0.014839$ | $-0.001872$ | $+0.0019\%$ |

- **Southern Ocean Dominance Confirmed**: Absorbs $>2.2\times$ more $\text{CO}_2$ than the North Atlantic over 2000–2018.
- **Robustness**: Trajectory discrepancies between fitted SST and exact paper equations are $<0.007\%$ across all basins.

---

## 📈 Visualizations

- **Figure 4 Validation**: [`sst_outputs/figure4_co2_trajectory_validation.png`](sst_outputs/figure4_co2_trajectory_validation.png)
- **Figure 5 Salinity Trajectories**: [`sst_outputs/figure5_salinity_evolution.png`](sst_outputs/figure5_salinity_evolution.png)
- **SST Linear Fits**: [`sst_outputs/sst_actual_vs_trend_paper_boundaries.png`](sst_outputs/sst_actual_vs_trend_paper_boundaries.png)
- **Error Decomposition**: [`sst_outputs/sst_to_ode_error_decomposition.png`](sst_outputs/sst_to_ode_error_decomposition.png)

---

## 📜 Citation & License

This project is developed as part of a 20-Credit research project at the **International Institute of Information Technology Bangalore (IIITB)**.

If you build upon this work, please cite:
```bibtex
@article{sunny2023ocean,
  title={The ocean carbon sinks and climate change},
  author={Sunny, Eros M and Ashok, Balakrishnan and Balakrishnan, Janaki and Kurths, J{\"u}rgen},
  journal={Chaos: An Interdisciplinary Journal of Nonlinear Science},
  volume={33},
  number={10},
  pages={103134},
  year={2023},
  publisher={AIP Publishing}
}
```

