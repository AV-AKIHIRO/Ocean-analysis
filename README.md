# Ocean Carbon Sink & Thermohaline Salinity Analysis

A coupled physics-based and data-driven investigation into ocean carbon uptake, salinity dynamics, and thermohaline circulation (1987–2014) with projections to 2030 across five global ocean basins.

---

## 🌟 Primary Reports & Deliverables

The core findings, mathematical derivations, and machine learning benchmarks are documented in the two primary reports:

1. 📄 **[Ocean Carbon Sink Implementation Report (PDF)](docs/reports/ocean_carbon_sink_implementation_report.pdf)**
   *Comprehensive milestone implementation report detailing the mathematical formulation, calibration against WOA hydrography, and integration with satellite freshwater forcing.*
2. 📊 **[Physics vs ML Ocean Prediction Report (Markdown)](docs/reports/physics_vs_ml_ocean_prediction_report.md)**
   *Full benchmark comparing the 5-box coupled ODE system against 6 machine learning architectures (Ridge, Linear Regression, Random Forest, XGBoost, LSTM, Transformer) with autoregressive projections to 2030.*

---

## 🌊 Project Overview

This repository implements a **five-box coupled ocean model** representing the world's major ocean basins:
- **North Atlantic (NA)**
- **South Atlantic (SA)**
- **Southern Ocean (SO)**
- **Pacific Ocean (PO)**
- **Indian Ocean (IO)**

### Key Components:
- **Physics-Based ODE System**: Simulates basin salinity $S_i(t)$ and dissolved inorganic carbon $C_i(t)$ driven by CM SAF HOAPS satellite evaporation/precipitation forcing ($E-P$), Keeling curve atmospheric $p\text{CO}_2$, and Weiss $K_0$ temperature-dependent solubility.
- **Machine Learning Benchmarks**: 6 ML architectures trained on observable surface data to test whether data-driven models can substitute physical ODEs, identifying where ML succeeds and where physical deep-ocean dynamics (e.g. NADW formation) remain indispensable.

---

## 📁 Repository Structure

```text
├── README.md                                             # Project overview & navigation hub
├── ocean_carbon_sink_implementation_report.pdf           # 🌟 Latest PDF implementation report
├── physics_vs_ml_ocean_prediction_report.md              # 🌟 Latest Markdown ML benchmark report
├── .gitignore                                            # Git ignore rules
│
├── docs/                                                 # 📚 Complete documentation hub
│   ├── reports/                                          # Milestone & final project reports
│   │   ├── ocean_carbon_sink_implementation_report.pdf   # 🌟 Primary PDF Deliverable
│   │   ├── physics_vs_ml_ocean_prediction_report.md      # 🌟 Primary Markdown Benchmark Report
│   │   ├── project_status.md                             # Methodology & tracking notes
│   │   └── archive/                                      # Historical milestone reports (REPORT_1, Progress_Report)
│   ├── literature/                                  # Reference papers (Birchfield '89, HOAPS manuals)
│   │   ├── birchfield_1989_salt_chaos.pdf
│   │   └── hoaps4_user_manual.pdf
│   └── notes/                                       # Supporting notes, data dictionaries & task sheets
│       ├── data_sources.md
│       ├── dataset_verification.md
│       ├── task_2_data.md
│       └── project_proposal.pdf
│
├── data/                                            # 💾 Satellite data & NetCDF files
│   ├── evaporation_CM_SAF_1987_TO_1999/
│   ├── evaporation_CM_SAF_2000_TO2014/
│   ├── precipitation_CM_SAF_1987_TO_1999/
│   ├── precipitation_CM_SAF_2000_TO_2014/
│   ├── precipitation_evaporation_CM_SAF_1987_TO_1999/
│   └── precipitation_evaporation_CM_SAF_2000_TO_2014/
│
├── scripts/                                         # ⚙️ Analysis and modeling scripts
│   ├── calculate_areas.py                           # Computes basin surface areas
│   ├── inspect_data.py                              # NetCDF variable & dimension inspector
│   ├── inspect_spatial.py                           # Spatial grid and coordinate checks
│   ├── visualize_masks.py                           # Visualizes 5-box ocean basin masks
│   ├── process_data.py                              # Computes basin-integrated freshwater fluxes (Sv)
│   ├── ode_simulation.py                            # Salinity-only ODE simulation
│   ├── co2_salinity_simulation.py                   # Full 10-D coupled CO2 + Salinity ODE simulation
│   ├── ml_prepare_data.py                           # Feature engineering & DISEQ computation
│   ├── ml_train_compare.py                          # Trains & evaluates 6 ML models
│   └── ml_forecast.py                               # Autoregressive rollout & 2030 projections
│
└── output/                                          # 📈 Generated figures, tables & model weights
    ├── *.png                                        # Comparison timeseries, R² heatmaps, forecasts
    ├── *.csv                                        # Extracted fluxes, features, targets, simulations
    ├── *.pkl                                        # Serialized ML results and metrics
    └── *.pt                                         # Trained PyTorch model weights (LSTM, Transformer)
```

---

## 🚀 Execution Workflow

To reproduce the analysis from raw satellite data to final ML predictions:

1. **Extract Basin Freshwater Fluxes**:
   ```bash
   python scripts/process_data.py
   ```
2. **Run Physics ODE Simulation**:
   ```bash
   python scripts/co2_salinity_simulation.py
   ```
3. **Prepare ML Datasets & Physics Features**:
   ```bash
   python scripts/ml_prepare_data.py
   ```
4. **Train & Benchmark ML Models**:
   ```bash
   python scripts/ml_train_compare.py
   ```
5. **Generate 2030 Projections**:
   ```bash
   python scripts/ml_forecast.py
   ```
