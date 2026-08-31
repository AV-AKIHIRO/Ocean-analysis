# SST Workstream — Exact Plan and Documentation

## Objective

Own and complete the **SST → physics-model validation** workstream for the five-ocean carbon box model based on Sunny et al. (2023).

## Ground truth from the original paper

The original paper defines five ocean boxes: North Atlantic, South Atlantic, Southern Ocean, Pacific and Indian Ocean.

For SST, Appendix D gives these linear relationships, in Kelvin, with `t` = months from January 2000:

| Ocean | Paper equation |
|---|---|
| NA | `T = 0.0017 t + 291.9` |
| SA | `T = -0.00015 t + 290.3` |
| SO | `T = -0.00056 t + 272.29` |
| PO | `T = 0.00155 t + 292.88` |
| IO | `T = 0.00026 t + 291.57` |

The paper says these were obtained from monthly SST series that contain seasonal oscillations, followed by linear fitting.

## What you should do

### 1. Reproduce the SST input

Use COBE-2 SST for January 2000–December 2018.

Expected output:

`228 months × 180 lat × 360 lon`

Convert the values to Kelvin before comparing coefficients with Appendix D.

### 2. Correct the basin masks

Use the explicit geographic boundaries from the original paper:

- `70°W = 290°E` at the South Atlantic/Pacific boundary
- `20°E` at the South Atlantic/Indian boundary
- `140°E` for the Indian/Pacific boundary around the Tasman region
- all seas south of `60°S` = Southern Ocean
- Atlantic wraps across `360°/0°`

Important: the paper also mentions the Indonesian archipelago. A pure rectangular longitude rule cannot reproduce that coastline exactly, so document this as a limitation rather than claiming pixel-perfect reproduction.

### 3. Recompute basin SST

For each ocean:

`basin SST(t) = mean(valid SST grid cells inside basin)`

Save both °C and K versions.

### 4. Refit Appendix D

Fit:

`T_i(t) = a_i * t + b_i`

where `t = 0,...,227`.

Report:

- slope
- intercept
- R²
- residual RMSE
- difference from paper slope
- difference from paper intercept

### 5. Perform the key diagnostic experiment

Run three cases:

**A — existing implementation**

Current masks + current SST fits.

**B — corrected SST preprocessing**

Paper-boundary approximation + newly fitted coefficients.

**C — paper SST forcing**

Use the exact Appendix-D equations from the paper.

Send all three SST variants through the same ODE implementation.

This tells you whether downstream differences are caused by:

- basin-mask definitions
- SST coefficients
- other parts of the ODE implementation

### 6. Validate the physics output

Compare the three runs against the original paper's Figure 4.

For each basin report:

- initial CO₂
- final CO₂
- trend
- RMSE / MAE against any digitized reference available
- qualitative trajectory shape
- Southern Ocean ranking

The paper's main temporal conclusion is that the Southern Ocean is the strongest CO₂ absorber, followed by the Atlantic.

### 7. Handoff

Give the physics-model owner:

- corrected SST CSV
- trend comparison CSV
- basin-mask definition
- exact equations
- validation plot
- discrepancy analysis

## Your final deliverable

Produce a folder containing:

```text
sst_outputs/
├── sst_monthly_paper_boundaries_C.csv
├── sst_monthly_paper_boundaries_K.csv
├── sst_trend_comparison.csv
├── sst_actual_vs_trend_paper_boundaries.png
└── README_SST_VALIDATION.md
```

And a short conclusion:

> The SST preprocessing was reproduced using the paper's geographic boundaries. The remaining difference from Appendix D is quantified basin-by-basin. The corrected SST forcing is then tested in the same physics ODE to determine its downstream effect on the CO₂ trajectory.

## After this

Your next contribution should be one of:

1. **Physics validation:** own the Figure-4 reproduction and error analysis.
2. **PINN investigation:** explain why the PINN collapses toward nearly constant basin values.
3. **Southern Ocean analysis:** investigate why ML and ODE CO₂ projections diverge strongly there.
4. **Reproducibility:** build automated tests for units, masks, initial conditions, solver settings and output sanity.

The first item is the most directly connected to your current SST work.
