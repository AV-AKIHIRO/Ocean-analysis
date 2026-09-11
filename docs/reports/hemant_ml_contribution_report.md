# Hemant's Project Contribution Report: Machine Learning & Symbolic Discovery

**Author**: Hemant Gupta  
**Project**: Ocean Carbon Sink & Thermohaline Salinity Analysis  
**Repository**: `Ocean-analysis`  
**Date**: September 2026  

---

## 1. Understanding the ML Models, Data, & Results

### 1.1 What does the Training Data look like?
The training dataset consists of 330 monthly timesteps (January 1987 to December 2014).
* **Time Split**:
  * **Train Set (1987–2009 / 270 months)**: Used to fit model parameters.
  * **Test Set (2010–2014 / 60 months)**: Held-out future data used exclusively for evaluation.
* **Feature Table (19 Input Columns)**:
  * **Satellite Freshwater Fluxes ($F_{\text{NA}}, F_{\text{SA}}, F_{\text{SO}}, F_{\text{PO}}, F_{\text{IO}}$)**: Measured in Sverdrups ($\text{Sv}$).
  * **Sea Surface Temperatures ($\text{SST}_{\text{NA}} \dots \text{SST}_{\text{IO}}$)**: Measured in Kelvin ($\text{K}$).
  * **Atmospheric $\text{CO}_2$ ($p\mathrm{CO}_{2,\mathrm{air}}$)**: Keeling Curve trajectory in $\mu\text{atm}$.
  * **Seasonality (`month_sin`, `month_cos`)**: Cyclical calendar features.
  * **Air-Sea Disequilibrium ($\mathrm{DISEQ}_i = K_0(\mathrm{SST}_i) \cdot p\mathrm{CO}_{2,\mathrm{air}} - C_i(t-1)$)**: Physics-informed feature measuring carbon concentration imbalance.
  * **Lagged State ($S_i(t-1), C_i(t-1)$)**: Previous month's state for autoregressive prediction.
* **Target**:
  * Predict **monthly increments** $\Delta S_i(t) = S_i(t) - S_i(t-1)$ and $\Delta C_i(t) = C_i(t) - C_i(t-1)$.
  * Full 27-year trajectories are reconstructed by adding predicted increments step-by-step (cumulative sum), mirroring numerical ODE integration.

---

### 1.2 What is $R^2$ Accuracy?
The $R^2$ (R-squared / Coefficient of Determination) score measures how well predictions match true data:
* $R^2 = 1.0$ (100%): Perfect prediction.
* $R^2 = 0.0$: Prediction is no better than taking the average.
* $R^2 < 0.0$: Prediction is worse than a simple average (extrapolation failure).

---

### 1.3 What is Ridge Regression?
**Linear Regression** fits a straight line ($y = w_1 x_1 + w_2 x_2 + b$). However, when features are correlated (like freshwater fluxes across neighboring oceans), standard linear regression can become unstable and overfit.

**Ridge Regression** is a regularized linear model that adds a small penalty to prevent weights from getting too large:
$$\text{Loss} = \text{MSE} + \alpha \sum w_i^2$$
This makes Ridge Regression robust against multi-collinearity while remaining simple, smooth, and physically realistic.

---

### 1.4 Key Findings & Physical Reasons (Results Table)

| Target | Winning Model | Test $R^2$ Score | Key Finding & Physical Reason |
|:---|:---|:---:|:---|
| **Salinity ($S_i$)** | **Ridge Regression** | **$0.984$** (Near Perfect) | **Physical Reason**: Salinity changes are dominated by freshwater dilution ($F_i$). Because salt dilution is nearly linear with freshwater volume input, Ridge Regression captures 98.4% of the variance across all 5 basins without overfitting. |
| **Dissolved $\text{CO}_2$ ($C_i$)** | **Linear Regression** | **$0.950$** (High Accuracy) | **Physical Reason**: The fundamental physics equation for air-sea gas exchange is $\frac{dC_i}{dt} \propto \text{DISEQ}_i$. Linear regression learns the exact physical gas transfer rate directly from the $\text{DISEQ}_i$ feature! |

---

## 2. Symbolic Regression Discovery (`scripts/na_io_symbolic_regression.py`)

### 2.1 Candidate Variables Evaluated
We tested symbolic regression on candidate variables to see which ones govern Indian Ocean dissolved $\text{CO}_2$ ($C_{\text{IO}}$):
* **Candidate Pool**: 
  1. $C_{\text{NA}}$ (North Atlantic Dissolved $\text{CO}_2$)
  2. $p\text{CO}_{2,\text{air}}$ (Atmospheric $\text{CO}_2$ Forcing)
  3. $S_{\text{diff}} = S_{\text{NA}} - S_{\text{IO}}$ (Salinity Gradient driving thermohaline flow)
  4. $F_{\text{IO}}$ (Indian Ocean Freshwater Flux)
  5. Inter-basin polynomial interaction terms: $(C_{\text{NA}} \cdot p\text{CO}_{2,\text{air}})$, $(S_{\text{diff}} \cdot C_{\text{NA}})$, etc.

### 2.2 Why certain variables were selected vs. excluded:
* **Selected Variables**:
  * $p\text{CO}_{2,\text{air}}$ and $(C_{\text{NA}} \cdot p\text{CO}_{2,\text{air}})$ were selected as dominant terms ($R^2 = 0.999999$).
  * **Why**: $p\text{CO}_{2,\text{air}}$ represents the global atmospheric driver pushing carbon into both basins simultaneously. The interaction term $(C_{\text{NA}} \cdot p\text{CO}_{2,\text{air}})$ captures the co-evolution of North Atlantic and Indian Ocean carbon uptake.
* **Excluded Variables**:
  * $S_{\text{diff}}$ and $F_{\text{IO}}$ had near-zero coefficients.
  * **Why**: At monthly basin scales, atmospheric gas exchange dominates over month-to-month salinity fluctuations. Thermohaline transport acts as a steady background baseline rather than a high-frequency monthly fluctuation.

### 2.3 Discovered Governing Algebraic Equation
$$\mathbf{C_{\text{IO}} = 0.016303 + (1.195 \times 10^{-6} \cdot p\text{CO}_{2,\text{air}}) + (6.234 \times 10^{-8} \cdot C_{\text{NA}} \cdot p\text{CO}_{2,\text{air}})}$$

![Symbolic Regression Visualization](../output/symbolic_regression_na_io.png)
*Figure 1: Symbolic Regression discovery fitting actual ODE C_IO with R² = 0.999999.*

---

## 3. Feature Importance Analysis (`scripts/ml_feature_importance.py`)

Using Random Forest feature importance algorithms, we determined which physical features drive basin-level predictions:

### 3.1 Salinity Drivers
![Salinity Feature Importance](../output/feature_importance_salinity.png)
*Figure 2: Relative Feature Importance for Salinity Increments.*

* **Top Feature**: $F_{\text{NA}}$ (North Atlantic Freshwater Flux, **35.6%** importance).
* **Second Feature**: $F_{\text{SA}}$ (South Atlantic Freshwater Flux, **25.0%** importance).
* **Physical Insight**: Freshwater evaporation/precipitation ($F_i$) completely dominates salinity variance, confirming why linear Ridge regression achieves $0.984\ R^2$.

### 3.2 $\text{CO}_2$ Drivers
![CO2 Feature Importance](../output/feature_importance_co2.png)
*Figure 3: Relative Feature Importance for CO2 Increments.*

* **Top Features**: Air-Sea Disequilibrium terms ($\text{DISEQ}_{\text{IO}}$, $\text{DISEQ}_{\text{PO}}$, $\text{DISEQ}_{\text{SO}}$) and $p\text{CO}_{2,\text{air}}$ account for over **50% of model importance**.
* **Physical Insight**: Proves that air-sea chemical imbalance is the primary driver of carbon dissolution in surface waters.

---

## 4. Next Step Proposal: Physics-Informed Neural Networks (PINNs)

### 4.1 Motivation
Pure data-driven models (tree models, LSTMs) fail when projecting to **2030** because test-set atmospheric $\text{CO}_2$ exceeds training bounds. Furthermore, surface models miss deep-ocean transport (like NADW carbon export in the South Atlantic).

### 4.2 PINN Architecture & Loss Penalty Formulation
We propose implementing a **Physics-Informed Neural Network (PINN)** that enforces physical conservation laws directly inside the loss function:

$$\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{Data}} + \lambda \cdot \mathcal{L}_{\text{Physics}}$$

1. **Data Loss ($\mathcal{L}_{\text{Data}}$)**: Standard Mean Squared Error against satellite observations.
   $$\mathcal{L}_{\text{Data}} = \frac{1}{N}\sum \left( y_{\text{pred}} - y_{\text{observed}} \right)^2$$
2. **Physics Loss ($\mathcal{L}_{\text{Physics}}$)**: Enforces conservation of mass according to the 5-box ODE differential equations:
   $$\mathcal{L}_{\text{Physics}} = \frac{1}{N}\sum \left| \frac{dC_i}{dt} - \left[ \frac{\gamma_i A_i}{V_i}(K_0 p\text{CO}_{2,\text{air}} - C_i) + \text{Transport}_{ij} \right] \right|^2$$

### 4.3 Expected Benefits
* **Guaranteed Physical Bounds**: Prevents AI models from predicting unphysical negative concentrations or infinite spikes during 2030 projections.
* **Deep Ocean Awareness**: Injects deep-ocean thermohaline transport constraints into surface ML models.
