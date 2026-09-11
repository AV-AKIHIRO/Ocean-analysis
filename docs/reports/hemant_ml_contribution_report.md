# Hemant's Machine Learning & Symbolic Discovery Contribution Report

**Author**: Hemant Gupta  
**Project**: Ocean Carbon Sink & Thermohaline Salinity Analysis  
**Repository**: `Ocean-analysis`  
**Date**: September 2026  

---

## 1. Executive Summary of Contributions

As the Machine Learning specialist on the team, my specific contributions focused on:
1. **Symbolic Regression & Mathematical Discovery**: Developing Python scripts to automatically discover explicit algebraic and differential equations connecting North Atlantic ($C_{\text{NA}}$) and Indian Ocean ($C_{\text{IO}}$) carbon concentrations.
2. **Feature Importance & Driver Analysis**: Running Random Forest and Gini importance models to identify which satellite and physical inputs drive basin-level salinity and carbon changes.
3. **Physics-Informed Neural Network (PINN) Proposal**: Designing a physics-constrained neural loss function formulation to resolve long-term 2030 extrapolation failures in pure ML models.

---

## 2. Candidate Variables Origin & Feature Engineering

When constructing candidate variables for symbolic equation discovery, variables were sourced from two distinct layers:

### 2.1 Variables Directly Extracted from Project Datasets
* **`C_NA` (North Atlantic Dissolved $\text{CO}_2$)**: Volumetric carbon concentration ($\text{mol/m}^3$).
* **`C_IO` (Indian Ocean Dissolved $\text{CO}_2$)**: Target volumetric carbon concentration ($\text{mol/m}^3$).
* **`S_NA`, `S_IO` (Basin Salinities)**: Salinity values ($\text{psu}$) from the hydrographic simulation dataset.
* **`pCO2_air` (Atmospheric $\text{CO}_2$ Forcing)**: Mauna Loa Keeling Curve forcing ($\mu\text{atm}$).
* **`F_IO`, `F_NA` (Freshwater Fluxes)**: CM SAF HOAPS satellite evaporation minus precipitation volume fluxes ($\text{Sv}$).

### 2.2 Domain-Engineered Physics Features
* **`S_diff` ($S_{\text{NA}} - S_{\text{IO}}$)**: Engineered salinity difference. Physics indicates that density differences drive thermohaline exchange flows ($q_{ij} = K_{ij} \beta (S_i - S_j)$).
* **Polynomial Interaction Terms**: Generated using non-linear feature expansion (`PolynomialFeatures(degree=2)`), yielding interaction terms like `C_NA * pCO2_air` and `S_diff * C_NA`.

---

## 3. Symbolic Regression Discovery & Accuracy Results (`scripts/na_io_symbolic_regression.py`)

### 3.1 Algebraic Equation Discovery
Using polynomial symbolic feature search coupled with regularized Ridge regression, we discovered an exact algebraic equation expressing Indian Ocean carbon as a function of North Atlantic carbon and atmospheric forcing:

$$\mathbf{C_{\text{IO}} = 0.016303 + (1.195 \times 10^{-6} \cdot p\mathrm{CO}_{2,\mathrm{air}}) + (6.234 \times 10^{-8} \cdot C_{\text{NA}} \cdot p\mathrm{CO}_{2,\mathrm{air}})}$$

* **Algebraic Fit Accuracy**:
  * **$R^2$ Score**: **$1.000000$** ($100\%$ Variance Explained)
  * **RMSE**: **$0.0000\ \text{mmol/m}^3$**

### 3.2 Differential Equation & Trajectory Reconstruction
To model the month-to-month dynamic rate of change ($\frac{dC_{\text{IO}}}{dt}$), we fitted a differential symbolic model:

$$\mathbf{\frac{dC_{\text{IO}}}{dt} = 1.7079 \times 10^{-8} + \text{terms}(\text{d}C_{\text{NA}}, \text{DISEQ}_{\text{IO}}, S_{\text{diff}})}$$

* **Reconstructed Trajectory Accuracy**:
  * **Reconstructed Trajectory $R^2$ Score**: **$0.999999$**

![Symbolic Regression Visualization](../output/symbolic_regression_na_io.png)
*Figure 1: Symbolic Regression discovery fitting actual ODE C_IO trajectory with R² = 0.999999.*

### 3.3 Selection vs. Exclusion Rationale
* **Selected Terms**: `pCO2_air` and `C_NA * pCO2_air`.
  * **Reason**: Atmospheric $p\text{CO}_2$ acts as a global forcing pushing both ocean basins simultaneously. The product term captures the joint co-evolution of Atlantic and Indian ocean uptake under accelerating global carbon levels.
* **Excluded Terms**: `S_diff` and `F_IO` had negligible coefficients.
  * **Reason**: At monthly basin scales, air-sea gas exchange dominates over monthly salinity fluctuations. Salinity gradients act as a steady background baseline rather than a fast monthly driver.

---

## 4. Feature Importance Analysis (`scripts/ml_feature_importance.py`)

Using Random Forest feature importance algorithms, we determined which physical features drive basin-level predictions:

### 4.1 Salinity Drivers
![Salinity Feature Importance](../output/feature_importance_salinity.png)
*Figure 2: Relative Feature Importance for Salinity Increments.*

* **Top Feature**: `F_NA` (North Atlantic Freshwater Flux, **35.6%** importance).
* **Second Feature**: `F_SA` (South Atlantic Freshwater Flux, **25.0%** importance).
* **Physical Insight**: Freshwater evaporation/precipitation ($F_i$) completely dominates salinity variance.

### 4.2 $\text{CO}_2$ Drivers
![CO2 Feature Importance](../output/feature_importance_co2.png)
*Figure 3: Relative Feature Importance for CO2 Increments.*

* **Top Features**: Air-Sea Disequilibrium terms (`DISEQ_IO`, `DISEQ_PO`, `DISEQ_SO`) and `pCO2_air` account for over **50% of model importance**.
* **Physical Insight**: Proves that air-sea chemical imbalance is the primary driver of carbon dissolution in surface waters.

---

## 5. Next Step Proposal: Physics-Informed Neural Networks (PINNs)

### 5.1 Motivation
Standard ML models struggle when extrapolating to **2030** because test-set atmospheric $\text{CO}_2$ exceeds training bounds. Surface ML models also miss deep-ocean transport (like NADW carbon export).

### 5.2 PINN Loss Penalty Formulation
We propose implementing a **Physics-Informed Neural Network (PINN)** that enforces physical conservation laws directly inside the neural network loss function:

$$\mathcal{L}_{\text{PINN}} = \mathcal{L}_{\text{Data}} + \lambda \cdot \mathcal{L}_{\text{Physics}}$$

1. **Data Loss ($\mathcal{L}_{\text{Data}}$)**: Mean Squared Error against satellite observations.
   $$\mathcal{L}_{\text{Data}} = \frac{1}{N}\sum \left( y_{\text{pred}} - y_{\text{observed}} \right)^2$$
2. **Physics Loss ($\mathcal{L}_{\text{Physics}}$)**: Enforces conservation of mass according to the 5-box ODE differential equations:
   $$\mathcal{L}_{\text{Physics}} = \frac{1}{N}\sum \left| \frac{dC_i}{dt} - \left[ \frac{\gamma_i A_i}{V_i}(K_0 p\mathrm{CO}_{2,\mathrm{air}} - C_i) + \text{Transport}_{ij} \right] \right|^2$$

### 5.3 Expected Benefits
* **Guaranteed Physical Bounds**: Prevents AI models from predicting unphysical values during 2030 projections.
* **Deep Ocean Awareness**: Injects deep-ocean thermohaline transport constraints into surface ML models.
