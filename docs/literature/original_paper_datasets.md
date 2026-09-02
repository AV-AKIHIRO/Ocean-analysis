# Observational Datasets and Data Sources in the Original Paper

**Associated Paper**: *The ocean carbon sinks and climate change*  
 
**Authors**: Eros M. Sunny, Balakrishnan Ashok, Janaki Balakrishnan, and Jürgen Kurths  
 

---

## 1. Summary Table of Datasets and Data Sources

| # | Dataset / Data Source Name | Source / Institution / Primary Reference | Variables & Parameters Extracted | Model Application & Short Description |
|---|---|---|---|---|
| **1** | **Keeling Curve Atmospheric $\text{CO}_2$ Record** | Mauna Loa Observatory / C. D. Keeling, R. F. Keeling (2008, Ref. 7) | Monthly atmospheric $p\text{CO}_2$ ($\mu\text{atm}$) from Jan 2000 | Linear fit $p\text{CO}_2(t) = 0.18152t + 367.41\ \mu\text{atm}$ provides the time-dependent atmospheric $\text{CO}_2$ forcing in air-sea gas exchange. |
| **2** | **Sea Surface Temperature (SST) Dataset (COBE-SST / ICOADS)** | NOAA PSL / Ishii et al. (2005, Ref. 30) | Monthly basin-averaged SST (2000–2018) for all 5 oceans | Linear fits $T_{\text{SS}i}(t)$ update temperature dynamically to compute Schmidt numbers, gas transfer velocities, $\text{CO}_2$ solubility, and deep-sea corrections. |
| **3** | **Global Gridded Sea Surface $p\text{CO}_2$ Product** | ETH Zurich / CDIAC / Landschützer et al. (Ref. 28) | Gridded monthly surface ocean $p\text{CO}_2$ (Jan 2000 baseline) | Average $p\text{CO}_2$ sets baseline initial aqueous $\text{CO}_2$ concentrations ($C_i^{\text{initial}}$) and chemical reaction products ($N_i^{\text{initial}}$) in Table I. |
| **4** | **Ocean Salinity Baseline Climatology & GCM Data** | FAMOUS GCM / Alkhayuon et al. (2019, Ref. 29); Drushka et al. (2019, Ref. 14) | Baseline basin salinities ($S_i$) and reference salinity ($S_0 = 35\text{ psu}$) | Sets initial salinity conditions in Table I for Jan 2000 and calculates density-driven deep-water transport ($q_{ij}$). |
| **5** | **Blended Global Sea Surface Wind Dataset** | NOAA NCEI (Ref. 31) / Wanninkhof (2014, 2017, Refs. 19, 20) | Monthly zonal & meridional wind vectors ($v_x, v_y$) and $10\text{ m}$ wind speed ($u_{10}$) | $5^\circ \times 5^\circ$ wind fields drive the 2D convective derivative PDE to resolve spatial sink/source maps (Fig. 7); $u_{10} = 6\text{ m/s}$ computes gas transfer velocity. |
| **6** | **Satellite Atmospheric Freshwater Flux (HOAPS 3.2)** | EUMETSAT CM SAF / Fennig et al. (2012, Ref. 18) | Monthly precipitation and evaporation rates ($E - P$) | Net atmospheric freshwater flux ($F_i$) for year 2000 into each basin enters the salinity rate equation ($-\frac{F_i S_0}{V_i}$). |
| **7** | **Polar Sea Ice Extent & Decadal Melt Rates** | Johannessen et al. (1995, Ref. 15); IPCC AR5 (2013, Ref. 16) | Polar sea ice extent, thickness, and decadal trend rates | Computes North Atlantic ice melt freshwater flux ($b_{\text{NA}} = 3.28 \times 10^3\text{ m}^3\text{/s}$) and Southern Ocean ice growth rate ($b_{\text{SO}} = -6.944 \times 10^2\text{ m}^3\text{/s}$). |
| **8** | **Amazon River Fluvial Discharge & Paleodrainage Data** | Gupta (2008, Ref. 17); Mapes et al. (2006, 2009, Refs. 43, 44); Shephard et al. (2010, Ref. 45) | Amazon River runoff ($q_{\text{Amaz}} = 0.24\text{ Sv}$) and Miocene westward flow history | Supplies freshwater dilution and carbon influx to the North Atlantic; simulated as Pacific outflow at $1\times, 50\times, 100\times$ rates for Miocene paleoclimate experiments. |
| **9** | **Ocean Geometry & Major Surface Currents Climatology** | Haidvogel & Bryan (1992, Ref. 9); Sprintall et al. (2009, Ref. 10); Stramma & Peterson (1990, Ref. 11); Johns et al. (1990, Ref. 12); Molinari (1982, Ref. 13) | Basin volumes ($V_i$), surface areas ($A_i$), and surface current transports ($Q_{ij}$) | Parameterizes fixed inter-basin surface currents (Indonesian Throughflow, North Brazil, Agulhas, South Atlantic Current) and box physical dimensions. |
| **10** | **Miocene Temperature Portal & Paleoclimate Proxy Records** | Bolin Centre Database / Lawrence et al. (2021, Ref. 46); Martinot et al. (2022, Ref. 50) | Reconstructed Miocene sea surface temperatures and ice sheet records | Used to validate and benchmark the simulated Miocene climatic shifts (West Antarctic Ice Sheet formation, Western European cooling, Indian Ocean cooling). |

---

## 2. Detailed Description of Datasets: What, Where, and How They Are Used

### 2.1 Atmospheric Carbon Dioxide: The Keeling Curve
* **Data Source**: Continuous in situ atmospheric carbon dioxide records from the Mauna Loa Observatory, Hawaii, initiated by Charles David Keeling (Keeling, 2008, Ref. 7).
* **Where Used in Paper**: Section II.F, Figure 3, Equations (5) and (7).
* **How and For What It Was Used**:
  * **Atmospheric Forcing Parameterization**: The paper establishes the external boundary condition for the atmosphere by fitting a linear trend to the monthly atmospheric $p\text{CO}_2$ time series starting in January 2000:
    $$p\text{CO}_2(t) = 0.18152 t + 367.41 \quad (\mu\text{atm})$$
    where $t$ represents time elapsed in months from January 2000.
  * **Air-Sea Gas Exchange**: This time-dependent function $p\text{CO}_2(t)$ is inserted into the gas transfer flux term across the ocean surface:
    $$\frac{dC_i}{dt}\Bigg|_{\text{air-sea}} = \frac{\gamma_i A_i}{V_i} \left( K_0^i\, p\text{CO}_2(\text{air}) - C_i \right)$$
    representing the rate at which rising atmospheric carbon dioxide forces carbon dissolution into each oceanic basin.

---

### 2.2 Sea Surface Temperature (SST): NOAA COBE-SST / ICOADS
* **Data Source**: Centennial in situ and satellite-blended Sea Surface Temperature dataset (COBE-SST) provided by the NOAA Physical Sciences Laboratory (PSL), Boulder, Colorado (Ishii et al., 2005, Ref. 30).
* **Where Used in Paper**: Section II.F, Appendix B, Appendix D, Figure 10, Equation (D1), Table II.
* **How and For What It Was Used**:
  * **Basin-Averaged Temperature Trends**: Spatially averaged monthly SST data from 2000 to 2018 were computed for each of the five ocean boxes.
  * **Seasonal Smoothing**: To prevent rapid seasonal oscillations from destabilizing long-term box model integration, linear regressions were fitted for each ocean basin (Equation D1, Figure 10):
    * **North Atlantic (Ocean 1)**: $T_{\text{SS}1}(t) = 0.0017 t + 291.90\text{ K}$
    * **South Atlantic (Ocean 2)**: $T_{\text{SS}2}(t) = -0.00015 t + 290.30\text{ K}$
    * **Southern Ocean (Ocean 3)**: $T_{\text{SS}3}(t) = -0.00056 t + 272.29\text{ K}$
    * **Pacific Ocean (Ocean 4)**: $T_{\text{SS}4}(t) = 0.00155 t + 292.88\text{ K}$
    * **Indian Ocean (Ocean 5)**: $T_{\text{SS}5}(t) = 0.00026 t + 291.57\text{ K}$
  * **Thermodynamic Property Updates**: At each integration time step, the updated $T_{\text{SS}i}(t)$ dynamically determines:
    1. The **Schmidt number** $Sc(T)$ via the empirical quartic polynomial (Eq. B1).
    2. The temperature-dependent **gas-transfer velocity** $\gamma_i = \gamma_{660} (Sc / 660)^{-0.5}$.
    3. The **solubility constant** $K_0(T, S)$ via the Weiss formulation (Eq. B2).
    4. The **deep-ocean $\text{CO}_{2\text{(aq)}}$ correction** $C'_i$ (Eq. 10).

---

### 2.3 Initial Sea Surface $p\text{CO}_2$: ETH SOM-FFN Gridded Product
* **Data Source**: Observation-based monthly gridded global sea surface $p\text{CO}_2$ product produced using a Self-Organizing Map Feed-Forward Neural Network (SOM-FFN) by Landschützer et al. (Ref. 28), hosted at the Carbon Dioxide Information Analysis Center (CDIAC).
* **Where Used in Paper**: Section II.E, Table I, Table II.
* **How and For What It Was Used**:
  * **Baseline Carbon Conditions**: Extracted the spatial mean surface ocean $p\text{CO}_2$ for each basin for the baseline initial month (January 2000).
  * **Conversion to Aqueous Concentration**: The observed $p\text{CO}_2$ values were converted into volumetric aqueous concentrations ($C_i^{\text{initial}}$ in $\text{mol/m}^3$) via:
    $$C_i^{\text{initial}} = \frac{[p\text{CO}_2]_i}{R\, T_i}$$
    giving the initial state vector in **Table I**:
    * North Atlantic: $0.01413\text{ mol/m}^3$
    * South Atlantic: $0.01579\text{ mol/m}^3$
    * Southern Ocean: $0.01400\text{ mol/m}^3$
    * Pacific Ocean: $0.01834\text{ mol/m}^3$
    * Indian Ocean: $0.01671\text{ mol/m}^3$
  * **Secondary Reaction Products**: Initialized the reaction product concentrations $N_i^{\text{initial}} = K_{\text{eq}} C_i^{\text{initial}}$ using the hydration equilibrium constant $K_{\text{eq}} = 0.0011$.

---

### 2.4 Baseline Ocean Salinity Data: GCM Climatology
* **Data Source**: Ocean climatology and global box model calibration data based on the FAMOUS coupled general circulation model (Alkhayuon et al., 2019, Ref. 29) and submesoscale surface salinity datasets (Drushka et al., 2019, Ref. 14).
* **Where Used in Paper**: Section II.E, Table I, Equation (4).
* **How and For What It Was Used**:
  * **Initial Salinity State Vector**: Defined the initial salinity conditions for January 2000 in **Table I**:
    * North Atlantic: $34.912\text{ psu}$
    * South Atlantic: $35.435\text{ psu}$
    * Southern Ocean: $34.427\text{ psu}$
    * Pacific Ocean: $34.668\text{ psu}$
    * Indian Ocean: $34.538\text{ psu}$
  * **Thermohaline Flow Formulation**: Used with reference seawater salinity $S_0 = 35\text{ psu}$ and haline contraction coefficient $\beta = 7.5 \times 10^{-4}\text{ psu}^{-1}$ to determine the density differences driving deep thermohaline flows $q_{ij} = K_{ij} \beta (S_i - S_j)$.

---

### 2.5 Sea Surface Wind Fields: NOAA NCEI Blended Wind Products
* **Data Source**: NOAA National Centers for Environmental Information (NCEI) Blended Global Sea Surface Wind Dataset (Ref. 31) and Wanninkhof wind-speed exchange parameterizations (Refs. 19, 20).
* **Where Used in Paper**: Section II.G, Appendix B, Appendix C, Figure 7, Equations (12), (13), (B1), Table II.
* **How and For What It Was Used**:
  * **Spatial Convective Derivative Mapping**: To resolve the geographic distribution of carbon sinks and sources within each ocean box, the paper solves the 2D steady-state advection PDE across a $5^\circ \times 5^\circ$ spatial grid:
    $$v_x \frac{\partial C}{\partial x} + v_y \frac{\partial C}{\partial y} = f$$
    where $v_x(x, y)$ and $v_y(x, y)$ are the observed zonal and meridional wind velocity components for January of specific target years (2003, 2005, 2007, 2011), and $f$ is the box-averaged net rate from the ODE solution.
  * **Iterative PDE Solver**: Discretized using finite differences (Eq. 13) and solved via Gauss–Seidel iterations with a symmetric successive over-relaxation (SSOR) preconditioner, producing the detailed sink/source maps in Figure 7.
  * **Gas Transfer Velocity Scale**: Prescribed a mean global $10\text{ m}$ wind speed $u_{10} = 6\text{ m/s}$ in the parameterization $\gamma_{660} = 0.24 \langle u_{10}^2 \rangle$ to standardize gas exchange rates.

---

### 2.6 Atmospheric Freshwater Flux: CM SAF HOAPS 3.2
* **Data Source**: Hamburg Ocean Atmosphere Parameters and Fluxes from Satellite Data (HOAPS), Version 3.2, produced by EUMETSAT CM SAF (Fennig et al., 2012, Ref. 18).
* **Where Used in Paper**: Section II.C, Section II.D, Equation (4), Table II.
* **How and For What It Was Used**:
  * **Net E - P Computation**: Computed the spatially averaged difference between satellite precipitation rate ($P$) and evaporation rate ($E$), multiplied by the respective ocean surface area for the year 2000.
  * **Net Basin Fluxes ($F_i$)**:
    * North Atlantic ($F_1$): $+0.475\text{ Sv}$
    * South Atlantic ($F_2$): $+0.526\text{ Sv}$
    * Southern Ocean ($F_3$): $-0.209\text{ Sv}$ (net evaporative / ice-growth deficit)
    * Pacific Ocean ($F_4$): $+0.064\text{ Sv}$
    * Indian Ocean ($F_5$): $+2.3495\text{ Sv}$
  * **Salinity Conservation**: Entered directly into the governing salinity evolution equation (Eq. 4) as dilution / concentration forcing terms ($-\frac{F_i S_0}{V_i}$).

---

### 2.7 Polar Sea Ice Melt & Growth Observations
* **Data Source**: Observational satellite sea-ice extent and thickness trends from Johannessen et al. (*Nature*, 1995, Ref. 15) for the Arctic; IPCC Working Group I Fifth Assessment Report (IPCC AR5, 2013, Ref. 16) for the Antarctic.
* **Where Used in Paper**: Section II.C, Section II.D, Equation (4), Table II.
* **How and For What It Was Used**:
  * **Arctic Melting Flux ($b_{\text{NA}}$)**:
    * Northern Hemisphere polar sea ice areal extent: $11.6 \times 10^{12}\text{ m}^2$.
    * Decadal decrease rate: $\approx 4.4\%$ per decade ($\approx 1.054 \times 10^{12}\text{ m}^3$ ice volume loss per decade with mean thickness $\sim 2\text{ m}$).
    * Calculated freshwater discharge into the North Atlantic:
      $$b_{\text{NA}} = 3.28 \times 10^3\text{ m}^3\text{/s}$$
  * **Antarctic Freezing / Expansion Flux ($b_{\text{SO}}$)**:
    * Southern Ocean sea ice areal expansion rate: $\approx 1.8\%$ increase per decade (mean thickness $\sim 1\text{ m}$).
    * Calculated equivalent freshwater extraction rate:
      $$b_{\text{SO}} = -6.944 \times 10^2\text{ m}^3\text{/s}$$
  * Added as localized boundary terms ($b_{\text{NA}} \delta_{i1}$ and $b_{\text{SO}} \delta_{i3}$) in the salinity rate equation (Eq. 4).

---

### 2.8 Fluvial Runoff and Paleodrainage Data: The Amazon River
* **Data Source**: Global fluvial discharge compilations in *Large Rivers: Geomorphology and Management* (Gupta, 2008, Ref. 17); geological provenance and drainage reversal reconstructions (Mapes et al., 2006, 2009, Refs. 43, 44; Shephard et al., 2010, Ref. 45).
* **Where Used in Paper**: Section II.C, Section III.D, Equations (4), (7), (8), Figures 7(f), 8, 9, Table II.
* **How and For What It Was Used**:
  * **Modern Fluvial Baseline**: Parameterized the Amazon as the sole dominant continental river discharge into the North Atlantic with runoff:
    $$q_{\text{Amaz}} = 0.24\text{ Sv}$$
    providing freshwater dilution in Eq. (4) and fluvial carbon influx ($q_{\text{Amaz}} C_{\text{Amaz}} / V_1$) in Eq. (7).
  * **Miocene Paleoclimate Scenario**: In the Miocene epoch, the Amazon flowed westward into the Pacific prior to the Andean uplift. The authors simulated this by redirecting $q_{\text{Amaz}}$ into Box 4 (Pacific Ocean) at three test magnitudes:
    1. $1\times$ present-day discharge ($0.24\text{ Sv}$).
    2. $50\times$ present-day discharge (telescoped long-term Miocene scenario).
    3. $100\times$ present-day discharge.
  * Used to examine whether major fluvial reorganization could serve as a tipping point for global climate shifts, redistribution of ocean carbon sinks, and cooling triggers (Figures 7f, 8, 9).

---

### 2.9 Ocean Geometry & Surface Current Transports
* **Data Source**: Oceanographic surveys and circulation compilations: Haidvogel & Bryan (1992, Ref. 9), Sprintall et al. (2009, Ref. 10), Stramma & Peterson (1990, Ref. 11), Johns et al. (1990, Ref. 12), Molinari (1982, Ref. 13).
* **Where Used in Paper**: Section II.A, Section II.B, Appendix A, Equations (2), (4), (7), (8), Table II.
* **How and For What It Was Used**:
  * **Basin Volume & Surface Area Specifications (Appendix A)**:
    * North Atlantic: $V_1 = 146 \times 10^{15}\text{ m}^3$, $A_1 = 41.49 \times 10^{12}\text{ m}^2$
    * South Atlantic: $V_2 = 160 \times 10^{15}\text{ m}^3$, $A_2 = 40.27 \times 10^{12}\text{ m}^2$
    * Southern Ocean: $V_3 = 71.8 \times 10^{15}\text{ m}^3$, $A_3 = 20.33 \times 10^{12}\text{ m}^2$
    * Pacific Ocean: $V_4 = 710 \times 10^{15}\text{ m}^3$, $A_4 = 165.25 \times 10^{12}\text{ m}^2$
    * Indian Ocean: $V_5 = 264 \times 10^{15}\text{ m}^3$, $A_5 = 70.56 \times 10^{12}\text{ m}^2$
  * **Surface Current Volume Transports ($Q_{ij}$)**:
    * Indonesian Throughflow (Pacific $\to$ Indian): $Q_{45} = 15\text{ Sv}$
    * North Brazil & Guiana Currents (South $\to$ North Atlantic): $Q_{21} = 26\text{ Sv}$
    * Agulhas Leakage (Indian $\to$ South Atlantic): $Q_{52} = 10\text{ Sv}$
    * Agulhas Current / Southern return flow (Indian $\to$ Southern Ocean): $Q_{53} = 60\text{ Sv}$
    * South Atlantic Current (Southern Ocean $\to$ South Atlantic): $Q_{32} = 30\text{ Sv}$
  * **Deep Thermohaline Scale Factor**: Set $K_{ij} \approx 60\text{ Sv}$ for density-driven overturning between adjacent boxes.

---

### 2.10 Miocene Paleoclimate Proxy Portals & Reconstructions
* **Data Source**: *Miocene Temperature Portal* (Bolin Centre Database, Lawrence et al., 2021, Ref. 46); Equatorial Indian Ocean paleoceanographic reconstructions (Martinot et al., 2022, Ref. 50).
* **Where Used in Paper**: Section III.D.
* **How and For What It Was Used**:
  * **Model Validation and Paleoclimate Synthesis**: Provided empirical paleoclimatic evidence used to validate the model's simulation of the reversed Amazon outflow.
  * Specifically, the authors cross-referenced their model predictions against proxy data showing:
    1. The late-Miocene thermal cooling and formation of the **West Antarctic Ice Sheet (WAIS)**, matching the model's simulated intense carbon sink off the Getz, Abbot, and Sulzberger Ice Shelves (Figure 7f).
    2. Regional surface cooling in **Western Europe** and the **Eastern Equatorial Indian Ocean** during the late Miocene.
