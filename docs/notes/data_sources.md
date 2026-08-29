# Source References and Citation Bibliography

**Project**: Extension of Five-Box Conceptual Ocean Model using Satellite Observational Forcing  
**Subject**: Comprehensive Listing of Data Sources, Technical Documentation, Literature Citations, and Web Portals  
**Date**: July 2026  

---

## 1. Original Conceptual Model Reference

* **Paper Title**: *The ocean carbon sinks and climate change*
* **Authors**: Eros M. Sunny, Balakrishnan Ashok, Janaki Balakrishnan, and Jürgen Kurths
* **Journal**: *Chaos: An Interdisciplinary Journal of Nonlinear Science*, Vol. 33, Issue 10, Art. No. 103134 (2023)
* **DOI**: [10.1063/5.0152504](https://doi.org/10.1063/5.0152504)
* **Local Workspace File**: [ba_chaos_X23.pdf](file:///mnt/c/Users/Areen%20Vaghasiya/OneDrive%20-%20iiit-b/Courses/Areen%20SEM9/Ocean%20analysis/ba_chaos_X23.pdf)
* **Usage**: Provides the theoretical framework, salinity evolution equations, density-driven flow formulation ($q_{ij}$), box volume parameters, and baseline year 2000 constant freshwater flux assumptions ($F_{\text{NA}}, F_{\text{SA}}, F_{\text{SO}}, F_{\text{PO}}, F_{\text{IO}}$).

---

## 2. Primary Observational Dataset: CM SAF HOAPS v4.0

### A. Dataset Metadata & Identification
* **Dataset Title**: Hamburg Ocean Atmosphere Parameters and Fluxes from Satellite Data (HOAPS), Version 4.0
* **Product Identifier / DOI**: [10.5676/EUM_SAF_CM/HOAPS/V002](https://doi.org/10.5676/EUM_SAF_CM/HOAPS/V002)
* **Provider / Institution**: EUMETSAT Satellite Application Facility on Climate Monitoring (CM SAF) / Deutscher Wetterdienst (DWD)
* **Web Portal / Data Access**: [http://www.cmsaf.eu/](http://www.cmsaf.eu/) | [https://wui.cmsaf.eu/](https://wui.cmsaf.eu/)
* **Contact Email**: `contact.cmsaf@dwd.de`
* **Local Data Directories**:
  * `data/precipitation_evaporation_CM_SAF_1987_TO_1999`
  * `data/precipitation_evaporation_CM_SAF_2000_TO_2014`
  * `data/precipitation_CM_SAF_1987_TO_1999` & `data/precipitation_CM_SAF_2000_TO_2014`
  * `data/evaporation_CM_SAF_1987_TO_1999` & `data/evaporation_CM_SAF_2000_TO2014`

### B. Technical User Manual
* **Document Title**: *Product User Manual - HOAPS Release 4.0 (Freshwater Flux, Precipitation, Evaporation)*
* **Report ID**: SAF/CM/DWD/PUM/HOAPS/4
* **Local Workspace File**: [saf_cm_dwd_pum_hoaps4_1_1_pdf.pdf](file:///mnt/c/Users/Areen%20Vaghasiya/OneDrive%20-%20iiit-b/Courses/Areen%20SEM9/Ocean%20analysis/saf_cm_dwd_pum_hoaps4_1_1_pdf.pdf)
* **Usage**: Guidance on NetCDF variable structures (`budg`, `rain`, `evap`), physical units ($\text{mm/d}$), spatial resolution ($0.5^\circ \times 0.5^\circ$), 50 km land masking buffers, and uncertainty bounds.

### C. Core Literature Citations for HOAPS
1. **Fennig, K., Schröder, M., Andersson, A., et al. (2017)**:
   * *Title*: CM SAF Climate Data Record HOAPS Version 4.0 — Monthly Means and 6-Hourly Composite Gridded Data.
   * *Publisher*: EUMETSAT CM SAF.
   * *DOI*: [10.5676/EUM_SAF_CM/HOAPS/V002](https://doi.org/10.5676/EUM_SAF_CM/HOAPS/V002)
2. **Andersson, A., Klepp, C., Fennig, K., Baca, S., Schulz, J., & Bumke, K. (2010)**:
   * *Title*: The HOAPS climatology: Validation and analysis of evaporation, precipitation and the freshwater budget over the global ocean.
   * *Journal*: *Tellus A: Dynamic Meteorology and Oceanography*, 62(4), 353–370.
   * *DOI*: [10.1111/j.1600-0870.2010.00457.x](https://doi.org/10.1111/j.1600-0870.2010.00457.x)

---

## 3. High-Latitude Ice Sheet & Sea-Ice References

*Note on Satellite Radiometry Limitations*: HOAPS 4.0 uses passive microwave radiometers (SSM/I and SSMIS on DMSP satellites). Over sea ice and polar continent ice sheets (Antarctica and Greenland), emissivity changes prevent reliable atmospheric retrieval. Cells covered by sea ice are masked as `NaN` in HOAPS.

To address Professor B. A.'s suggestion to double-check high-latitude ice numbers and melt/freeze rates, the following standard polar climate sources are identified:

1. **National Snow and Ice Data Center (NSIDC)**:
   * *Dataset*: Sea Ice Index, Version 3 & Multisensor Analyzed Sea Ice Extent (MASAM2).
   * *URL*: [https://nsidc.org/data](https://nsidc.org/data)
   * *Citation*: Fetterer, F., et al. (2017). *Sea Ice Index, Version 3*. Boulder, Colorado USA. NSIDC. DOI: [10.5067/N8PHQueries](https://doi.org/10.5067/N8PHQueries)
2. **EUMETSAT Ocean and Sea Ice SAF (OSI SAF)**:
   * *Dataset*: Global Sea Ice Concentration Climate Data Record (OSI-450 / OSI-430-b).
   * *URL*: [https://osi-saf.eumetsat.int/](https://osi-saf.eumetsat.int/)
3. **ECMWF Reanalysis v5 (ERA5)**:
   * *Dataset*: ERA5 Monthly Averaged Single Levels — Surface Water Fluxes over Sea Ice, Snowmelt, and Sea-Ice Cover.
   * *Publisher*: Copernicus Climate Change Service (C3S) Climate Data Store (CDS).
   * *URL*: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)
   * *DOI*: [10.24381/cds.f17050d7](https://doi.org/10.24381/cds.f17050d7)

---

## 4. River Runoff & Continental Freshwater Inputs

*Note on Amazon & Major Rivers*: Atmospheric datasets ($E - P$) capture air-sea exchange. Major river discharge dumps freshwater directly into ocean coastal boxes (e.g., the Amazon River into the Atlantic Ocean).

1. **Dai, A., & Trenberth, K. E. (2002)**:
   * *Title*: Estimates of Freshwater Discharge from Earth's Major Rivers into the Global Oceans.
   * *Journal*: *Journal of Hydrometeorology*, 3(6), 660–687.
   * *DOI*: [10.1175/1525-7541(2002)003<0660:EOFDFE>2.0.CO;2](https://doi.org/10.1175/1525-7541(2002)003<0660:EOFDFE>2.0.CO;2)
2. **Global Runoff Data Centre (GRDC)**:
   * *Provider*: World Meteorological Organization (WMO) / Federal Institute of Hydrology (BfG), Koblenz, Germany.
   * *URL*: [https://www.bafg.de/GRDC/](https://www.bafg.de/GRDC/)

---

## 5. Ocean Climatology & Salinity References (For Next Phase)

For validating salinity fixed points, time evolution, and density-driven circulation ($q_{ij}$):

1. **NOAA World Ocean Atlas 2018 (WOA18)**:
   * *Variables*: Monthly & Seasonal Climatologies of Ocean Temperature and Salinity.
   * *Publisher*: NOAA National Centers for Environmental Information (NCEI).
   * *URL*: [https://www.ncei.noaa.gov/products/world-ocean-atlas](https://www.ncei.noaa.gov/products/world-ocean-atlas)
   * *Citations*: 
     * Zweng, M. M., et al. (2018). *World Ocean Atlas 2018, Volume 2: Salinity*. NOAA Atlas NESDIS 82.
     * Locarnini, R. A., et al. (2018). *World Ocean Atlas 2018, Volume 1: Temperature*. NOAA Atlas NESDIS 81.

---

## 6. Open Questions & Points of Clarification for Professor B. A.

To follow up on Professor B. A.'s response email and ensure seamless alignment, the following points can be discussed during the upcoming meeting:

1. **Original Calculation Notes for Indian Ocean ($F_{\text{IO}}$)**:
   * We found that $F_{\text{IO}} = 2.3495\text{ Sv}$ in Table 1 of the paper matches the ratio of the Pacific Ocean area to the Indian Ocean area ($A_{\text{PO}} / A_{\text{IO}} = 165.25 / 70.56 = 2.342$) to within $0.3\%$. 
   * *Question for Prof*: When you examine your notes, could you check whether the Indian Ocean unit rate $(\text{m/s})$ was multiplied by $A_{\text{PO}}$ or if an area-ratio factor was inadvertently applied during table aggregation?
2. **Model Equation Sign Convention ($F_i$ in Eq. 4)**:
   * In Eq. (4) of the paper, $F_i$ is subtracted in the salinity rate equation ($-\frac{F_i S_0}{V_i}$), which acts as a dilution source (precipitation).
   * *Question for Prof*: Was the value $2.3495\text{ Sv}$ entered as a positive dilution term in the simulation code, or was it treated as net evaporation ($\frac{F_i S_0}{V_i}$)?
3. **Ice Melt & Freeze Coupling**:
   * For the Southern Ocean box ($F_{\text{SO}}$), satellite HOAPS data covers the ice-free portion ($\approx 11.17 \times 10^{12}\text{ m}^2$), yielding an average atmospheric flux of $-0.10\text{ Sv}$. When scaled up to the full $20.33 \times 10^{12}\text{ m}^2$ area (including sea ice), the net flux is $-0.198\text{ Sv}$, very close to the paper's $-0.209\text{ Sv}$.
   * *Question for Prof*: Should we keep the scaled atmospheric flux for the Southern Ocean, or incorporate an explicit sea-ice melt/freeze seasonal cycle parameter (e.g. from ERA5 / NSIDC)?
