# Project: Extending "The Ocean Carbon Sinks and Climate Change" using Real-World Freshwater Flux Data

## 1. Project Background

This project extends the research paper:

"The Ocean Carbon Sinks and Climate Change" (Chaos, 2023)

The original paper uses a five-box ocean model containing:

1. North Atlantic (NA)
2. South Atlantic (SA)
3. Southern Ocean (SO)
4. Pacific Ocean (PO)
5. Indian Ocean (IO)

The model studies the evolution of:
- Ocean CO2 concentration
- Salinity
- Density-driven circulation
- Carbon sinks and sources


---

## 2. Our Team's Extension

The original model makes several simplifying assumptions.

Different team members are working on improving different parts:

- Salinity and CO2 dynamics
- Density-driven and surface flows
- Freshwater flux (this project)

Our goal for the freshwater component is:

"Replace the static freshwater forcing assumptions of the original model with real-world time-varying observational data."


---

## 3. What the Original Paper Assumed

The paper calculates freshwater flux values for the year 2000 and keeps them constant throughout the simulation.

The atmospheric freshwater flux values are:

North Atlantic:
F_NA = 0.475 Sv

South Atlantic:
F_SA = 0.526 Sv

Southern Ocean:
F_SO = -0.209 Sv

Pacific Ocean:
F_PO = 0.064 Sv

Indian Ocean:
F_IO = 2.3495 Sv


The original model ignores:
- Seasonal variability
- Interannual variability
- Long-term trends in freshwater forcing
- Most river runoff except Amazon


---

## 4. Data Currently Available

We have downloaded the following CM SAF HOAPS datasets covering Jan 2000 - Dec 2014:

### Primary dataset

precipitation_evaporation_CM_SAF_2000_TO_2014

This contains EMP (evaporation - precipitation), which directly represents the atmospheric freshwater budget.

This is the main dataset to use.


### Additional datasets

precipitation_CM_SAF_2000_TO_2014

Contains precipitation data.

evaporation_CM_SAF_2000_TO2014

Contains evaporation data.

These are optional and should only be used for further analysis after EMP processing.


---

## 5. Current Objective

The immediate goal is to transform the gridded satellite data into the same five-box representation used in the paper.

The final desired output is:

F_NA(t)
F_SA(t)
F_SO(t)
F_PO(t)
F_IO(t)

where each quantity is a monthly freshwater flux time series from Jan 2000 to Dec 2014.


---

## 6. Immediate Next Steps

### Step 1: Inspect the NetCDF files

Open a sample EMP file and identify:

- Variable names
- Dimensions
- Latitude array
- Longitude array
- Time array
- Units of EMP


Example using Python:

import xarray as xr

ds = xr.open_dataset("path_to_file.nc")

print(ds)
print(ds.variables)


---

### Step 2: Understand units

Determine whether EMP is stored as:

- mm/day
- kg/m²/s
- m/s
- or another unit

Determine the sign convention:

Does positive value mean:
- Evaporation > Precipitation (E - P)
or
- Precipitation > Evaporation (P - E)


---

### Step 3: Create ocean basin masks

Create masks corresponding to the original paper's five ocean boxes.

The Southern Ocean is all waters south of 60°S.

The other ocean boundaries should match the paper's Appendix A as closely as possible.


---

### Step 4: Calculate monthly freshwater forcing

For each month and each ocean:

- Apply the ocean mask
- Perform area-weighted averaging

The weighting should use:

weight = cos(latitude)

because grid cells have different physical areas at different latitudes.


---

### Step 5: Validate against the paper

Calculate the average freshwater flux around the year 2000 and compare against:

NA = 0.475 Sv
SA = 0.526 Sv
SO = -0.209 Sv
PO = 0.064 Sv
IO = 2.3495 Sv


---

### Step 6: Produce final outputs

Generate:

1. CSV file:
date, F_NA, F_SA, F_SO, F_PO, F_IO


2. Plots:
- Freshwater flux vs time for all five oceans
- Comparison between observed variability and the paper's constant values


3. Statistical analysis:
For each ocean calculate:
- Mean
- Standard deviation
- Minimum
- Maximum


---

## 7. Important Notes

Do NOT start with machine learning.

The current priority is data extraction and validation.

The entire purpose of this component is to create a realistic time-varying freshwater forcing dataset that can later be integrated with the salinity and CO2 components of the project.

Use the EMP dataset as the main source.

Use precipitation and evaporation datasets only after the EMP workflow is complete.