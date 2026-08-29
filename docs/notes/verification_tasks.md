# Objective

Perform a rigorous scientific audit of the freshwater-flux analysis and determine whether the remaining discrepancies arise from:

1. Basin area mismatches
2. Basin boundary mismatches
3. Sign convention inconsistencies
4. Unit conversion errors
5. Differences between our implementation and the methodology used in Sunny et al. (2023)

The goal is to either:

* Fully validate the current results, OR
* Identify the precise source of the remaining Indo-Pacific discrepancy.

---

# Context

The project reproduces the freshwater flux calculations from:

"The Ocean Carbon Sinks and Climate Change" (Chaos, 2023)

using HOAPS 4.0 freshwater flux (budg) data.

We have already:

* Downloaded HOAPS data (2000–2014)
* Built ocean masks
* Converted fluxes to Sverdrups
* Reproduced North Atlantic and South Atlantic values with high accuracy
* Explained the Southern Ocean discrepancy through ice masking

Remaining issue:

| Basin   | Paper     | Observed  |
| ------- | --------- | --------- |
| Pacific | 0.064 Sv  | 0.6215 Sv |
| Indian  | 2.3495 Sv | 0.7641 Sv |

Need to determine whether the discrepancy comes from our implementation or the paper.

---

# Required Tasks

## Task 1: Verify Basin Areas

Find the exact computed area of each mask currently used:

* North Atlantic
* South Atlantic
* Pacific
* Indian
* Southern Ocean

Report:

| Basin | Computed Area | Paper Area | Difference (%) |

Paper areas:

NA = 41.49 × 10¹² m²
SA = 40.27 × 10¹² m²
PO = 165.25 × 10¹² m²
IO = 70.56 × 10¹² m²
SO = 20.33 × 10¹² m²

Determine whether any basin differs by more than 5%.

If so:

* identify exactly where area is being lost or gained
* visualize the mismatch

---

## Task 2: Audit Ocean Boundaries

Compare our implemented masks against Appendix A of the paper.

Specifically inspect:

### Atlantic boundaries

Paper:

* South America boundary at 70°W
* Africa boundary at 20°E

Verify implementation.

---

### Indian-Pacific separation

Paper states:

* Indonesian Archipelago north of Australia
* Tasman Sea (140°E) south of Australia

Audit whether our implementation matches this.

Pay special attention to:

* Indonesia
* Java Sea
* South China Sea
* Arafura Sea
* Timor Sea
* Coral Sea
* Tasman region

Create visual overlays.

Determine how much area changes when using alternative interpretations.

---

## Task 3: Verify HOAPS Sign Convention

Using:

* HOAPS documentation
* precipitation dataset
* evaporation dataset

Verify numerically:

budg ≈ evap − precip

or

budg ≈ precip − evap

Perform this check using actual data.

Do not rely only on documentation.

Compute:

budg - (evap - precip)

and

budg - (precip - evap)

for sample months.

Report numerical error.

Determine conclusively which convention is correct.

---

## Task 4: Verify Reproduction of Paper's Methodology

The paper states:

Freshwater Flux = (Precipitation − Evaporation) × Ocean Area

Determine whether our code reproduces exactly this methodology.

Check:

* area calculation
* unit conversions
* temporal averaging
* handling of missing values
* handling of ice-covered cells
* ocean mask definitions

Identify any deviations.

---

## Task 5: Independent Recalculation

Using only:

* HOAPS precipitation
* HOAPS evaporation

Ignore budg entirely.

Compute:

P − E

for each basin.

Generate independent freshwater flux estimates.

Compare:

1. P−E calculation
2. budg-based calculation
3. Paper constants

Determine whether all three agree.

---

## Task 6: Investigate Indo-Pacific Discrepancy

Determine whether any of the following explain the discrepancy:

### Hypothesis A

Basin masks are incorrect.

### Hypothesis B

Paper used different ocean areas.

### Hypothesis C

Paper used different dataset version.

### Hypothesis D

Paper used climatological averages from literature rather than direct HOAPS computation.

### Hypothesis E

Paper contains a sign inconsistency.

### Hypothesis F

Paper contains a numerical or transcription error.

For each hypothesis:

* gather evidence
* quantify support
* assign confidence level

---

## Task 7: Challenge the Current Report

Do not assume the report is correct.

Act as a hostile reviewer.

Attempt to falsify:

* the Southern Ocean explanation
* the sign-convention argument
* the parameterization discrepancy argument

Identify weaknesses.

---

# Deliverables

Produce:

1. Technical audit report
2. Basin area comparison table
3. Basin boundary visualizations
4. Sign-convention verification
5. Independent recalculation from PRE and EVA
6. Root-cause analysis of Indo-Pacific discrepancy
7. Final confidence assessment

Final section must answer:

"Based on all available evidence, how likely is it that the discrepancy originates from our implementation versus the original paper?"

Provide confidence percentages.
