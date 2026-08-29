I went through the paper and extracted the equations and parameter values you need.

### 1. Thermohaline Flow Equation (Density-driven flow)

This is Eq. (2), later simplified to Eq. (3).

**Density of seawater:**

$$
\rho(T,S) = \rho_0 \left(1 - \alpha T + \beta S\right)
$$

where  
- $\rho_0 = 1.029\ \mathrm{g/cm^3}$  
- $\alpha = 2.07 \times 10^{-4}\ ^\circ\mathrm{C}^{-1}$  
- $\beta = 7.5 \times 10^{-4}\ \mathrm{psu}^{-1}$

**General density-gradient flow:**

$$
q_{ij} = K_{ij} \frac{\rho_i - \rho_j}{\rho_0} \approx K_{ij} \left[ \alpha (T_j - T_i) + \beta (S_i - S_j) \right]
$$

Since deep water temperature is assumed constant, the paper simplifies this to:

$$
\boxed{q_{ij} = K_{ij} \beta (S_i - S_j)}
$$

where $K_{ij} \approx 60\ \mathrm{Sv}$.

---

### 2. Salinity ODE (Eq. 4)

For each ocean box:

$$
\boxed{
\frac{dS_i}{dt} = 
\frac{(F_i + b_{NA}\delta_{i1} + b_{SO}\delta_{i3} + q_{Amaz}\delta_{i1})S_0}{V_i}
+ \sum_{j \in I_i} \left( \frac{q_{ji}}{V_i}S_j + \frac{Q_{ji}}{V_i}S_j \right)
- \sum_{j \in O_i} \left( \frac{q_{ij}}{V_i}S_i + \frac{Q_{ij}}{V_i}S_i \right)
}
$$

where  
- $F_i$ = freshwater flux  
- $b_{NA}$ = Arctic melt  
- $b_{SO}$ = Antarctic melt  
- $q_{Amaz}$ = Amazon discharge  
- $Q_{ij}$ = surface currents  
- $q_{ij}$ = deep currents  
- $S_0 = 35\ \mathrm{psu}$

---

### 3. CO₂ ODE (Eq. 7)

The aqueous CO₂ equation is:

$$
\boxed{
\begin{aligned}
\frac{dC_i}{dt} &= 
\frac{\gamma_i A_i}{V_i} (K_0 pCO_2 - C_i) 
+ \delta_{i1} \frac{q_{Amaz} C_{Amaz}}{V_i} \\
&\quad - K_c C_i + K_n N_i \\
&\quad + \sum_{j \in I_i} \left( \frac{q_{ji}}{V_i} C_j' + \frac{Q_{ji}}{V_i} C_j \right) \\
&\quad - \sum_{j \in O_i} \left( \frac{q_{ij}}{V_i} C_i' + \frac{Q_{ij}}{V_i} C_i \right)
\end{aligned}
}
$$

Reaction products satisfy:

$$
\boxed{
\frac{dN_i}{dt} = \delta_{i1} \frac{q_{Amaz} N_{Amaz}}{V_i} + K_c C_i - K_n N_i + \text{transport terms}
}
$$

and

$$
\boxed{\frac{dN_i}{dt} = K_{eq} \frac{dC_i}{dt}}
$$

---

### 4. Deep-water CO₂ correction

The paper replaces $C_i$ in deep-water transport by:

$$
C_i' = C_i \left[
(1 + a - bT) 
+ (gT + l) \frac{\bar{M}}{\rho} C_i 
+ (hT - \psi) \left( \frac{\bar{M}}{\rho} \right)^2 C_i^2
\right]
$$

---

### 5. Initial Conditions (Table I)

| Ocean            | Initial CO₂ ($C_i$) (mol/m³) | Initial Salinity ($S_i$) (psu) |
|------------------|------------------------------|--------------------------------|
| North Atlantic   | 0.01413                      | 34.912                         |
| South Atlantic   | 0.01579                      | 35.435                         |
| Southern Ocean   | 0.01400                      | 34.427                         |
| Pacific          | 0.01834                      | 34.668                         |
| Indian           | 0.01671                      | 34.538                         |

---

### 6. Ocean Volumes & Areas (Appendix A)

| Ocean            | Volume ($V_i$) (m³)       | Area ($A_i$) (m²)         |
|------------------|---------------------------|---------------------------|
| North Atlantic   | $146 \times 10^{15}$      | $41.49 \times 10^{12}$    |
| South Atlantic   | $160 \times 10^{15}$      | $40.27 \times 10^{12}$    |
| Pacific          | $710 \times 10^{15}$      | $165.25 \times 10^{12}$   |
| Indian           | $264 \times 10^{15}$      | $70.56 \times 10^{12}$    |
| Southern         | $71.8 \times 10^{15}$     | $20.33 \times 10^{12}$    |

---

### 7. Important Constants

| Parameter                        | Value                              |
|----------------------------------|------------------------------------|
| Reference salinity ($S_0$)       | 35 psu                             |
| Density ($\rho_0$)               | 1.029 g/cm³                        |
| Thermal expansion ($\alpha$)     | $2.07 \times 10^{-4}\ ^\circ\mathrm{C}^{-1}$ |
| Haline contraction ($\beta$)     | $7.5 \times 10^{-4}\ \mathrm{psu}^{-1}$ |
| Deep-current coefficient ($K_{ij}$) | 60 Sv                           |
| Amazon discharge ($q_{Amaz}$)    | 0.24 Sv                            |
| North Atlantic ice melt ($b_{NA}$)| $3.28 \times 10^3\ \mathrm{m^3/s}$ |
| Southern Ocean ice term ($b_{SO}$)| $-6.944 \times 10^2\ \mathrm{m^3/s}$ |
| Reaction rate ($K_c$)            | 0.037 s⁻¹                          |
| Reaction rate ($K_n$)            | 23 s⁻¹                             |
| Hydration equilibrium ($K_{eq}$) | 0.0011                             |

---

### 8. Surface Current Values

| Flow     | Value |
|----------|-------|
| $Q_{45}$ | 15 Sv |
| $Q_{21}$ | 26 Sv |
| $Q_{53}$ | 60 Sv |
| $Q_{52}$ | 10 Sv |
| $Q_{32}$ | 30 Sv |

These are the complete governing equations and numerical parameters you'll need to reproduce or extend the five-box model.