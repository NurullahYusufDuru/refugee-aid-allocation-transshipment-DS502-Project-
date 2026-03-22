# Humanitarian Aid Allocation Model

## Overview

This project implements a simplified optimization model for allocating humanitarian aid across multiple refugee camps.

The model is based on the paper:

> *Azizi et al. (2021) – Aid Allocation for Camp-Based and Urban Refugees with Uncertain Demand and Replenishments*

The goal is to determine how a central decision-maker should distribute limited inventory to minimize the total cost associated with unmet demand and inventory holding.

---

## Model Description

The model considers:

- **Internal demand** (camp-based refugees)
- **External demand** (urban refugees)

A key assumption in this implementation is that:

> Internal demand is prioritized over external demand.

---

## Decision Variables

- `X[i]`: order up to allocation level to camp *i*
- `u[i]`: unmet internal demand (auxillary)
- `r[i]`: rejected external demand (auxillary)
- `l[i]`: leftover inventory (auxillary)

---

## Objective Function

The model minimizes total cost:

- Deprivation cost (internal unmet demand)
- Referral cost (external unmet demand)
- Holding cost (leftover inventory)

---


## Data

The model uses two input files:

- `data/rates.csv`  
  Contains internal and external demand rates for each camp.

- `data/parameters.csv`  
  Contains cost parameters:
  - holding cost
  - deprivation cost
  - referral cost

---
## Example Results

For a test instance with total supply = 100,000:

Objective value: 36,635.52
Runtime: 0.0010 seconds

Key observations:

-All internal demand is satisfied
-Some external demand remains unmet
-No leftover inventory (all supply is used)

---

Original link to paper:
Azizi, S., Bozkir, C. D. C., Trapp, A. C., Kundakcioglu, O. E., & Kurbanzade, A. K. (2021). Aid Allocation for Camp‐Based and Urban Refugees with Uncertain Demand and Replenishments. Production and Operations Management, 30(12), 4455-4474. https://doi.org/10.1111/poms.13531 (Original work published 2021)

