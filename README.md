# Aid Allocation with Lateral Transshipment

## Pathway

Path A – Replication and Extension of a Research Paper

This project replicates and extends the model proposed in:

Azizi et al. (2021)
"Aid Allocation for Camp-Based and Urban Refugees with Uncertain Demand and Replenishments"

The original model studies optimal allocation of humanitarian aid
to refugee camps under uncertain demand and replenishment cycles.

## Project Goals

1. Replicate the allocation model using the provided parameter files.
2. Implement the piecewise linear MILP formulation described in the paper.
3. Extend the model by introducing lateral transshipment between camps.

## Model Overview

### Original Model

The original model determines the allocation level for each refugee camp in order to minimize the expected total cost of the system.  
The cost structure includes:

- deprivation cost from unmet internal demand
- referral cost from rejected external demand
- holding cost for remaining inventory

The expected cost function derived in the paper is nonlinear and is approximated using a **piecewise linear MILP formulation**.

### Extension

This project introduces **lateral transshipment between camps**.

Additional decision variables allow camps to transfer inventory to one another.  
This enables camps with surplus inventory to support camps experiencing shortages.

The extended model includes:

- allocation decisions
- transshipment decisions
- inventory balance constraints
- transshipment feasibility constraints

## Repository Structure

data/
    Input parameter files

src/
    Model implementation scripts

## Tools

Python  
Gurobi solver

## Installation

Install required packages:

pip install -r requirements.txt

## Current Status

This repository contains the **initial mathematical model implementation (Model v1)**.  
The piecewise linear expected cost formulation from the original paper will be integrated in later stages of the project.


Original link to paper:
Azizi, S., Bozkir, C. D. C., Trapp, A. C., Kundakcioglu, O. E., & Kurbanzade, A. K. (2021). Aid Allocation for Camp‐Based and Urban Refugees with Uncertain Demand and Replenishments. Production and Operations Management, 30(12), 4455-4474. https://doi.org/10.1111/poms.13531 (Original work published 2021)

