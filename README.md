# Aid Allocation with Lateral Transshipment

This project replicates and extends the model proposed in:

Azizi et al. (2021)
"Aid Allocation for Camp-Based and Urban Refugees with Uncertain Demand and Replenishments"

The original model studies optimal allocation of humanitarian aid
to refugee camps under uncertain demand and replenishment cycles.

## Project Goals

1. Replicate the allocation model using the provided parameter files.
2. Implement the piecewise linear MILP formulation described in the paper.
3. Extend the model by introducing lateral transshipment between camps.

## Repository Structure

data/
    Input parameter files

src/
    Model implementation scripts

## Extension

The extension introduces inventory transfers between camps
to reduce shortages and improve resource utilization.

## Tools

Python  
Gurobi solver

## Installation

Install required packages:

pip install -r requirements.txt

Original link to paper:
Azizi, S., Bozkir, C. D. C., Trapp, A. C., Kundakcioglu, O. E., & Kurbanzade, A. K. (2021). Aid Allocation for Camp‐Based and Urban Refugees with Uncertain Demand and Replenishments. Production and Operations Management, 30(12), 4455-4474. https://doi.org/10.1111/poms.13531 (Original work published 2021)

