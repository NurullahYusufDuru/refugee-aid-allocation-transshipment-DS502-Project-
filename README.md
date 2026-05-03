# Humanitarian Aid Allocation with Binary Activation Decisions

This project studies a humanitarian aid allocation problem in which limited resources must be distributed across multiple refugee camps with both internal and external demand.

The key feature of the model is the inclusion of **binary activation decisions**, which represent whether a specific type of aid is delivered to a particular camp. This reflects the operational reality that delivering aid requires activating logistics such as transportation and coordination.

---

## Problem Description

We consider a setting with:
- Multiple refugee camps
- Multiple aid types
- Limited total supply
- Internal and external demand at each camp

The goal is to determine:
1. Which camp–aid pairs should be activated  
2. How much aid should be allocated  

while minimizing:
- Unmet internal demand (deprivation cost)
- Unmet external demand (referral cost)
- Activation costs

---

## Methodology

Two solution approaches are implemented:

### 1. Mixed-Integer Linear Programming (MILP)

- Formulated using decision variables for allocation and activation
- Solved using **Gurobi**
- Provides **optimal solutions**

---

### 2. Genetic Algorithm (GA)

- Heuristic approach focusing on **activation decisions**
- Each solution is a binary matrix \(y_{ik}\)
- Allocation decisions are computed using a **greedy procedure**
- Includes:
  - Selection (tournament)
  - Crossover (uniform)
  - Mutation (bit flip)
  - Repair mechanism for feasibility

---

## Computational Experiments

We conducted **10 experimental runs** with varying:
- Number of camps
- Number of aid types
- Constraint parameters (L, C, budget)

Results are stored in:
ds502_experiments.csv


Each run reports:
- Problem size
- Gurobi objective (optimal)
- GA objective
- Optimality gap
- Runtime

---

## Key Findings

- The MILP model solves all instances efficiently and provides optimal solutions.
- The GA produces feasible solutions but with an average gap of ~20%.
- The gap increases as problem size grows.

---

## How to Run

### Requirements

- Python 3.x
- Gurobi (with valid license)

  Author

Nurullah Yusuf Duru
Özyeğin University – Industrial Engineering

