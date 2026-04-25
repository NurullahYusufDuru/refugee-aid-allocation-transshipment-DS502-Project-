# Humanitarian Aid Allocation 

## Project Overview

This project studies the allocation of limited humanitarian aid across multiple refugee camps. 
Each camp has internal demand (camp-based refugees) and external demand (urban refugees). 
The goal is to allocate available aid efficiently while minimizing humanitarian costs.

## Original Model

The original formulation is a deterministic, single-period optimization model. 
The objective minimizes:
- Deprivation cost (unmet internal demand)
- Referral cost (unmet external demand)
- Holding cost (unused inventory)

## MDP Reformulation

In Deliverable 6, the problem is reformulated as a Markov Decision Process (MDP).

### State

The state includes:
- Total available supply
- Camp-level inventory
- Internal demand
- External demand

### Action

The action is the allocation decision:
- How much aid to send to each camp

### Transition

- Inventory evolves based on allocation and demand
- Supply decreases over time (no replenishment)
- Demand is deterministic

### Cost

The stage cost includes:
- Internal unmet demand penalty (high priority)
- External unmet demand penalty
- Holding cost

### Policy

A policy defines the allocation decision for each state.

## Key Contribution

The MDP formulation enables:
- Sequential decision making
- Multi-period planning
- Extension to stochastic demand
- Integration of fairness objectives

## Planned Experiments

We will evaluate the model by varying:
- Supply levels
- Demand patterns
- Cost parameters
- Number of camps

Performance metrics:
- Total cost
- Unmet demand
- Inventory utilization




