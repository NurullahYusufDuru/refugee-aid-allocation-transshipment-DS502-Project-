import time
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum


# ============================================================
# 1. Shared Random Data Generation
# ============================================================

def generate_shared_random_data(num_camps, num_aid_types, seed):
    """
    Generates one common random instance.
    The same instance is used by both Gurobi and GA.
    """

    np.random.seed(seed)

    camps = [f"Camp_{i}" for i in range(1, num_camps + 1)]
    aid_types = [f"Aid_{k}" for k in range(1, num_aid_types + 1)]

    internal_array = np.random.randint(500, 8000, size=(num_camps, num_aid_types))
    external_array = np.random.randint(500, 10000, size=(num_camps, num_aid_types))
    total_demand_array = internal_array + external_array

    total_supply_array = np.zeros(num_aid_types)

    for k in range(num_aid_types):
        total_supply_array[k] = np.sum(total_demand_array[:, k]) * np.random.uniform(0.55, 0.80)

    leftover_penalty_array = np.random.uniform(0.5, 3.0, size=num_aid_types)
    referral_cost_array = np.random.uniform(2.0, 15.0, size=num_aid_types)
    deprivation_cost_array = np.random.uniform(30.0, 120.0, size=num_aid_types)

    fixed_activation_cost_array = np.random.uniform(
        5000, 30000, size=(num_camps, num_aid_types)
    )

    # Dictionary version for Gurobi
    internal_demand = {}
    external_demand = {}
    fixed_activation_cost = {}

    for i_idx, i in enumerate(camps):
        for k_idx, k in enumerate(aid_types):
            internal_demand[i, k] = float(internal_array[i_idx, k_idx])
            external_demand[i, k] = float(external_array[i_idx, k_idx])
            fixed_activation_cost[i, k] = float(fixed_activation_cost_array[i_idx, k_idx])

    total_supply = {
        k: float(total_supply_array[k_idx])
        for k_idx, k in enumerate(aid_types)
    }

    leftover_penalty = {
        k: float(leftover_penalty_array[k_idx])
        for k_idx, k in enumerate(aid_types)
    }

    referral_cost = {
        k: float(referral_cost_array[k_idx])
        for k_idx, k in enumerate(aid_types)
    }

    deprivation_cost = {
        k: float(deprivation_cost_array[k_idx])
        for k_idx, k in enumerate(aid_types)
    }

    gurobi_data = {
        "camps": camps,
        "aid_types": aid_types,
        "internal_demand": internal_demand,
        "external_demand": external_demand,
        "total_supply": total_supply,
        "leftover_penalty": leftover_penalty,
        "referral_cost": referral_cost,
        "deprivation_cost": deprivation_cost,
        "fixed_activation_cost": fixed_activation_cost,
    }

    # Array version for GA
    ga_data = {
        "camps": camps,
        "aid_types": aid_types,
        "internal_demand": internal_array,
        "external_demand": external_array,
        "total_demand": total_demand_array,
        "total_supply": total_supply_array,
        "leftover_penalty": leftover_penalty_array,
        "referral_cost": referral_cost_array,
        "deprivation_cost": deprivation_cost_array,
        "fixed_activation_cost": fixed_activation_cost_array,
    }

    return gurobi_data, ga_data


# ============================================================
# 2. Gurobi Model
# ============================================================

def solve_gurobi(
    data,
    max_aid_types_per_camp,
    max_camps_per_aid_type,
    activation_budget_ratio,
    output_flag=0
):
    camps = data["camps"]
    aid_types = data["aid_types"]
    internal_demand = data["internal_demand"]
    external_demand = data["external_demand"]
    total_supply = data["total_supply"]
    leftover_penalty = data["leftover_penalty"]
    referral_cost = data["referral_cost"]
    deprivation_cost = data["deprivation_cost"]
    fixed_activation_cost = data["fixed_activation_cost"]

    model = Model("MultiAidAllocation_CoupledBinaryActivation")
    model.ModelSense = GRB.MINIMIZE
    model.Params.OutputFlag = output_flag

    X = model.addVars(camps, aid_types, lb=0.0, vtype=GRB.CONTINUOUS, name="X")
    u = model.addVars(camps, aid_types, lb=0.0, vtype=GRB.CONTINUOUS, name="u")
    r = model.addVars(camps, aid_types, lb=0.0, vtype=GRB.CONTINUOUS, name="r")
    l = model.addVars(camps, aid_types, lb=0.0, vtype=GRB.CONTINUOUS, name="l")
    y = model.addVars(camps, aid_types, vtype=GRB.BINARY, name="y")

    model.setObjective(
        quicksum(
            deprivation_cost[k] * u[i, k]
            + referral_cost[k] * r[i, k]
            + leftover_penalty[k] * l[i, k]
            + fixed_activation_cost[i, k] * y[i, k]
            for i in camps
            for k in aid_types
        ),
        GRB.MINIMIZE
    )

    model.addConstrs(
        (
            quicksum(X[i, k] for i in camps) <= total_supply[k]
            for k in aid_types
        ),
        name="supply"
    )

    model.addConstrs(
        (
            X[i, k] <= (internal_demand[i, k] + external_demand[i, k]) * y[i, k]
            for i in camps
            for k in aid_types
        ),
        name="activation"
    )

    model.addConstrs(
        (
            u[i, k] >= internal_demand[i, k] - X[i, k]
            for i in camps
            for k in aid_types
        ),
        name="internal_shortage"
    )

    model.addConstrs(
        (
            r[i, k] >= internal_demand[i, k] + external_demand[i, k] - X[i, k] - u[i, k]
            for i in camps
            for k in aid_types
        ),
        name="external_rejection"
    )

    model.addConstrs(
        (
            l[i, k] >= X[i, k] - internal_demand[i, k] - external_demand[i, k]
            for i in camps
            for k in aid_types
        ),
        name="leftover_inventory"
    )

    model.addConstrs(
        (
            quicksum(y[i, k] for k in aid_types) <= max_aid_types_per_camp
            for i in camps
        ),
        name="max_aid_types_per_camp"
    )

    model.addConstrs(
        (
            quicksum(y[i, k] for i in camps) <= max_camps_per_aid_type
            for k in aid_types
        ),
        name="max_camps_per_aid_type"
    )

    activation_budget = activation_budget_ratio * sum(
        fixed_activation_cost[i, k]
        for i in camps
        for k in aid_types
    )

    model.addConstr(
        quicksum(
            fixed_activation_cost[i, k] * y[i, k]
            for i in camps
            for k in aid_types
        ) <= activation_budget,
        name="activation_budget"
    )

    start = time.time()
    model.optimize()
    runtime = time.time() - start

    if model.Status == GRB.OPTIMAL:
        active_pairs = sum(
            int(round(y[i, k].X))
            for i in camps
            for k in aid_types
        )

        return {
            "status": "OPTIMAL",
            "objective": model.ObjVal,
            "runtime": runtime,
            "active_pairs": active_pairs,
        }

    return {
        "status": str(model.Status),
        "objective": np.nan,
        "runtime": runtime,
        "active_pairs": np.nan,
    }


# ============================================================
# 3. GA Helper Functions
# ============================================================

def repair_solution(
    y,
    fixed_activation_cost,
    max_aid_types_per_camp,
    max_camps_per_aid_type,
    activation_budget
):
    y = y.copy()
    num_camps, num_aid_types = y.shape

    for i in range(num_camps):
        active = np.where(y[i, :] == 1)[0]

        if len(active) > max_aid_types_per_camp:
            costs = fixed_activation_cost[i, active]
            remove_count = len(active) - max_aid_types_per_camp
            remove_indices = active[np.argsort(costs)[-remove_count:]]
            y[i, remove_indices] = 0

    for k in range(num_aid_types):
        active = np.where(y[:, k] == 1)[0]

        if len(active) > max_camps_per_aid_type:
            costs = fixed_activation_cost[active, k]
            remove_count = len(active) - max_camps_per_aid_type
            remove_indices = active[np.argsort(costs)[-remove_count:]]
            y[remove_indices, k] = 0

    total_activation_cost = np.sum(fixed_activation_cost * y)

    while total_activation_cost > activation_budget and np.sum(y) > 0:
        active_pairs = np.argwhere(y == 1)

        pair_costs = np.array([
            fixed_activation_cost[i, k]
            for i, k in active_pairs
        ])

        remove_pair = active_pairs[np.argmax(pair_costs)]
        y[remove_pair[0], remove_pair[1]] = 0

        total_activation_cost = np.sum(fixed_activation_cost * y)

    return y


def evaluate_solution(y, data):
    internal = data["internal_demand"]
    external = data["external_demand"]
    total_demand = data["total_demand"]
    total_supply = data["total_supply"]

    deprivation_cost = data["deprivation_cost"]
    referral_cost = data["referral_cost"]
    leftover_penalty = data["leftover_penalty"]
    fixed_activation_cost = data["fixed_activation_cost"]

    num_camps, num_aid_types = y.shape

    X = np.zeros((num_camps, num_aid_types))

    for k in range(num_aid_types):
        active_camps = np.where(y[:, k] == 1)[0]
        remaining_supply = total_supply[k]

        if len(active_camps) == 0:
            continue

        priority = (
            deprivation_cost[k] * internal[active_camps, k]
            + referral_cost[k] * external[active_camps, k]
        )

        order = active_camps[np.argsort(priority)[::-1]]

        for i in order:
            allocation = min(total_demand[i, k], remaining_supply)
            X[i, k] = allocation
            remaining_supply -= allocation

            if remaining_supply <= 1e-9:
                break

    u = np.maximum(internal - X, 0)
    r = np.maximum(internal + external - X - u, 0)
    l = np.maximum(X - internal - external, 0)

    objective = (
        np.sum(deprivation_cost.reshape(1, -1) * u)
        + np.sum(referral_cost.reshape(1, -1) * r)
        + np.sum(leftover_penalty.reshape(1, -1) * l)
        + np.sum(fixed_activation_cost * y)
    )

    return objective, X, u, r, l


def create_initial_solution(
    num_camps,
    num_aid_types,
    activation_probability,
    fixed_activation_cost,
    max_aid_types_per_camp,
    max_camps_per_aid_type,
    activation_budget
):
    y = (np.random.rand(num_camps, num_aid_types) < activation_probability).astype(int)

    return repair_solution(
        y,
        fixed_activation_cost,
        max_aid_types_per_camp,
        max_camps_per_aid_type,
        activation_budget
    )


def solve_ga(
    data,
    max_aid_types_per_camp,
    max_camps_per_aid_type,
    activation_budget_ratio,
    n_pop=80,
    n_gen=300,
    n_crossover=40,
    n_mutation=40,
    tournament_size=5,
    activation_probability=0.15,
    mutation_probability=0.02,
    seed=42
):
    np.random.seed(seed)

    num_camps = len(data["camps"])
    num_aid_types = len(data["aid_types"])
    fixed_activation_cost = data["fixed_activation_cost"]
    activation_budget = activation_budget_ratio * np.sum(fixed_activation_cost)

    population = []

    start = time.time()

    for _ in range(n_pop):
        y = create_initial_solution(
            num_camps,
            num_aid_types,
            activation_probability,
            fixed_activation_cost,
            max_aid_types_per_camp,
            max_camps_per_aid_type,
            activation_budget
        )

        obj, _, _, _, _ = evaluate_solution(y, data)
        population.append((y, obj))

    population = sorted(population, key=lambda x: x[1])

    best_y = population[0][0].copy()
    best_obj = population[0][1]

    for gen in range(1, n_gen + 1):
        offspring = []

        for _ in range(n_crossover):
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            selected = sorted([population[idx] for idx in tournament], key=lambda x: x[1])

            parent1 = selected[0][0]
            parent2 = selected[1][0]

            mask = np.random.randint(0, 2, size=parent1.shape)

            child1 = np.where(mask == 1, parent1, parent2)
            child2 = np.where(mask == 1, parent2, parent1)

            child1 = repair_solution(
                child1,
                fixed_activation_cost,
                max_aid_types_per_camp,
                max_camps_per_aid_type,
                activation_budget
            )

            child2 = repair_solution(
                child2,
                fixed_activation_cost,
                max_aid_types_per_camp,
                max_camps_per_aid_type,
                activation_budget
            )

            obj1, _, _, _, _ = evaluate_solution(child1, data)
            obj2, _, _, _, _ = evaluate_solution(child2, data)

            offspring.append((child1, obj1))
            offspring.append((child2, obj2))

        for _ in range(n_mutation):
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            selected = sorted([population[idx] for idx in tournament], key=lambda x: x[1])

            parent = selected[0][0]
            child = parent.copy()

            mutation_mask = (
                np.random.rand(num_camps, num_aid_types) < mutation_probability
            )

            child[mutation_mask] = 1 - child[mutation_mask]

            child = repair_solution(
                child,
                fixed_activation_cost,
                max_aid_types_per_camp,
                max_camps_per_aid_type,
                activation_budget
            )

            obj, _, _, _, _ = evaluate_solution(child, data)
            offspring.append((child, obj))

        population.extend(offspring)
        population = sorted(population, key=lambda x: x[1])
        population = population[:n_pop]

        if population[0][1] < best_obj:
            best_y = population[0][0].copy()
            best_obj = population[0][1]

    runtime = time.time() - start

    final_obj, X, u, r, l = evaluate_solution(best_y, data)

    return {
        "objective": final_obj,
        "runtime": runtime,
        "active_pairs": int(np.sum(best_y)),
    }


# ============================================================
# 4. Experiment Runner
# ============================================================

def run_experiments():
    experiments = [
        {"run": 1, "num_camps": 50,  "num_aid_types": 10, "L": 3, "C": 20, "budget_ratio": 0.25, "seed": 101},
        {"run": 2, "num_camps": 50,  "num_aid_types": 15, "L": 4, "C": 25, "budget_ratio": 0.30, "seed": 102},
        {"run": 3, "num_camps": 75,  "num_aid_types": 15, "L": 4, "C": 30, "budget_ratio": 0.30, "seed": 103},
        {"run": 4, "num_camps": 100, "num_aid_types": 20, "L": 5, "C": 40, "budget_ratio": 0.30, "seed": 104},
        {"run": 5, "num_camps": 100, "num_aid_types": 25, "L": 5, "C": 45, "budget_ratio": 0.35, "seed": 105},
        {"run": 6, "num_camps": 150, "num_aid_types": 25, "L": 6, "C": 60, "budget_ratio": 0.35, "seed": 106},
        {"run": 7, "num_camps": 150, "num_aid_types": 30, "L": 6, "C": 65, "budget_ratio": 0.35, "seed": 107},
        {"run": 8, "num_camps": 200, "num_aid_types": 30, "L": 7, "C": 70, "budget_ratio": 0.35, "seed": 108},
        {"run": 9, "num_camps": 200, "num_aid_types": 40, "L": 8, "C": 80, "budget_ratio": 0.35, "seed": 109},
        {"run": 10, "num_camps": 250, "num_aid_types": 40, "L": 8, "C": 90, "budget_ratio": 0.40, "seed": 110},
    ]

    results = []

    for exp in experiments:
        print("=" * 80)
        print(f"Running experiment {exp['run']}")
        print(exp)

        gurobi_data, ga_data = generate_shared_random_data(
            num_camps=exp["num_camps"],
            num_aid_types=exp["num_aid_types"],
            seed=exp["seed"]
        )

        gurobi_res = solve_gurobi(
            data=gurobi_data,
            max_aid_types_per_camp=exp["L"],
            max_camps_per_aid_type=exp["C"],
            activation_budget_ratio=exp["budget_ratio"],
            output_flag=0
        )

        ga_res = solve_ga(
            data=ga_data,
            max_aid_types_per_camp=exp["L"],
            max_camps_per_aid_type=exp["C"],
            activation_budget_ratio=exp["budget_ratio"],
            n_pop=80,
            n_gen=300,
            n_crossover=40,
            n_mutation=40,
            tournament_size=5,
            activation_probability=0.15,
            mutation_probability=0.02,
            seed=exp["seed"] + 1000
        )

        if gurobi_res["status"] == "OPTIMAL":
            gap_percent = (
                (ga_res["objective"] - gurobi_res["objective"])
                / gurobi_res["objective"]
            ) * 100
        else:
            gap_percent = np.nan

        row = {
            "run": exp["run"],
            "seed": exp["seed"],
            "num_camps": exp["num_camps"],
            "num_aid_types": exp["num_aid_types"],
            "binary_variables": exp["num_camps"] * exp["num_aid_types"],
            "continuous_variables": 4 * exp["num_camps"] * exp["num_aid_types"],
            "max_aid_types_per_camp_L": exp["L"],
            "max_camps_per_aid_type_C": exp["C"],
            "activation_budget_ratio": exp["budget_ratio"],
            "gurobi_status": gurobi_res["status"],
            "gurobi_objective": gurobi_res["objective"],
            "gurobi_runtime_sec": gurobi_res["runtime"],
            "gurobi_active_pairs": gurobi_res["active_pairs"],
            "ga_objective": ga_res["objective"],
            "ga_runtime_sec": ga_res["runtime"],
            "ga_active_pairs": ga_res["active_pairs"],
            "gap_percent": gap_percent,
        }

        results.append(row)

        print(
            f"Run {exp['run']} completed | "
            f"Gurobi Obj: {gurobi_res['objective']:.4f} | "
            f"GA Obj: {ga_res['objective']:.4f} | "
            f"Gap: {gap_percent:.2f}%"
        )

    df = pd.DataFrame(results)
    df.to_csv("ds502_experiments.csv", index=False)

    print("=" * 80)
    print("All experiments completed.")
    print("Saved results to ds502_experiments.csv")
    print(df)

    return df


if __name__ == "__main__":
    run_experiments()

