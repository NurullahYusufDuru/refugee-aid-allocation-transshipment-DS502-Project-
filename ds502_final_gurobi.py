import numpy as np
from gurobipy import Model, GRB, quicksum
import time


def generate_random_data(
    num_camps=200,
    num_aid_types=40,
    seed=42
):
    np.random.seed(seed)

    camps = [f"Camp_{i}" for i in range(1, num_camps + 1)]
    aid_types = [f"Aid_{k}" for k in range(1, num_aid_types + 1)]

    internal_demand = {}
    external_demand = {}

    for i in camps:
        for k in aid_types:
            internal_demand[i, k] = np.random.randint(500, 8000)
            external_demand[i, k] = np.random.randint(500, 10000)

    total_supply = {}

    for k in aid_types:
        total_demand_k = sum(
            internal_demand[i, k] + external_demand[i, k]
            for i in camps
        )

        supply_ratio = np.random.uniform(0.55, 0.80)
        total_supply[k] = total_demand_k * supply_ratio

    leftover_penalty = {
        k: np.random.uniform(0.5, 3.0)
        for k in aid_types
    }

    referral_cost = {
        k: np.random.uniform(2.0, 15.0)
        for k in aid_types
    }

    deprivation_cost = {
        k: np.random.uniform(30.0, 120.0)
        for k in aid_types
    }

    fixed_activation_cost = {}

    for i in camps:
        for k in aid_types:
            fixed_activation_cost[i, k] = np.random.uniform(5000, 30000)

    return (
        camps,
        aid_types,
        internal_demand,
        external_demand,
        total_supply,
        leftover_penalty,
        referral_cost,
        deprivation_cost,
        fixed_activation_cost,
    )


def build_model(
    camps,
    aid_types,
    internal_demand,
    external_demand,
    total_supply,
    leftover_penalty,
    deprivation_cost,
    referral_cost,
    fixed_activation_cost,
    max_aid_types_per_camp=8,
    max_camps_per_aid_type=80,
    activation_budget_ratio=0.35,
):
    model = Model("MultiAidAllocation_CoupledBinaryActivation")
    model.ModelSense = GRB.MINIMIZE
    model.Params.OutputFlag = 1

    # Decision variables
    X = model.addVars(
        camps, aid_types,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="X"
    )

    u = model.addVars(
        camps, aid_types,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="u"
    )

    r = model.addVars(
        camps, aid_types,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="r"
    )

    l = model.addVars(
        camps, aid_types,
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="l"
    )

    y = model.addVars(
        camps, aid_types,
        vtype=GRB.BINARY,
        name="y"
    )

    # Objective function
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

    # 1. Supply constraint for each aid type
    model.addConstrs(
        (
            quicksum(X[i, k] for i in camps) <= total_supply[k]
            for k in aid_types
        ),
        name="supply"
    )

    # 2. Activation constraint
    model.addConstrs(
        (
            X[i, k] <= (
                internal_demand[i, k] + external_demand[i, k]
            ) * y[i, k]
            for i in camps
            for k in aid_types
        ),
        name="activation"
    )

    # 3. Internal unmet demand
    model.addConstrs(
        (
            u[i, k] >= internal_demand[i, k] - X[i, k]
            for i in camps
            for k in aid_types
        ),
        name="internal_shortage"
    )

    # 4. External rejected demand
    model.addConstrs(
        (
            r[i, k] >= (
                internal_demand[i, k]
                + external_demand[i, k]
                - X[i, k]
                - u[i, k]
            )
            for i in camps
            for k in aid_types
        ),
        name="external_rejection"
    )

    # 5. Leftover inventory / unused aid
    model.addConstrs(
        (
            l[i, k] >= (
                X[i, k]
                - internal_demand[i, k]
                - external_demand[i, k]
            )
            for i in camps
            for k in aid_types
        ),
        name="leftover_inventory"
    )

    # 6. Maximum number of aid types per camp
    model.addConstrs(
        (
            quicksum(y[i, k] for k in aid_types)
            <= max_aid_types_per_camp
            for i in camps
        ),
        name="max_aid_types_per_camp"
    )

    # 7. Maximum number of camps per aid type
    model.addConstrs(
        (
            quicksum(y[i, k] for i in camps)
            <= max_camps_per_aid_type
            for k in aid_types
        ),
        name="max_camps_per_aid_type"
    )

    # 8. Activation budget
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

    model.update()

    vars_dict = {
        "X": X,
        "u": u,
        "r": r,
        "l": l,
        "y": y,
    }

    return model, vars_dict


def solve_instance():
    (
        camps,
        aid_types,
        internal_demand,
        external_demand,
        total_supply,
        leftover_penalty,
        referral_cost,
        deprivation_cost,
        fixed_activation_cost,
    ) = generate_random_data(
        num_camps=200,
        num_aid_types=40,
        seed=42
    )

    model, vars_dict = build_model(
        camps=camps,
        aid_types=aid_types,
        internal_demand=internal_demand,
        external_demand=external_demand,
        total_supply=total_supply,
        leftover_penalty=leftover_penalty,
        deprivation_cost=deprivation_cost,
        referral_cost=referral_cost,
        fixed_activation_cost=fixed_activation_cost,
        max_aid_types_per_camp=8,
        max_camps_per_aid_type=80,
        activation_budget_ratio=0.35,
    )

    print("=" * 70)
    print("Problem Size")
    print("=" * 70)
    print(f"Number of camps: {len(camps)}")
    print(f"Number of aid types: {len(aid_types)}")
    print(f"Number of binary variables: {len(camps) * len(aid_types)}")
    print(f"Number of continuous variables: {4 * len(camps) * len(aid_types)}")
    print("=" * 70)

    start_time = time.time()
    model.optimize()
    end_time = time.time()

    runtime = end_time - start_time

    print("=" * 70)
    print("Model solved.")
    print(f"Status code: {model.Status}")
    print(f"Runtime: {runtime:.4f} seconds")

    if model.Status == GRB.OPTIMAL:
        print(f"Objective value: {model.ObjVal:.4f}")

        active_count = sum(
            int(round(vars_dict["y"][i, k].X))
            for i in camps
            for k in aid_types
        )

        print(f"Number of activated camp-aid pairs: {active_count}")

        print("\nSample Results")
        print("-" * 70)

        for i in camps[:5]:
            for k in aid_types[:5]:
                print(
                    f"{i}, {k}: "
                    f"y = {int(round(vars_dict['y'][i, k].X))}, "
                    f"X = {vars_dict['X'][i, k].X:.2f}, "
                    f"u = {vars_dict['u'][i, k].X:.2f}, "
                    f"r = {vars_dict['r'][i, k].X:.2f}, "
                    f"l = {vars_dict['l'][i, k].X:.2f}"
                )

    elif model.Status == GRB.INFEASIBLE:
        print("Model is infeasible.")

    else:
        print("No optimal solution found.")

    return model, vars_dict


if __name__ == "__main__":
    solve_instance()