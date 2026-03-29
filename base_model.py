import pandas as pd
from gurobipy import Model, GRB, quicksum
import time

def load_input_data():
    rates = pd.read_csv("data/rates.csv")
    parameters = pd.read_csv("data/parameters.csv")

    camps = rates["Camp Name"].tolist()

    lambda_c = {
        row["Camp Name"]: float(row["Internal Rate of Arrival (/yr)"])
        for _, row in rates.iterrows()
    }

    lambda_u = {
        row["Camp Name"]: float(row["External Rate of Arrival (/yr)"])
        for _, row in rates.iterrows()
    }

    params = {
        "holding_cost": float(parameters.loc[0, "Annual Holding Cost Per Item"]),
        "referral_cost": float(parameters.loc[0, "Referral Cost Per External Demand"]),
        "deprivation_cost": float(parameters.loc[0, "Deprivation Cost Per Time Unit Per Internal Demand"]),
    }

    return camps, lambda_c, lambda_u, params


def build_model(
    camps,
    lambda_c,
    lambda_u,
    total_supply,
    initial_inventory=None,
    holding_cost=1.0,
    deprivation_cost=10.0,
    referral_cost=2.0,
):
    """
    

    Decision variables
    ------------------
    X[i] : allocation to camp i
    u[i] : unmet internal demand
    r[i] : rejected external demand
    l[i] : leftover inventory
    """

    if initial_inventory is None:
        initial_inventory = {i: 0.0 for i in camps}

    model = Model("AidAllocation_InternalPriority")
    model.ModelSense = GRB.MINIMIZE
    model.Params.OutputFlag = 0

    # Decision variables
    X = model.addVars(camps, lb=0.0, vtype=GRB.CONTINUOUS, name="X")

    # Auxiliary variables
    u = model.addVars(camps, lb=0.0, vtype=GRB.CONTINUOUS, name="u")
    r = model.addVars(camps, lb=0.0, vtype=GRB.CONTINUOUS, name="r")
    l = model.addVars(camps, lb=0.0, vtype=GRB.CONTINUOUS, name="l")

    # Objective
    model.setObjective(
        quicksum(deprivation_cost * u[i] for i in camps)
        + quicksum(referral_cost * r[i] for i in camps)
        + quicksum(holding_cost * l[i] for i in camps),
        GRB.MINIMIZE
    )

    # 1. Total supply
    model.addConstr(
        quicksum(X[i] - initial_inventory[i] for i in camps) <= total_supply,
        name="total_supply"
    )

    # 2. Allocation feasibility
    model.addConstrs(
        (X[i] >= initial_inventory[i] for i in camps),
        name="inventory_feasibility"
    )

    # 3. Internal unmet demand
    
    model.addConstrs(
        (u[i] >= lambda_c[i] - X[i] for i in camps),
        name="internal_shortage"
    )

    # 4. External unmet demand
    
    model.addConstrs(
        (r[i] >= lambda_c[i] + lambda_u[i] - X[i] - u[i] for i in camps),
        name="external_rejection"
    )

    # 5. Leftover inventory
    # l[i] = max(X[i] - lambda_c[i] - lambda_u[i], 0)
    model.addConstrs(
        (l[i] >= X[i] - lambda_c[i] - lambda_u[i] for i in camps),
        name="leftover_inventory"
    )

    model.update()

    vars_dict = {
        "X": X,
        "u": u,
        "r": r,
        "l": l,
    }

    return model, vars_dict


def solve_small_instance(total_supply=100000):
    camps, lambda_c, lambda_u, params = load_input_data()

    initial_inventory = {i: 0.0 for i in camps}

    model, vars_dict = build_model(
        camps=camps,
        lambda_c=lambda_c,
        lambda_u=lambda_u,
        total_supply=total_supply,
        initial_inventory=initial_inventory,
        holding_cost=params["holding_cost"],
        deprivation_cost=params["deprivation_cost"],
        referral_cost=params["referral_cost"],
    )
     # START TIMER
    start_time = time.time()
    model.optimize()

    # END TIMER
    end_time = time.time()
    runtime = end_time - start_time

    print("=" * 60)
    print("Model solved.")
    print(f"Status code: {model.Status}")
    print(f"Runtime: {runtime:.4f} seconds")

    if model.Status == GRB.OPTIMAL:
        print(f"Objective value: {model.ObjVal:.4f}")
        print(f"Total supply: {total_supply:.2f}")

        print("\nAllocation decisions:")
        for i in camps:
            print(f"Camp {i}: X = {max(vars_dict['X'][i].X, 0):.4f}")

        print("\nUnmet internal demand:")
        for i in camps:
            print(f"Camp {i}: u = {max(vars_dict['u'][i].X, 0):.4f}")

        print("\nRejected external demand:")
        for i in camps:
            print(f"Camp {i}: r = {max(vars_dict['r'][i].X, 0):.4f}")

        print("\nLeftover inventory:")
        for i in camps:
            print(f"Camp {i}: l = {max(vars_dict['l'][i].X, 0):.4f}")

        print("\nDemand summary:")
        for i in camps:
            print(
                f"Camp {i}: internal = {lambda_c[i]:.2f}, "
                f"external = {lambda_u[i]:.2f}, "
                f"total = {lambda_c[i] + lambda_u[i]:.2f}"
            )

    elif model.Status == GRB.INFEASIBLE:
        print("Model is infeasible.")

    else:
        print("No optimal solution found.")

    return model, vars_dict


if __name__ == "__main__":
    solve_small_instance(total_supply=100000)
