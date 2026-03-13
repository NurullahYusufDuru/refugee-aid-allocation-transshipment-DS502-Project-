from gurobipy import Model, GRB, quicksum
#This code is generated with the help of AI

def build_model(camps, supply, initial_inventory=None, transshipment_cost=None):
    """
    Build the first version of the aid allocation model with lateral transshipment.

    Parameters
    ----------
    camps : list
        List of camp indices or names.
    supply : float
        Total available supply from the central decision-maker.
    initial_inventory : dict, optional
        Initial inventory at each camp. If None, all are set to 0.
    transshipment_cost : dict, optional
        Dictionary with keys (i, j) and values c_ij.
        If None, unit cost 1.0 is used for all i != j.

    Returns
    -------
    model : gurobipy.Model
        Gurobi model object.
    vars_dict : dict
        Dictionary containing decision and auxiliary variables.
    """

    if initial_inventory is None:
        initial_inventory = {i: 0.0 for i in camps}

    if transshipment_cost is None:
        transshipment_cost = {(i, j): 1.0 for i in camps for j in camps if i != j}

    model = Model("AidAllocationWithTransshipment")
    model.ModelSense = GRB.MINIMIZE

    # ----------------------------
    # Decision variables
    # ----------------------------
    # Allocation level at each camp
    X = model.addVars(camps, lb=0.0, vtype=GRB.CONTINUOUS, name="X")

    # Transshipment from camp i to camp j
    T = model.addVars(
        [(i, j) for i in camps for j in camps if i != j],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="T"
    )

    # ----------------------------
    # Auxiliary variables
    # ----------------------------
    # Effective inventory after transshipment
    s = model.addVars(camps, lb=0.0, vtype=GRB.CONTINUOUS, name="s")

    # ----------------------------
    # Objective (placeholder)
    # ----------------------------
    # For D3, I use a simple placeholder objective:
    # minimize total allocation + transshipment cost
    #
    # Later this can be replaced by:
    # piecewise expected cost evaluated at s[i]
    model.setObjective(
        quicksum(X[i] for i in camps) +
        quicksum(transshipment_cost[i, j] * T[i, j] for i, j in T.keys()),
        GRB.MINIMIZE
    )

    # ----------------------------
    # Constraints
    # ----------------------------

    # 1. Total supply constraint
    model.addConstr(
        quicksum(X[i] - initial_inventory[i] for i in camps) <= supply,
        name="total_supply"
    )

    # 2. Inventory feasibility: allocation level must be at least initial inventory
    model.addConstrs(
        (X[i] >= initial_inventory[i] for i in camps),
        name="inventory_feasibility"
    )

    # 3. Inventory balance after transshipment
    model.addConstrs(
        (
            s[i] == X[i]
            - quicksum(T[i, j] for j in camps if j != i)
            + quicksum(T[j, i] for j in camps if j != i)
            for i in camps
        ),
        name="inventory_balance"
    )

    # 4. Transshipment feasibility:
    # a camp cannot send more inventory than it has after allocation
    model.addConstrs(
        (
            quicksum(T[i, j] for j in camps if j != i) <= X[i]
            for i in camps
        ),
        name="transshipment_feasibility"
    )

    model.update()

    vars_dict = {
        "X": X,
        "T": T,
        "s": s
    }

    return model, vars_dict


if __name__ == "__main__":
    camps = [1, 2, 3, 4, 5]
    supply = 100

    initial_inventory = {
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0
    }

    transshipment_cost = {
        (i, j): 1.0
        for i in camps for j in camps if i != j
    }

    model, vars_dict = build_model(
        camps=camps,
        supply=supply,
        initial_inventory=initial_inventory,
        transshipment_cost=transshipment_cost
    )

    model.write("model_v1.lp")
    print("Model created successfully.")
    print(f"Number of variables: {model.NumVars}")
    print(f"Number of constraints: {model.NumConstrs}")
