import numpy as np
import time


# ============================================================
# 1. Random Data Generation
# ============================================================

def generate_random_data(
    num_camps=200,
    num_aid_types=40,
    seed=42
):
    np.random.seed(seed)

    camps = [f"Camp_{i}" for i in range(1, num_camps + 1)]
    aid_types = [f"Aid_{k}" for k in range(1, num_aid_types + 1)]

    internal_demand = np.random.randint(
        500, 8000, size=(num_camps, num_aid_types)
    )

    external_demand = np.random.randint(
        500, 10000, size=(num_camps, num_aid_types)
    )

    total_demand = internal_demand + external_demand

    total_supply = np.zeros(num_aid_types)

    for k in range(num_aid_types):
        total_supply[k] = np.sum(total_demand[:, k]) * np.random.uniform(0.55, 0.80)

    leftover_penalty = np.random.uniform(0.5, 3.0, size=num_aid_types)
    referral_cost = np.random.uniform(2.0, 15.0, size=num_aid_types)
    deprivation_cost = np.random.uniform(30.0, 120.0, size=num_aid_types)

    fixed_activation_cost = np.random.uniform(
        5000, 30000, size=(num_camps, num_aid_types)
    )

    return {
        "camps": camps,
        "aid_types": aid_types,
        "internal_demand": internal_demand,
        "external_demand": external_demand,
        "total_demand": total_demand,
        "total_supply": total_supply,
        "leftover_penalty": leftover_penalty,
        "referral_cost": referral_cost,
        "deprivation_cost": deprivation_cost,
        "fixed_activation_cost": fixed_activation_cost,
    }


# ============================================================
# 2. Feasibility Repair
# ============================================================

def repair_solution(
    y,
    fixed_activation_cost,
    max_aid_types_per_camp,
    max_camps_per_aid_type,
    activation_budget
):
    """
    Repairs binary activation matrix y so that:
    1. each camp receives at most L aid types
    2. each aid type is sent to at most C camps
    3. total activation cost is within budget
    """

    y = y.copy()
    num_camps, num_aid_types = y.shape

    # Camp-level repair: sum_k y[i,k] <= L
    for i in range(num_camps):
        active = np.where(y[i, :] == 1)[0]

        if len(active) > max_aid_types_per_camp:
            costs = fixed_activation_cost[i, active]
            remove_count = len(active) - max_aid_types_per_camp

            # remove the most expensive active aid types
            remove_indices = active[np.argsort(costs)[-remove_count:]]
            y[i, remove_indices] = 0

    # Aid-level repair: sum_i y[i,k] <= C
    for k in range(num_aid_types):
        active = np.where(y[:, k] == 1)[0]

        if len(active) > max_camps_per_aid_type:
            costs = fixed_activation_cost[active, k]
            remove_count = len(active) - max_camps_per_aid_type

            # remove the most expensive active camps
            remove_indices = active[np.argsort(costs)[-remove_count:]]
            y[remove_indices, k] = 0

    # Budget repair
    total_activation_cost = np.sum(fixed_activation_cost * y)

    while total_activation_cost > activation_budget and np.sum(y) > 0:
        active_pairs = np.argwhere(y == 1)

        pair_costs = np.array([
            fixed_activation_cost[i, k]
            for i, k in active_pairs
        ])

        # remove the most expensive active pair
        remove_pair = active_pairs[np.argmax(pair_costs)]
        y[remove_pair[0], remove_pair[1]] = 0

        total_activation_cost = np.sum(fixed_activation_cost * y)

    return y


# ============================================================
# 3. Evaluation Function
# ============================================================

def evaluate_solution(y, data):
    """
    Given a binary activation matrix y, compute allocation X greedily.

    Logic:
    - If y[i,k] = 0, then X[i,k] = 0.
    - If y[i,k] = 1, then camp i is eligible to receive aid type k.
    - For each aid type k, available supply is allocated among active camps
      according to priority score.
    """

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
    u = np.zeros((num_camps, num_aid_types))
    r = np.zeros((num_camps, num_aid_types))
    l = np.zeros((num_camps, num_aid_types))

    for k in range(num_aid_types):
        active_camps = np.where(y[:, k] == 1)[0]
        remaining_supply = total_supply[k]

        if len(active_camps) == 0:
            continue

        # Priority score: higher deprivation and referral impact first
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

    # Compute shortages and leftovers
    u = np.maximum(internal - X, 0)

    r = np.maximum(
        internal + external - X - u,
        0
    )

    l = np.maximum(
        X - internal - external,
        0
    )

    objective = (
        np.sum(deprivation_cost.reshape(1, -1) * u)
        + np.sum(referral_cost.reshape(1, -1) * r)
        + np.sum(leftover_penalty.reshape(1, -1) * l)
        + np.sum(fixed_activation_cost * y)
    )

    return objective, X, u, r, l


# ============================================================
# 4. Initial Population
# ============================================================

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

    y = repair_solution(
        y,
        fixed_activation_cost,
        max_aid_types_per_camp,
        max_camps_per_aid_type,
        activation_budget
    )

    return y


# ============================================================
# 5. Genetic Algorithm
# ============================================================

def genetic_algorithm(
    data,
    max_aid_types_per_camp=8,
    max_camps_per_aid_type=80,
    activation_budget_ratio=0.35,
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

        obj, X, u, r, l = evaluate_solution(y, data)
        population.append((y, obj))

    population = sorted(population, key=lambda x: x[1])

    best_y = population[0][0].copy()
    best_obj = population[0][1]

    for gen in range(1, n_gen + 1):

        offspring = []

        # Crossover
        for _ in range(n_crossover):
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            selected = sorted(
                [population[idx] for idx in tournament],
                key=lambda x: x[1]
            )

            parent1 = selected[0][0]
            parent2 = selected[1][0]

            # Uniform crossover
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

        # Mutation
        for _ in range(n_mutation):
            tournament = np.random.choice(len(population), tournament_size, replace=False)
            selected = sorted(
                [population[idx] for idx in tournament],
                key=lambda x: x[1]
            )

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

        if gen % 10 == 0 or gen == 1:
            print(f"Generation {gen}: Best objective = {best_obj:.4f}")

    final_obj, X, u, r, l = evaluate_solution(best_y, data)

    return best_y, final_obj, X, u, r, l


# ============================================================
# 6. Main
# ============================================================

if __name__ == "__main__":
    data = generate_random_data(
        num_camps=200,
        num_aid_types=40,
        seed=42
    )

    start_time = time.time()

    best_y, best_obj, X, u, r, l = genetic_algorithm(
        data=data,
        max_aid_types_per_camp=8,
        max_camps_per_aid_type=80,
        activation_budget_ratio=0.35,
        n_pop=80,
        n_gen=300,
        n_crossover=40,
        n_mutation=40,
        tournament_size=5,
        activation_probability=0.15,
        mutation_probability=0.02,
        seed=42
    )

    end_time = time.time()

    print("=" * 70)
    print("GA completed.")
    print(f"Runtime: {end_time - start_time:.4f} seconds")
    print(f"Best objective value: {best_obj:.4f}")
    print(f"Number of activated camp-aid pairs: {int(np.sum(best_y))}")

    print("\nSample Results")
    print("-" * 70)

    for i in range(5):
        for k in range(5):
            print(
                f"Camp_{i+1}, Aid_{k+1}: "
                f"y = {best_y[i, k]}, "
                f"X = {X[i, k]:.2f}, "
                f"u = {u[i, k]:.2f}, "
                f"r = {r[i, k]:.2f}, "
                f"l = {l[i, k]:.2f}"
            )

