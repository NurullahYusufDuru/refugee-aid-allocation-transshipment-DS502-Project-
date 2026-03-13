from model_v1 import build_model

camps = [1, 2, 3, 4, 5]
supply = 100

model, vars_dict = build_model(camps, supply)
model.optimize()

if model.status == 2:
    print("Optimal solution found.")
    for i in camps:
        print(f"X[{i}] =", vars_dict["X"][i].X)
