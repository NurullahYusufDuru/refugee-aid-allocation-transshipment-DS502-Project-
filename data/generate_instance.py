import numpy as np

def generate_instance(n_camps=5, seed=42):

    np.random.seed(seed)

    lambda_internal = np.random.uniform(2000,10000,n_camps)
    lambda_external = np.random.uniform(1000,5000,n_camps)

    return lambda_internal, lambda_external

if __name__ == "__main__":

    lam_c, lam_u = generate_instance()

    print("Internal demand rates:", lam_c)
    print("External demand rates:", lam_u)
