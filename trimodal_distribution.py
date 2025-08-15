import numpy as np

def generate_marginals(n_samples):

    # Marginal at t = 0
    mu_0 = 0.0
    sigma_0 = 0.3
    samples_t0 = np.random.normal(loc=mu_0, scale=sigma_0, size=n_samples)

    # Marginal at t = 0.5: Trimodal GMM
    means = np.array([-1.5, 0.0, 1.5])
    stds = np.array([0.2, 0.2, 0.2])
    weights = np.array([1/3, 1/3, 1/3])

    components = np.random.choice(len(means), size=n_samples, p=weights)
    samples_t05 = np.random.normal(loc=means[components], scale=stds[components])

    # Marginal at t = 1
    mu_1 = 0.0
    sigma_1 = 0.3
    samples_t1 = np.random.normal(loc=mu_1, scale=sigma_1, size=n_samples)

    return samples_t0.reshape((-1, 1)), samples_t05.reshape((-1, 1)), samples_t1.reshape((-1, 1))