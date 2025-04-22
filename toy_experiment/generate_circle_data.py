import torch as T
import numpy as np

N = 100


def p_y(t, N=100):
    N_1, N_2 = np.random.multinomial(N, [0.5, 0.5])
    t_1 = t[:N_1]
    t_2 = t[N_1:]

    mu_base = 2
    mu_1t = T.zeros((N_1, 2))
    mu_1t[:, 0] = mu_base * t_1 + (1 - t_1) * -mu_base
    mu_1t[:, 1] = -mu_1t[:, 0]
    mu_2t = T.zeros((N_2, 2))
    mu_2t[:, 0] = -mu_base * t_2 + (1 - t_2) * mu_base
    mu_2t[:, 1] = -mu_2t[:, 0]
    sigma_t = T.tensor([[1, 0.],
                  [0., 1]])

    dist1 = T.distributions.MultivariateNormal(loc=mu_1t,
                                                covariance_matrix=sigma_t)
    dist2 = T.distributions.MultivariateNormal(loc=mu_2t,
                                                covariance_matrix=sigma_t)
    y_1 = dist1.sample()
    y_2 = dist2.sample()
    y = T.cat([y_1, y_2], 0)
    return y


# Transform y to x
def true_kernel(y, t):
    x = T.zeros_like(y)
    norm = T.sqrt(y[:, 0]**2 + y[:, 1]**2)
    r = np.sqrt((t * 2 + (1 - t) * -2) ** 2 + 1)
    x[:, 0] = r * (y[:, 0] / norm) + (t - (1 - t))
    x[:, 1] = r * (y[:, 1] / norm)
    return x


def approximate_kernel(y, t, noise_strength=2.):
    x = T.zeros_like(y)
    norm = T.sqrt(y[:, 0]**2 + y[:, 1]**2)
    r = np.sqrt((t * 2 + (1 - t) * -2) ** 2 + 1)
    radius = r * (1 - np.exp(-1 / noise_strength * norm ** 2))
    x[:, 1] = radius * (y[:, 1] / norm)
    x[:, 0] = radius * (y[:, 0] / norm) + (t - (1 - t))
    return x


def sample_interm(bs, t, lam=2.):
    y_t = p_y(t, bs)
    x = true_kernel(y_t, t)
    x_hat = approximate_kernel(y_t, t, noise_strength=lam)
    return x.type(T.float32), x_hat.type(T.float32), y_t