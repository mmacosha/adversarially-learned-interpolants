import torch as T
from matplotlib import pyplot as plt
import numpy as np

N = 100


def p_y(t, N=100):
    mu_base = 2
    mu_1t = T.zeros((N, 2))
    mu_1t[:, 0] = mu_base * t + (1 - t) * -mu_base
    mu_1t[:, 1] = -mu_1t[:, 0]
    sigma_t = T.tensor([[1, 0.9],
                        [0.9, 1]])

    dist1 = T.distributions.MultivariateNormal(loc=mu_1t,
                                               covariance_matrix=sigma_t)

    y = dist1.sample()
    return y, mu_1t


# Transform y to x
def true_kernel(y, t, mu):
    theta = T.pi * t
    c = T.cos(theta)
    s = T.sin(theta)

    row1 = T.stack([c, -s], dim=-1)  # (batch,2)
    row2 = T.stack([s, c], dim=-1)  # (batch,2)

    A = T.stack([row1, row2], dim=1)

    y_shifted = y - mu
    y_shifted_and_rotated = T.einsum("bij,bj->bi", A, y_shifted)
    return y_shifted_and_rotated + mu


def approximate_kernel(y, t, mu):
    sigma_t = T.tensor([[1, 0.9],
                        [0.9, 1]])

    dist1 = T.distributions.MultivariateNormal(loc=mu,
                                               covariance_matrix=sigma_t)

    y = dist1.sample()
    x = true_kernel(y, t, mu)
    return x


def sample_interm(bs, t, lam=None):
    y_t, mu = p_y(t, bs)
    x = true_kernel(y_t, t, mu)
    x_hat = approximate_kernel(y_t, t, mu)
    return x, x_hat, y_t


if __name__ == '__main__':
    time_stamps = np.arange(0, 11) / 10
    fig, axes = plt.subplots(3, 11, figsize=(15, 6), sharex=True, sharey=True)
    bs = 1000
    lam = 0.1
    for i, t in enumerate(time_stamps):
        t = T.tensor(T.ones(bs) * t)
        x_t, xhat_t, y_t = sample_interm(bs, t, lam=lam)

        axes[0, i].scatter(y_t[:, 0], y_t[:, 1], s=1)
        axes[1, i].scatter(x_t[:, 0], x_t[:, 1], s=1)
        axes[0, 5].set_title(f'$t$ = {time_stamps[5]:.1f}')
        axes[2, i].scatter(xhat_t[:, 0], xhat_t[:, 1], s=1)
    axes[0, 0].set_ylabel(r'True Kernel, $\kappa(x|y_t)$')
    axes[1, 0].set_ylabel(r'Approx Kernel, $\hat{\kappa}(x|y_t)$')
    axes[2, 0].set_ylabel(r'Generator')

    for ax in axes[-1, :]:
        ax.set_xlabel('$x_1$')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_aspect('equal')
            ax.grid(True)

    plt.tight_layout()
    plt.show()