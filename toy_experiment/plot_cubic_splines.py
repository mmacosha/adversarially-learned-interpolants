import numpy as np
from scipy import interpolate
from torchcfm import OTPlanSampler
import torch as T
from generate_circle_data import sample_interm
import matplotlib.pyplot as plt


def couple_marginals(observed_x):
    # Sequentially get OT couplings between pairs of data
    # observed_x tensor that contains samples from the data distributions, ordered sequentially
    # observed_x.shape = (bs, J, dims), where J is the number of time stamps such that t_j for j in [0, J] is a time stamp
    # where we have observed data, and t_0 = 0, t_J = 1


    otplan = OTPlanSampler('exact')

    J = observed_x.shape[1]

    for j in np.arange(0, J - 1):
        map = otplan.get_map(observed_x[:, j], observed_x[:, j + 1])
        target_indices = map.argmax(axis=1)
        observed_x[:, j + 1] = observed_x[target_indices, j + 1]

    return observed_x


def get_cubic_spline_interpolation(observed_x, observed_t):
    # observed_x tensor that contains samples from the data distributions, ordered sequentially
    # observed_x.shape = (bs, J, dims), where J is the number of time stamps such that t_j for j in [0, J] is a time stamp
    # observed_t tensor that holds the corresponding times (bs, J), observed_x[0, j] was sampled at time observed_t[0, j]

    coupled_x = couple_marginals(observed_x)
    bs = coupled_x.shape[0]

    # below is copied from https://github.com/Genentech/MMFM/blob/main/src/mmfm/multi_marginal_fm.py#L459
    if isinstance(observed_t, T.Tensor):
        observed_t = observed_t.cpu().numpy()
    if isinstance(coupled_x, T.Tensor):
        coupled_x = coupled_x.cpu().numpy()

    return [
        interpolate.CubicSpline(
            observed_t[b],
            coupled_x[b],
        )
        for b in range(bs)
    ]


def get_cubic_spline_interpolants(observed_x, observed_t, all_t, sigma_flow=0.01):
    # observed_t is a tensor holding a subset of the times in all_t
    # all_t is an array containing time stamps, e.g. [0., 0.1, ..., 1.]

    splines = get_cubic_spline_interpolation(observed_x, observed_t)
    # splines is a bs long list containing the inferred cubic spline objects

    bs = observed_x.shape[0]
    interpolants = T.zeros((bs, all_t.size, observed_x.shape[2]))
    for i, t in enumerate(all_t):
        for b, cs in enumerate(splines):
            mu_t = cs(t)
            x = mu_t + sigma_flow * np.random.randn(mu_t.shape[-1])  # draw a d-dimensional draw from the probability path
            interpolants[b, i] = T.tensor(x, dtype=T.float32)
    return interpolants


if __name__ == '__main__':
    all_t = np.arange(0, 11) / 10
    time_stamps = np.array([0.] + list((np.arange(1, 10) / 10)[::2]) + [1.])
    bs = 128

    observed_t = T.zeros((bs, time_stamps.size))
    observed_x = T.zeros((bs, time_stamps.size, 2))
    true_x = T.zeros((bs, all_t.size, 2))

    j = 0
    for i, t in enumerate(all_t):
        t_ = T.ones(bs) * t
        x_t, xhat_t, y_t = sample_interm(bs, t_, 0.2)
        true_x[:, i] = x_t

        if t in time_stamps:
            observed_t[:, j] = t_
            observed_x[:, j] = xhat_t
            j += 1

    interpolants = get_cubic_spline_interpolants(observed_x, observed_t, all_t)

    fig, axes = plt.subplots(3, 11, figsize=(15, 6), sharex=True, sharey=True)

    j = 0
    for i, t in enumerate(all_t):

        x_t = true_x[:, i]
        xt_fake = interpolants[:, i]

        axes[0, i].scatter(x_t[:, 0], x_t[:, 1], s=1)
        axes[0, i].set_title(f'$t$ = {all_t[i]:.1f}')
        if (t in time_stamps):
            if (t == 0.) or (t == 1.):
                pass
            else:
                xhat_t = observed_x[:, j]
                axes[1, i].scatter(xhat_t[:, 0], xhat_t[:, 1], s=1)
            j = j + 1
        else:
            axes[1, i].set_title('Unobserved')
        axes[2, i].scatter(xt_fake[:, 0], xt_fake[:, 1], s=1)
    axes[0, 0].set_ylabel(r'True Kernel, $\kappa(x|y_t)$')
    axes[1, 0].set_ylabel(r'Approx Kernel, $\hat{\kappa}(x|y_t)$')
    axes[2, 0].set_ylabel(r'Cubic Spline')

    for ax in axes[-1, :]:
        ax.set_xlabel('$x_1$')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_aspect('equal')
            ax.grid(True)

    plt.tight_layout()
    plt.show()

