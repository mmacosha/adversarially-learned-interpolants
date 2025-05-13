import numpy as np
from scipy import interpolate
from torchcfm import OTPlanSampler
import torch as T
# from generate_circle_data import sample_interm
# from generate_rotating_gaussians import sample_interm
import matplotlib.pyplot as plt
import ot


# def couple_marginals(observed_x):
#     # Sequentially get OT couplings between pairs of data
#     # observed_x tensor that contains samples from the data distributions, ordered sequentially
#     # observed_x.shape = (bs, J, dims), where J is the number of time stamps such that t_j for j in [0, J] is a time stamp
#     # where we have observed data, and t_0 = 0, t_J = 1
#
#
#     otplan = OTPlanSampler('exact')
#
#     J = observed_x.shape[1]
#
#     for j in np.arange(0, J - 1):
#         map = otplan.get_map(observed_x[:, j], observed_x[:, j + 1])
#         target_indices = map.argmax(axis=1)
#         observed_x[:, j + 1] = observed_x[target_indices, j + 1]
#
#     return observed_x

def minibatch_couple_marginals(observed_x):
    bs, J, d = observed_x.shape
    otplan = OTPlanSampler('exact')

    # Pre-allocate output and index array
    aligned = T.zeros_like(observed_x)
    idxs = T.arange(bs)  # each path starts at sample i→i

    # Time 0 is just the original samples
    aligned[:, 0] = observed_x[:, 0]

    for j in range(J - 1):
        pi = otplan.get_map(observed_x[:, j], observed_x[:, j + 1])  # (bs, bs)

        # Draw next-step indices for each path
        probs = T.from_numpy(pi[idxs])  # (bs, bs)
        probs = probs / T.sum(probs, 1, keepdim=True)
        idxs = T.multinomial(probs, num_samples=1).squeeze(1)  # (bs,)

        # Record the sampled points at time j+1
        aligned[:, j + 1] = observed_x[idxs, j + 1]

    return aligned

def couple_marginals(X, bs, pi=None):
    """
        Samples bs triplets (x0, xt, x1) from the factorized joint
           π*(x0, xt, x1) ∝ π1*(x0, xt) · π2*(xt, x1) / μt(xt)
        assuming μt is uniform.

        Args:
            X : tuple of three tensors (X0, Xt, X1) with shapes
                X0: (n0, d), Xt: (nt, d), X1: (n1, d)
            bs: number of samples to draw (with replacement)

        Returns:
            aligned: Tensor of shape (bs, 3, d)
        """
    X0, Xt, X1 = X
    n0, d = X0.shape
    nt, _ = Xt.shape
    n1, _ = X1.shape
    device = X0.device
    aligned = T.zeros((bs, 3, d), device=device)

    if pi is None:
        otplan = OTPlanSampler('exact')
        # 1) compute the two pairwise plans as numpy arrays
        pi1_np = otplan.get_map(X0, Xt)  # shape (n0, nt)
        pi2_np = otplan.get_map(Xt, X1)  # shape (nt, n1)

    else:
        pi1_np, pi2_np = pi

    # 2) convert to torch and move to device
    pi1 = T.from_numpy(pi1_np).to(device)  # (n0, nt)
    pi2 = T.from_numpy(pi2_np).to(device)  # (nt, n1)

    # 3) sample xt uniformly (μt is uniform over nt points)
    # idx_t = T.randint(0, nt, (bs,), device=device)
    idx_t = T.tensor(np.random.choice(np.arange(0, nt), bs, replace=False), dtype=T.int, device=device)

    # 4) sample x0 | xt  using columns of pi1
    #    pi1[:, idx_t] → (n0, bs) → transpose → (bs, n0)
    probs0 = pi1[:, idx_t].t()
    probs0 = probs0 / probs0.sum(dim=1, keepdim=True)
    idx_0 = T.multinomial(probs0, num_samples=1).squeeze(1)

    # 5) sample x1 | xt using rows of pi2
    probs2 = pi2[idx_t, :]  # (bs, n1)
    probs2 = probs2 / probs2.sum(dim=1, keepdim=True)
    idx_1 = T.multinomial(probs2, num_samples=1).squeeze(1)

    # 6) gather the actual points
    aligned[:, 0] = X0[idx_0]
    aligned[:, 1] = Xt[idx_t]
    aligned[:, 2] = X1[idx_1]

    return aligned

def get_cubic_spline_interpolation(observed_x, observed_t, pi=None):
    # observed_x tensor that contains samples from the data distributions, ordered sequentially
    # observed_x.shape = (bs, J, dims), where J is the number of time stamps such that t_j for j in [0, J] is a time stamp
    # observed_t tensor that holds the corresponding times (bs, J), observed_x[0, j] was sampled at time observed_t[0, j]
    bs = observed_t.shape[0]
    coupled_x = couple_marginals(observed_x, bs, pi=pi)

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

    bs = observed_t.shape[0]
    interpolants = T.zeros((bs, all_t.size, 2))
    for i, t in enumerate(all_t):
        for b, cs in enumerate(splines):
            mu_t = cs(t)
            x = mu_t # + sigma_flow * np.random.randn(mu_t.shape[-1])  # draw a d-dimensional draw from the probability path
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

