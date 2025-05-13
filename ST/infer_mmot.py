import numpy as np
import torch

from toy_experiment.plot_cubic_splines import couple_marginals
from torchcfm.optimal_transport import wasserstein
from utils import Plotter, load_data
import torch as T

def pre_compute_couplings(X0, Xt1, Xt2, X1, unobserved, bs, n_batches):
    coupled_x = T.zeros((n_batches, bs, 3 ,2), device=X0.device)
    for batch in range(n_batches):
        observed_x = T.zeros((bs, 3, 2), device=X0.device)
        x0_indices = T.randint(0, X0.size(0), (bs,), device=X0.device)
        observed_x[:, 0] = X0[x0_indices]
        x1_indices = T.randint(0, X1.size(0), (bs,))
        observed_x[:, 2] = X1[x1_indices]

        if unobserved == 0.75:
            # U4 left out
            xt1_indices = T.randint(0, Xt1.size(0), (bs,))
            observed_x[:, 1] = Xt1[xt1_indices]
        else:
            # U3 left out
            xt2_indices = T.randint(0, Xt2.size(0), (bs,))
            observed_x[:, 1] = Xt2[xt2_indices]

        coupled_x[batch] = couple_marginals(observed_x).to(X0.device)
    return coupled_x

def get_ot_interpolant_given_coupling(coupled_x, unobserved, t, bs):
    device = coupled_x.device
    xhat_t = T.zeros((bs, 2), device=device)
    dx_t = T.zeros((bs, 2), device=device)

    observed_t = (1 - unobserved)
    idx_i = t.squeeze(-1) > observed_t
    idx_j = t.squeeze(-1) < observed_t

    # Linear interpolant between t = observed_t and t = 1
    denom_i = (1 - observed_t)
    a_i = (1. - t[idx_i]) / denom_i
    b_i = (t[idx_i] - observed_t) / denom_i
    xhat_t[idx_i] = a_i * coupled_x[idx_i, 1] + b_i * coupled_x[idx_i, 2]
    dx_t[idx_i] = coupled_x[idx_i, 2] / denom_i - coupled_x[idx_i, 1] / denom_i

    # Linear interpolant between t = 0 and t = observed_t
    denom_j = (observed_t - 0)
    a_j = (observed_t - t[idx_j]) / denom_j
    b_j = (t[idx_j] - 0) / denom_j
    xhat_t[idx_j] = a_j * coupled_x[idx_j, 0] + b_j * coupled_x[idx_j, 1]
    dx_t[idx_j] = coupled_x[idx_j, 1] / denom_j - coupled_x[idx_j, 0] / denom_j

    return xhat_t, dx_t

def get_ot_interpolant(X0, Xt1, Xt2, X1, unobserved, t, bs):
    observed_x = T.zeros((bs, 3, 2), device=X0.device)
    x0_indices = T.randint(0, X0.size(0), (bs,), device=X0.device)
    observed_x[:, 0] = X0[x0_indices]
    x1_indices = T.randint(0, X1.size(0), (bs,))
    observed_x[:, 2] = X1[x1_indices]

    if unobserved == 0.75:
        # U4 left out
        xt1_indices = T.randint(0, Xt1.size(0), (bs,))
        observed_x[:, 1] = Xt1[xt1_indices]
    else:
        # U3 left out
        xt2_indices = T.randint(0, Xt2.size(0), (bs,))
        observed_x[:, 1] = Xt2[xt2_indices]


    coupled_x = couple_marginals(observed_x)
    coupled_x = coupled_x.to(X0.device)

    xhat_t = T.zeros((bs, 2), device=X0.device)
    dx_t = T.zeros((bs, 2), device=X0.device)

    observed_t = (1 - unobserved)
    idx_i = t.squeeze(-1) > observed_t
    idx_j = t.squeeze(-1) < observed_t

    # Linear interpolant between t = observed_t and t = 1
    denom_i = (1 - observed_t)
    a_i = (1. - t[idx_i]) / denom_i
    b_i = (t[idx_i] - observed_t) / denom_i
    xhat_t[idx_i] = a_i * coupled_x[idx_i, 1] + b_i * coupled_x[idx_i, 2]
    dx_t[idx_i] = coupled_x[idx_i, 2] / denom_i - coupled_x[idx_i, 1] / denom_i

    # Linear interpolant between t = 0 and t = observed_t
    denom_j = (observed_t - 0)
    a_j = (observed_t - t[idx_j]) / denom_j
    b_j = (t[idx_j] - 0) / denom_j
    xhat_t[idx_j] = a_j * coupled_x[idx_j, 0] + b_j * coupled_x[idx_j, 1]
    dx_t[idx_j] = coupled_x[idx_j, 1] / denom_j - coupled_x[idx_j, 0] / denom_j

    return xhat_t, dx_t

if __name__ == '__main__':
    scale_factor = 1.
    device = 'cpu'
    X0, Xt1, Xt2, X1 = load_data(scale_factor, device)

    # INIT PLOT OBJECT
    pl = Plotter("../data/ST_images/ref_U5_warped_images",
                 [0., 0.25, 0.75, 1.], coordinate_scaling=scale_factor)
    bs = min(X0.shape[0], Xt1.shape[0], Xt2.shape[0], X1.shape[0])

    torch.manual_seed(0)
    emds = []
    n_runs = 20
    for i in range(n_runs):
        observed_x = T.zeros((bs, 3, 2))
        x0_indices = T.randint(0, X0.size(0), (bs,))
        observed_x[:, 0] = X0[x0_indices]

        # U4 left out
        xt1_indices = T.randint(0, Xt1.size(0), (bs,))
        observed_x[:, 1] = Xt1[xt1_indices]
        t = 0.75
        t_0 = 0.25
        t_1 = 1.

        # U3 left out
        # xt2_indices = T.randint(0, Xt2.size(0), (bs,))
        # observed_x[:, 1] = Xt2[xt2_indices]
        # t = 0.25
        # t_0 = 0.
        # t_1 = .75

        x1_indices = T.randint(0, X1.size(0), (bs,))
        observed_x[:, 2] = X1[x1_indices]

        coupled_x = couple_marginals([X0, Xt1, X1], bs)

        denom = (t_1 - t_0)
        a_t = (t_1 - t) / denom
        b_t = (t - t_0) / denom

        # U4 left out
        xhat_t = a_t * coupled_x[:, 1] + b_t * coupled_x[:, 2]
        emds.append(wasserstein(Xt2 * scale_factor, xhat_t * scale_factor, power=1))

        # U3 left out
        # xhat_t = a_t * coupled_x[:, 0] + b_t * coupled_x[:, 1]
        # emds.append(wasserstein(Xt1 * scale_factor, xhat_t * scale_factor, power=1))



    print("Avg. EMD: ", np.mean(emds), "\pm", np.std(emds))

    # pl.plot_interpolants(None, [X0, Xt1, Xt2, X1], np.ones(4, dtype=int), mmot_interpolants=[
    #     coupled_x[:, 0], xhat_t1, coupled_x[:, 1], coupled_x[:, 2]
    # ])




