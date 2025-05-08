import numpy as np
from toy_experiment.plot_cubic_splines import couple_marginals
from torchcfm.optimal_transport import wasserstein
from utils import Plotter, load_data
import torch as T


if __name__ == '__main__':
    scale_factor = 1.
    device = 'cpu'
    X0, Xt1, Xt2, X1 = load_data(scale_factor, device)

    # INIT PLOT OBJECT
    pl = Plotter("../data/ST_images/ref_U5_warped_images",
                 [0., 0.25, 0.75, 1.], coordinate_scaling=scale_factor)
    bs = max(X0.shape[0], Xt1.shape[0], Xt2.shape[0], X1.shape[0])

    emds = []
    n_runs = 10
    for i in range(n_runs):
        observed_x = T.zeros((bs, 3, 2))
        x0_indices = T.randint(0, X0.size(0), (bs,))
        observed_x[:, 0] = X0[x0_indices]

        xt2_indices = T.randint(0, Xt2.size(0), (bs,))
        observed_x[:, 1] = Xt2[xt2_indices]

        x1_indices = T.randint(0, Xt1.size(0), (bs,))
        observed_x[:, 2] = X1[x1_indices]

        coupled_x = couple_marginals(observed_x)
        t = 0.25
        xhat_t1 = (1 - t) * coupled_x[:, 0] + t * coupled_x[:, 1]  # linear interpolant (1 - t) x_0 + t * x_t2

        emds.append(wasserstein(Xt1, xhat_t1, power=1))

    print("Avg. EMD: ", np.mean(emds), "\pm", np.std(emds))

    pl.plot_interpolants(None, [X0, Xt1, Xt2, X1], np.ones(4, dtype=int), mmot_interpolants=[
        coupled_x[:, 0], xhat_t1, coupled_x[:, 1], coupled_x[:, 2]
    ])




