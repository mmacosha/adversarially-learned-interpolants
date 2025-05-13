import numpy as np
from toy_experiment.plot_cubic_splines import get_cubic_spline_interpolants
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
    bs = min(X0.shape[0], Xt1.shape[0], Xt2.shape[0], X1.shape[0])

    emds = []
    n_runs = 20
    all_t = np.array([0., 0.25, 0.75, 1.])
    observed_t = T.zeros((bs, 3))
    observed_t[:, 0] = 0.
    observed_t[:, 2] = 1.
    T.manual_seed(0)

    for i in range(n_runs):
        observed_x = T.zeros((bs, 3, 2))
        x0_indices = T.randint(0, X0.size(0), (bs,))
        observed_x[:, 0] = X0[x0_indices]

        # U4 left out
        xt1_indices = T.randint(0, Xt1.size(0), (bs,))
        observed_x[:, 1] = Xt1[xt1_indices]
        observed_t[:, 1] = 0.25

        # U3 left out
        # xt2_indices = T.randint(0, Xt2.size(0), (bs,))
        # observed_x[:, 1] = Xt2[xt2_indices]
        # observed_t[:, 1] = 0.75

        x1_indices = T.randint(0, X1.size(0), (bs,))
        observed_x[:, 2] = X1[x1_indices]

        # interpolants.shape = (bs, len(all_t), 2)
        interpolants = get_cubic_spline_interpolants([X0, Xt1, X1], observed_t, all_t)

        # U4 left out
        xhat_t = interpolants[:, 2]
        emds.append(wasserstein(Xt2 * scale_factor, xhat_t * scale_factor, power=1))

        # U3 left out
        # xhat_t = interpolants[:, 1]
        # emds.append(wasserstein(Xt1 * scale_factor, xhat_t * scale_factor, power=1))

    print("Avg. EMD: ", np.mean(emds), "\pm", np.std(emds))

    #pl.plot_interpolants(None, [X0, Xt1, Xt2, X1], np.ones(4, dtype=int), mmot_interpolants=[
    #    coupled_x[:, 0], xhat_t1, coupled_x[:, 1], coupled_x[:, 2]
    #])




