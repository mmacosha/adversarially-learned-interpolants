import numpy as np
from toy_experiment.plot_cubic_splines import couple_marginals
from torchcfm.optimal_transport import wasserstein
from utils import Plotter, load_data
import torch as T
import matplotlib.pyplot as plt


if __name__ == '__main__':
    time_stamps = np.arange(0, 16) / (16 - 1)
    X = load_data()
    pl = Plotter(time_stamps)

    bs = min([x.shape[0] for x in X])

    n_observed = 3
    observed_slides = np.linspace(0, 15, n_observed, dtype=int)

    observed_x = T.zeros((bs, n_observed, 2))
    j = 0
    for i, x in enumerate(X):
        if i in observed_slides:
            idx = T.randint(0, x.size(0), (bs,))
            observed_x[:, j] = x[idx]
            j += 1

    coupled_x = couple_marginals(observed_x)
    emds = []
    emds_observed = []

    fig, axes = plt.subplots(2, time_stamps.size, figsize=(30, 4), sharex=True, sharey=True)
    j = 1
    k = 0
    x0 = coupled_x[:, 0]
    x1 = coupled_x[:, 1]
    t_0 = 0
    t_1 = observed_slides[1] / 15
    for i, t in enumerate(time_stamps):
        if i not in observed_slides:
            axes[0, i].set_title('Unobserved')
            if i > observed_slides[j]:
                # t is now between the next two observed marginals
                j += 1
                x0 = coupled_x[:, j - 1]
                x1 = coupled_x[:, j]
                t_0 = observed_slides[j - 1] / 15
                t_1 = observed_slides[j] / 15

            t = i / 15
            denom = (t_1 - t_0)
            a_t = (t_1 - t) / denom
            b_t = (t - t_0) / denom

            x_t = a_t * x0 + b_t * x1
            emds.append(wasserstein(x_t, X[i], power=1))
        else:
            x_t = coupled_x[:, k]
            k += 1
            if (i > 0) and (i < 15):
                emds_observed.append(wasserstein(x_t, X[i], power=1))

        axes[0, i].scatter(X[i][:, 0], X[i][:, 1], s=1)
        axes[1, i].scatter(x_t[:, 0], x_t[:, 1], s=1)

    for ax in axes[-1, :]:
        ax.set_xlabel('$x_1$')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_aspect('equal')
            ax.grid(True)

    plt.tight_layout()
    plt.show()

    print("Mean EMD (Unobserved)", np.mean(emds))
    print("Mean EMD (Observed)", np.mean(emds_observed))




