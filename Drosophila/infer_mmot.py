import numpy as np
from toy_experiment.plot_cubic_splines import couple_marginals
from torchcfm.optimal_transport import wasserstein
from utils import Plotter, load_data
import torch as T


if __name__ == '__main__':
    time_stamps = np.arange(0, 16) / (16 - 1)
    X = load_data()
    pl = Plotter(time_stamps)

    bs = min([x.shape[0] for x in X])

    observed_slides = np.arange(16)[::2]
    n_observed = len(observed_slides)
    observed_x = T.zeros((bs, n_observed, 2))
    for i, x in enumerate(X):
        if i in observed_slides:
            idx = T.randint(0, x.size(0), (bs,))
            observed_x[:, i] = x[idx]

    coupled_x = couple_marginals(observed_x)
    emds = []
    j = 0
    for i, t in enumerate(time_stamps):
        if t not in time_stamps[::2]:
            x_t = (1 - t) * coupled_x[:, j] + t * coupled_x[:, j + 1]
            emds.append(wasserstein(x_t, X[i]))
            j += 1

    print(np.mean(emds))




