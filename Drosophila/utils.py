import torch as T
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


def sample_interm_single_slice(h5ad, bs, slice_ID):
    X = h5ad[h5ad.obs.slice_ID == str(slice_ID)]
    sampled_indices = X.obs.index.to_series().sample(bs)
    X = X[sampled_indices].obsm['spatial'][:, :-1]
    return T.tensor(X, dtype=T.float32)


def sample_interm(X, slice_ID):
    # slice_ID is a tensor with dtype = float32
    bs = len(slice_ID)
    out = np.zeros((bs, 2), dtype=np.float32)

    # for each unique slice, sample once
    i = 0
    ids, counts = np.unique(slice_ID, return_counts=True)
    for sid, count in zip(ids, counts):
        sid = int(sid)
        sub = X[sid]
        coords = sub[T.randint(0, sub.shape[0], (count,))]
        out[i:i + count, :] = coords
        i += count

    return T.tensor(out, dtype=T.float32)

def load_data(scale_factor=1.):
    h5ad = sc.read_h5ad("../data/Drosophila/drosophila_p100.h5ad")

    n_slides = 16
    X = []
    for slide_id in range(1, n_slides + 1):
        sub = h5ad[h5ad.obs.slice_ID == str(slide_id)]
        X.append(
            T.tensor(sub.obsm['spatial'][:, :-1], dtype=T.float32) / scale_factor
        )

    return X


class Plotter(object):
    def __init__(self, time_stamps, slide_names=None):
        self.time_stamps = time_stamps
        self.n_slides = len(self.time_stamps)

        self.slide_names = [u for u in range(self.n_slides)] if slide_names is None else slide_names

    def plot_spot_coordinates(self, s_t, slide, ax, observed):
        # x_list is a list of coordinate matrices, e.g. containing four matrices of interpolants
        # observed is a boolean array indicating if a slide was used during training
        t = self.time_stamps[slide]
        x = s_t[:, 0]
        y = s_t[:, 1]
        ax.scatter(x, y, color='blue')
        ax.set_title(f""
                     f"{self.slide_names[slide]}" +
                     " (Observed)" * observed +
                     " (Unobserved)" * (1 - observed) +
                     f" t = {t}"
                     )
        ax.axis("off")

    def plot_interpolants(self, interpolant, X, observed_list, mmot_interpolants=None):
        # if plotting mmot-based interpolants, like cubic splines, OT-CFM or MFM, input a list of interpolants
        # as the mmot_interpolants argument
        fig, axes = plt.subplots(2, 16, figsize=(24, 12))
        x0 = X[0]
        x1 = X[-1]
        N = x0.shape[0]
        for slide, s_t in enumerate(X):
            self.plot_spot_coordinates(s_t.to('cpu'), slide, axes[0, slide], observed_list[slide])

            t = self.time_stamps[slide]
            t = T.ones(N, dtype=T.float32, device=x0.device) * t
            t = t.unsqueeze(-1)
            if mmot_interpolants is not None:
                xt_fake = mmot_interpolants[slide]
            else:
                with T.no_grad():
                    xt_fake = interpolant(x0, x1, t)
            self.plot_spot_coordinates(xt_fake.to('cpu'), slide, axes[1, slide], observed_list[slide])
        plt.tight_layout()
        plt.show()
