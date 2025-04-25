import torch as T
import pandas as pd
import numpy as np


def sample_interm_single_slice(h5ad, bs, slice_ID):
    X = h5ad[h5ad.obs.slice_ID == str(slice_ID)]
    sampled_indices = X.obs.index.to_series().sample(bs)
    X = X[sampled_indices].obsm['spatial'][:, :-1]
    return T.tensor(X, dtype=T.float32)


def sample_interm(h5ad, slice_ID):
    # slice_ID is a tensor with dtype = float32
    slice_ID = slice_ID.cpu().numpy().astype(int).astype(str).tolist()

    slice_series = pd.Series(slice_ID)
    bs = len(slice_series)
    out = np.zeros((bs, 2), dtype=np.float32)

    # for each unique slice, sample once
    for sid, idxs in slice_series.groupby(slice_series).groups.items():
        count = len(idxs)
        sub = h5ad[h5ad.obs.slice_ID == sid]
        chosen = sub.obs.sample(n=count).index
        coords = sub[chosen].obsm['spatial'][:, :-1]  # shape (bs, D)
        out[list(idxs), :] = coords

    return T.tensor(out, dtype=T.float32)


