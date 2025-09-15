import random

import math
import torch
import ot as pot
import numpy as np
import pandas as pd

from pathlib import Path

from tqdm import trange

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


### Prepare batches for training ###

def get_batch(FM, X, batch_size, timesteps, return_noise=False, device='cpu'):
    """Construct a batch with point sfrom each timepoint pair"""
    ts = []
    xts = []
    uts = []
    noises = []
    for t_curr in range(len(timesteps) - 1):
        t_start = timesteps[t_curr]
        t_end = timesteps[t_curr + 1]

        idx = np.random.randint(X[t_start].shape[0], size=batch_size)
        x0 = X[t_start][idx].float().to(device)
        
        idx = np.random.randint(X[t_end].shape[0], size=batch_size)
        x1 = X[t_end][idx].float().to(device)
        
        if return_noise:
            t, xt, ut, eps = FM.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise, ts=0, te=t_end - t_start
            )
            noises.append(eps)
        else:
            t, xt, ut = FM.sample_location_and_conditional_flow(
                x0, x1, return_noise=return_noise, ts=0, te=t_end - t_start
            )

        ts.append(t + t_start)
        xts.append(xt)
        uts.append(ut)

    t = torch.cat(ts)
    xt = torch.cat(xts)
    ut = torch.cat(uts)

    if return_noise:
        noises = torch.cat(noises)
        return t, xt, ut, noises
    return t, xt, ut


def get_batch_for_cubic(FM, X, batch_size, timesteps, device='cpu'):
    batch = [x[:batch_size] for t, x in enumerate(X) if t in timesteps]
    batch = torch.from_numpy(np.stack(batch)).float().to(device)
    
    timesteps = torch.cat(
        [torch.ones(1, batch_size) * t for t in timesteps], 
        0).to(device)
    

    t, xt, ut, *_ = FM.sample_location_and_conditional_flow(
        batch.permute(1, 0, 2), timesteps.permute(1, 0)
    )

    xt = xt.squeeze(1)
    ut = ut.squeeze(1)
    
    return t, xt, ut


def sample_x0_x1(X, batch_size, device='cpu'):
    x0 = X[0][np.random.randint(X[0].shape[0], size=batch_size)].to(device)
    x1 = X[-1][np.random.randint(X[-1].shape[0], size=batch_size)].to(device)
    return x0, x1


def sample_gan_batch(X, batch_size, ot_sampler, divisor,
                     time=None, ot='none', times=(0, 1, 2, 3)):
    time = time or random.choice(times[1:-1])
    x0 = sample_x_batch(X[0], batch_size)
    x1 = sample_x_batch(X[-1], batch_size)
    xt = sample_x_batch(X[time], batch_size)
    t = torch.ones(batch_size, 1) * time / divisor

    if ot == 'full':
        x0, xt = sample_deterministic_ot_plan(x0, xt, ot_sampler)
        xt, x1 = sample_deterministic_ot_plan(xt, x1, ot_sampler)
    elif ot == 'border':
        x0, x1 = ot_sampler.sample_plan(x0, x1)
    elif ot == 'mmot':
        x0, x1 = mmot_couple_marginals(x0, x1, xt, ot_sampler)
    elif ot == 'none':
        pass
    else:
        raise ValueError(f"Unknown OT type: {ot}")

    return x0, x1, xt, t


def sample_x_batch(X, batch_size):
    return X[np.random.randint(0, X.shape[0], size=batch_size)]


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=3.6)
        m.bias.data.fill_(0.00)


### Sample OT Plans ###

def sample_deterministic_ot_plan(x0, x1, ot_sampler):
    pi = ot_sampler.get_map(x0, x1)
    pi = pi / pi.sum()
    return x0, x1[pi.argmax(axis=1)]


def mmot_couple_marginals(X0, X1, Xt, otplan):
    """
        Samples bs triplets (x0, xt, x1) from the factorized joint
           π*(x0, xt, x1) ∝ π1*(x0, xt) · π2*(xt, x1) / μt(xt)
        assuming μt is uniform.

        Args:
            minibatch samples from the three marginals
            X0: (bs, d)
            X1: (bs, d)
            Xt: (bs, d)
            bs: number of samples to draw (with replacement)

        Returns:
            aligned: Tensor of shape (bs, 3, d)
        """
    # X0, Xt, X1 = X
    bs, d = X0.shape

    device = X0.device

    # 1) compute the two pairwise plans as numpy arrays
    pi1_np = otplan.get_map(X0, Xt)  # shape (n0, nt)
    pi2_np = otplan.get_map(Xt, X1)  # shape (nt, n1)


    # 2) convert to torch and move to device
    pi1 = torch.from_numpy(pi1_np).to(device)  # (n0, nt)
    pi2 = torch.from_numpy(pi2_np).to(device)  # (nt, n1)

    idx_t = torch.tensor(np.arange(0, bs), dtype=torch.int, device=device)

    # 4) sample x0 | xt  using columns of pi1
    probs0 = pi1[:, idx_t].t()
    probs0 = probs0 / probs0.sum(dim=1, keepdim=True)
    idx_0 = torch.multinomial(probs0, num_samples=1, replacement=True).squeeze(1)

    # 5) sample x1 | xt using rows of pi2
    probs2 = pi2[idx_t, :]  # (bs, n1)
    probs2 = probs2 / probs2.sum(dim=1, keepdim=True)
    idx_1 = torch.multinomial(probs2, num_samples=1, replacement=True).squeeze(1)

    # 6) return the coupled points
    return X0[idx_0], X1[idx_1]


### Others ###

def integrate_interpolant(x0, x1, n_steps, interpolant):
    device = x0.device
    dt = 1 / n_steps
    t = torch.zeros(x0.shape[0], 1, device=device)

    integrated_trajectories = [x0.detach().cpu()]
    for _ in trange(n_steps, leave=False):
        xtm1 = integrated_trajectories[-1].to(device)
        xtm1 = xtm1 + interpolant.dI_dt(x0, x1, t) * dt
        integrated_trajectories.append(xtm1.cpu().detach())
        t += dt

    return torch.stack(integrated_trajectories)


def init_cfm_from_checkpoint(*args, **kwargs):
    raise NotImplementedError(
        "CFM checkpoint loading not implemented yet."
    )


def init_interpolant_from_checkpoint(*args, **kwargs):
    raise NotImplementedError(
        "Interpolant checkpoint loading not implemented yet."
    )


def sc_plot_fn(interpolant, epoch, seed, t_max, data, ot_sampler, device, metric_prefix, train_timesteps, wandb, min_max):
    with torch.no_grad():
        batch = sample_gan_batch(
            data, 256,
            divisor=t_max,
            ot_sampler=ot_sampler,
            ot='full',
            times=train_timesteps
        )
        x0, x1, xt, t = (x.to(device) for x in batch)
        xt_fake = interpolant(x0, x1, t)

        pca = PCA(n_components=2, random_state=seed)

        xt_pca = pca.fit_transform(xt.cpu())
        xt_fake_pca = pca.transform(xt_fake.cpu())
        fig = plt.figure()

        plt.scatter(*xt_fake_pca.T, c='red', alpha=0.5, label="Fake")
        plt.scatter(*xt_pca.T, c='blue', alpha=0.5, label="Real")
        plt.legend()
        plt.title(f"PCA of `True` and `Fake` samples for t={int(t[0] * t_max)}")

        wandb.log({
            f"{metric_prefix}/scatter_image": wandb.Image(fig),
            f"{metric_prefix}_step": epoch
        })
        plt.close(fig)
