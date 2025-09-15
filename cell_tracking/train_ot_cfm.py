import torch

import os
import wandb
import warnings
from tqdm.auto import trange, tqdm
from hydra import compose, initialize
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import interpolate

from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
from torchcfm.conditional_flow_matching import OTPlanSampler

from cell_tracking_utils import CellOverlayViewer
from ali_cfm.nets import MLP
from ali_cfm.data_utils import get_dataset, denormalize, denormalize_gradfield


def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def couple_across_time_sampled(
    X: torch.Tensor,
    times: torch.Tensor,
    device=None,
) -> torch.Tensor:
    """
    Couple trajectories across time by SAMPLING from the OT map between each
    consecutive snapshot.

    X: (K, n, d)  e.g. (10, 300, 2)
    times: (K,)
    Returns: Xc with same shape, where row i tracks a (random) particle over time.
    """
    X = X.permute(1, 0, 2)
    if device is None:
        device = X.device
    Xc = X.to(device).clone()
    n, K, d = Xc.shape

    for k in range(K - 1):
        A = Xc[:, k, :]       # (n, d)
        B = Xc[:, k + 1, :]   # (n, d)

        otp = OTPlanSampler(method="exact")

        # Estimate dense plan P (n,n) from samples and normalize
        P = otp.get_map(A, B)
        P = P / (P.sum(axis=1, keepdims=True) + 1e-12)

        idxB = np.argmax(P[np.arange(n)], 1)  # torch.multinomial(P[np.arange(n)], num_samples=1).squeeze(1)  # (n,)

        # Reorder B according to sampled matches
        Xc[:, k + 1, :] = B[idxB, :]

    return Xc



def get_ot_interpolant_generalized(x0, x1, t0, t1, t):
    denom = (t1 - t0)     # (bs,)

    a = (t1 - t) / denom
    b = (t - t0) / denom

    xhat_t = a * x0 + b * x1
    dx_t = (x1 - x0) / denom

    return xhat_t, dx_t


def train_ot_cfm(data, interpolant, cfm_model, cfm_optimizer, batch_size, n_epochs, device,
                  plot_fn, min_max, ot="independent", times=(0, 1)):
    X = torch.concat([torch.tensor(X.unsqueeze(0), dtype=torch.float32).to(device) for X in data], dim=0)
    t_max = max(times)
    times_np = np.array(times) / t_max
    times = torch.tensor(times, dtype=torch.float32).to(device) / t_max
    losses = []

    if ot == "ot":
        X = couple_across_time_sampled(X, times, device)

    X = X.permute(1, 0, 2)
    # t = times.view(-1, 1)
    t = torch.linspace(0, 1, 1000, device=device).view(-1, 1)
    if interpolant == 'linear':
        idx = torch.bucketize(t.squeeze(-1), times) - 1
        idx = idx.clamp(0, X.shape[0] - 2)

        x_k = X[idx]
        x_kp1 = X[idx + 1]  # (K, n, d)

        t0 = times[idx].view(-1, 1, 1)
        t1 = times[idx + 1].view(-1, 1, 1)
        tt = t.view(-1, 1, 1)

        xt, dxt = get_ot_interpolant_generalized(x_k, x_kp1, t0, t1, tt)  # (K, n, d)

        xt_torch = xt
        xt = xt.detach().cpu().numpy()
        title = f"Piecewise Linear Interpolants ($K=${times.shape[0]})"
    else:
        y = X.detach().cpu().numpy()
        x = times.detach().cpu().numpy()  # (K,)

        # vectorized spline fit over (n, 2)
        splines = interpolate.CubicSpline(x, y, axis=0)  # no loops

        t_np = t.squeeze(-1).detach().cpu().numpy()  # (T,)
        xt = splines(t_np)  # (T, n, 2)   interpolated positions

        title = f"Cubic Splines Interpolants ($K=${times.shape[0]})"
    plot_fn(xt_torch, None, None, t_max, data, None, device, None, times_np, None, min_max, method="ot_cfm", animate=False)

    # import matplotlib as mpl
    # from mpl_toolkits.axes_grid1 import make_axes_locatable
    # fig, ax = plt.subplots(figsize=(6, 5))
    #
    # x0, y0 = data[1][:, 0, 0], data[1][:, 0, 1]
    # x1, y1 = data[1][:, -1, 0], data[1][:, -1, 1]
    #
    # # Combine data and assign labels
    # X = np.concatenate([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    # labels = np.array([0] * len(x0) + [1] * len(x1))
    #
    # # Build a discrete 2-color colormap
    # cmap = mpl.colors.ListedColormap(["red", "cyan"])
    # norm = mpl.colors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=2)
    #
    # # Scatter with colormap + labels
    # sc = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap, norm=norm, s=6, alpha=1)
    #
    # plt.rcParams.update({'font.size': 15})
    # # Add tight colorbar
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    # cbar = plt.colorbar(sc, cax=cax)
    # cbar.set_ticks([0, 1])
    # cbar.set_ticklabels([0, 1])  # or "red", "cyan" etc.
    # cbar.set_label(r"$t$")
    #
    # # Axes settings
    # ax.set_xlim(-3.5, 3.5)
    # ax.set_ylim(-0.5, 2.5)
    # ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 0.5, 1.0, 1.5, 2.0], fontsize=15)
    # ax.set_xticks([-2, 0, 2], [-2, 0, 2], fontsize=15)
    #
    # plt.tight_layout()
    # plt.show()


    for step in trange(n_epochs, desc="Training CFM", leave=False):
        cfm_optimizer.zero_grad()

        t = torch.rand(batch_size, 1, device=device)

        if interpolant == 'linear':
            idx = torch.bucketize(t.squeeze(-1), times) - 1
            idx = idx.clamp(0, X.shape[0] - 2)

            x_k = X[idx]
            x_kp1 = X[idx + 1]  # (K, n, d)

            t0 = times[idx].view(-1, 1, 1)
            t1 = times[idx + 1].view(-1, 1, 1)
            tt = t.view(-1, 1, 1)

            xt, ut = get_ot_interpolant_generalized(x_k, x_kp1, t0, t1, tt)
        else:
            t_np = t.squeeze(-1).cpu().numpy()
            xt = torch.tensor(splines(t_np), device=device, dtype=torch.float32).squeeze(1)
            ut = torch.tensor(splines(t_np, 1), device=device, dtype=torch.float32).squeeze(1)

        B, N, D = xt.shape  # (batch_size, 10, 2)
        col_idx = torch.randint(0, N, (B, 1, 1), device=xt.device)
        xt = xt.gather(1, col_idx.expand(-1, 1, D)).squeeze(1)
        ut = ut.gather(1, col_idx.expand(-1, 1, D)).squeeze(1)  # (batch_size, 2)

        vt = cfm_model(torch.cat([xt, t], dim=-1))

        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        cfm_optimizer.step()
        losses.append(loss.item())
    return cfm_model, losses


def main(cfg):
    os.environ["HYDRA_FULL_ERROR"] = '1'
    seed_list = cfg.seed_list
    warnings.filterwarnings("ignore")
    ot_sampler = OTPlanSampler('exact', reg=0.1)
    n_samples = 10

    data, min_max = get_dataset(cfg.dataset, cfg.n_data_dims,
                                cfg.normalize_dataset, cfg.whiten)
    timesteps_list = [t for t in range(len(data))]
    # This code assumes that timesteps are in [0, ..., T_max]
    num_int_steps = cfg.num_int_steps_per_timestep

    ot_fm = "ot"
    interpolant = 'linear'  # 'linear' or 'cubic'
    cov = CellOverlayViewer('/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/data/PhC-C2DH-U373/01/',
                            method=interpolant)

    cfm_results = {}

    for seed in tqdm(seed_list, desc="Seeds"):
        fix_seed(seed)
        cfm_results[f"seed={seed}"] = []

        curr_timesteps = timesteps_list

        # N_min = min([x.shape[0] for x in data])
        # idx = np.random.choice(np.arange(0, N_min), n_samples, replace=False)
        # subset_data = [x[idx] for x in data]
        subset_data = [x[np.random.choice(np.arange(0, x.shape[0]), n_samples, replace=False)] for x in data]

        ot_cfm_model = MLP(dim=cfg.dim, time_varying=True, w=cfg.net_hidden).to(cfg.device)
        ot_cfm_optimizer = torch.optim.Adam(ot_cfm_model.parameters(), 1e-3)
        ot_cfm_model, losses = train_ot_cfm(subset_data, interpolant, ot_cfm_model, ot_cfm_optimizer, cfg.batch_size,
                                         n_epochs=cfg.n_ot_cfm_epochs, device=cfg.device, ot=ot_fm,
                                            times=curr_timesteps, plot_fn=cov.plot_fn, min_max=min_max)
        plt.plot(np.array(losses)[1000:])
        plt.show()

        node = NeuralODE(torch_wrapper(ot_cfm_model),
                         solver="dopri5", sensitivity="adjoint").to(cfg.device)

        X0 = torch.tensor(data[0], dtype=torch.float32).to(cfg.device)
        with torch.no_grad():
            cfm_traj = node.trajectory(X0, t_span=torch.tensor(timesteps_list, dtype=torch.float32).to(cfg.device) / max(timesteps_list),
            )

        cov.plot_fn(cfm_traj, None, None, max(timesteps_list), data,
                    None, cfg.device, None, np.array(timesteps_list) / max(timesteps_list),
                    None, min_max, method="ot_cfm")





if __name__ == '__main__':
    with initialize(config_path="./configs"):
        cfg = compose(config_name="ot_cfm.yaml")
        main(cfg)

