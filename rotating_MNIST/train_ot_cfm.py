import torch

import os
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

from mnist_utils import plot_fn
from ali_cfm.nets import MLP, RotationCFM, CorrectionUNet, UNetCFM
from ali_cfm.data_utils import get_dataset, denormalize, denormalize_gradfield
from ali_cfm.loggin_and_metrics import compute_emd


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
    test_times_np = np.arange(17, dtype=np.float32) / 16.
    test_times = torch.tensor(test_times_np, dtype=torch.float32).to(device)
    losses = []

    if ot == "ot":
        X = couple_across_time_sampled(X, times, device)

    X = X.permute(1, 0, 2)
    t = test_times.view(-1, 1)
    # t = torch.linspace(0, 1, 100, device=device).view(-1, 1)
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

        # vectorized spline fit over (n, D)
        splines = interpolate.CubicSpline(x, y, axis=0)  # no loops

        t_np = t.squeeze(-1).detach().cpu().numpy()  # (T,)
        xt = splines(t_np)  # (T, n, D)   interpolated positions

        title = f"Cubic Splines Interpolants ($K=${times.shape[0]})"
    img_dim = int(np.sqrt(xt.shape[-1]))
    fig, ax = plt.subplots(2, len(test_times_np) // 2, figsize=(15, 3))
    row, col = 0, 0
    for i, t_ in enumerate(test_times_np[:-1]):
        x = xt[i]
        ax[row, col].imshow(x[0].reshape(img_dim, img_dim), cmap='gray')
        ax[row, col].set_title(f"t={360 * test_times_np[i]:.0f}°")
        ax[row, col].axis('off')
        col += 1
        if col >= ax.shape[1]:
            col = 0
            row += 1
    plt.show()



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

        B, N, D = xt.shape
        # There are B x N samples of dim D, we randomly choose one of the N for each batch element
        col_idx = torch.randint(0, N, (B, 1, 1), device=xt.device)
        xt = xt.gather(1, col_idx.expand(-1, 1, D)).squeeze(1)
        ut = ut.gather(1, col_idx.expand(-1, 1, D)).squeeze(1)  # (batch_size, D)

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

    data, min_max = get_dataset("RotatingMNIST_train", cfg.n_data_dims, normalize=cfg.normalize_dataset)
    timesteps_list = [t for t in range(len(data))]
    # This code assumes that timesteps are in [0, ..., T_max]
    num_int_steps = cfg.num_int_steps_per_timestep

    ot_fm = "ot"
    interpolant = 'cubic'  # 'linear' or 'cubic'

    cfm_results = {}
    cfg.dim = data[0].shape[1]

    for seed in tqdm(seed_list, desc="Seeds"):
        fix_seed(seed)
        cfm_results[f"seed={seed}"] = []

        curr_timesteps = timesteps_list

        ot_cfm_model = CorrectionUNet().to(cfg.device)
        ot_cfm_optimizer = torch.optim.Adam(ot_cfm_model.parameters(), 1e-3)
        ot_cfm_model, losses = train_ot_cfm(data, interpolant, ot_cfm_model, ot_cfm_optimizer, cfg.batch_size,
                                         n_epochs=cfg.n_ot_cfm_epochs, device=cfg.device, ot=ot_fm,
                                            times=curr_timesteps, plot_fn=plot_fn, min_max=min_max)
        plt.plot(np.array(losses)[1000:])
        plt.show()

        node = NeuralODE(torch_wrapper(ot_cfm_model),
                         solver="dopri5", sensitivity="adjoint").to(cfg.device)

        test_data, min_max = get_dataset("RotatingMNIST_test", cfg.n_data_dims, normalize=cfg.normalize_dataset)
        # test_data, min_max = get_dataset("RotatingMNIST_train", cfg.n_data_dims, normalize=cfg.normalize_dataset)
        timesteps_list_test = [t for t in range(len(test_data))]

        X0 = torch.tensor(test_data[0], dtype=torch.float32).to(cfg.device)
        t_s = torch.linspace(0, 1, 101)
        with torch.no_grad():
            cfm_traj = node.trajectory(denormalize(X0, min_max),
                                       t_s
                                       )

        img_dim = int(np.sqrt(cfg.dim))
        fig, ax = plt.subplots(1, len(timesteps_list_test) - 1, figsize=(20, 3))
        for i, t in enumerate(timesteps_list_test[1:]):
            cfm_t = torch.argmin(torch.abs(t_s - t / max(timesteps_list_test)))
            cfm_emd = compute_emd(
                denormalize(test_data[t], min_max).to(cfg.device),
                cfm_traj[cfm_t].float().to(cfg.device),
            )
            print(f"t={t}, EMD={cfm_emd.item():.4f}")

            ax[i].imshow(cfm_traj[cfm_t][0].reshape(img_dim, img_dim).cpu(), cmap='gray')
            ax[i].set_title(f"t={360 * (t / max(timesteps_list_test)):.0f}°")
            ax[i].axis('off')
            cfm_results[f"seed={seed}"].append(cfm_emd.item())

        plt.tight_layout()
        plt.show()
        print(np.mean(cfm_results[f"seed={seed}"]))


if __name__ == '__main__':
    with initialize(config_path="./configs"):
        cfg = compose(config_name="ot_cfm.yaml")
        main(cfg)

