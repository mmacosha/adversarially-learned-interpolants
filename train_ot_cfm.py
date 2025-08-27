from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm.auto import trange
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper

from train_ali_cfm import MLP, generate_data
from scipy import interpolate

import torch
from torchcfm.conditional_flow_matching import OTPlanSampler

@torch.no_grad()
def couple_across_time_sampled(
    X: torch.Tensor,
    times: torch.Tensor,
    device=None,
) -> torch.Tensor:
    """
    Couple trajectories across time by SAMPLING from the OT map between each
    consecutive snapshot.

    X: (n, K, d)  e.g. (10, 300, 2)
    times: (K,)
    Returns: Xc with same shape, where row i tracks a (random) particle over time.
    """
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
                  ot="independent", times=(0, 1)):
    X0, Xt, X1 = (torch.tensor(X, dtype=torch.float32).to(device) for X in data)
    times = torch.tensor(times, dtype=torch.float32).to(device)
    losses = []

    X = torch.concat([X0.unsqueeze(1), Xt, X1.unsqueeze(1)], dim=1)
    if ot == "ot":
        X = couple_across_time_sampled(X, times, device)
    X = X.permute(1, 0, 2)

    t = torch.linspace(0, 1, 10000, device=device).view(-1, 1)
    if interpolant == 'linear':
        idx = torch.bucketize(t.squeeze(-1), times) - 1
        idx = idx.clamp(0, X.shape[0] - 2)

        x_k = X[idx]
        x_kp1 = X[idx + 1]  # (K, n, d)

        t0 = times[idx].view(-1, 1, 1)
        t1 = times[idx + 1].view(-1, 1, 1)
        tt = t.view(-1, 1, 1)

        xt, dxt = get_ot_interpolant_generalized(x_k, x_kp1, t0, t1, tt)  # (K, n, d)

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
    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 15})
    plt.plot(xt[:, 0, 0], xt[:, 0, 1], color='blue', alpha=0.6)
    # plt.title(title)
    plt.scatter(data[0][:, 0], data[0][:, 1], color='red', alpha=0.5, s=1)
    plt.scatter(data[1][..., 0], data[1][..., 1], color='red', label='real data', alpha=0.5, s=1)
    plt.scatter(data[2][:, 0], data[2][:, 1], color='red', alpha=0.5, s=1)
    plt.xlim(-3.5, 3.5)
    plt.tight_layout()
    plt.show()



    for step in trange(n_epochs, desc="Training CFM", leave=False):
        cfm_optimizer.zero_grad()

        t = torch.rand(batch_size, 1, device=device)

        if interpolant == 'linear':
            idx = torch.bucketize(t.squeeze(-1), times) - 1
            idx = idx.clamp(0, X.shape[1] - 2)

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


def main(distribution):
    seed = 0
    size = 3000
    batch_size = 128
    ot_fm = "ot"
    interpolant = 'cubic'

    data, times = generate_data(seed, distribution, size)
    dims = data[0].shape[-1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)

    cfm_model = MLP(dims, time_varying=True, w=64).to(device)
    cfm_optimizer = torch.optim.Adam(cfm_model.parameters(), 1e-3)
    cfm_model, losses = train_ot_cfm(data, interpolant, cfm_model, cfm_optimizer, batch_size, n_epochs=40_001,
                                     device=device, ot=ot_fm, times=times)
    plt.plot(np.array(losses)[1000:])
    plt.show()

    node = NeuralODE(torch_wrapper(cfm_model),
                     solver="dopri5", sensitivity="adjoint").to(device)

    num_int_steps = 1000
    X0 = torch.tensor(data[0], dtype=torch.float32).to(device)
    with torch.no_grad():
        cfm_traj = node.trajectory(X0, t_span=torch.linspace(0, 1, num_int_steps + 1),
        )
        cfm_traj = cfm_traj.cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.rcParams.update({'font.size': 15})
    plt.plot(cfm_traj[..., 0], cfm_traj[..., 1], color='blue', alpha=0.2)
    # plt.scatter(data[0][:, 0], data[0][:, 1], color='red', alpha=0.5, s=1)
    plt.scatter(data[1][..., 0], data[1][..., 1], color='red', alpha=0.5, s=1)
    # plt.scatter(data[2][:, 0], data[2][:, 1], color='red', alpha=0.5, s=1)
    titles = {"linear": "OT-CFM Trajectories", "cubic": "OT-MMFM Trajectories"}
    # plt.title(titles[interpolant])
    plt.tight_layout()
    # plt.legend(loc='upper right')
    plt.xlim(-3.5, 3.5)
    plt.ylim(-0.5, 2.5)
    plt.yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    plt.show()



if __name__ == '__main__':
    main("knot")

