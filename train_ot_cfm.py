from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm.auto import trange
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
from triton.language import dtype

from train_ali_cfm import MLP, generate_data
from scipy import interpolate


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

    t = torch.linspace(0, 1, 10000, device=device).view(-1, 1)
    if interpolant == 'linear':
        idx = torch.bucketize(t, times) - 1
        idx = idx.clamp(0, Xt.shape[0] - 2)
        x_k = Xt[idx].squeeze(1)
        x_kp1 = Xt[idx + 1].squeeze(1)
        xt, _ = get_ot_interpolant_generalized(x_k, x_kp1, times[idx], times[idx + 1], t)
        xt = xt.cpu().numpy()
        title = f"Piecewise Linear Interpolants ($K=${times.shape[0]})"
    else:
        splines = interpolate.CubicSpline(times.cpu().numpy(), Xt.cpu().numpy(), axis=0)
        t_np = t.cpu().numpy()
        xt =  splines(t_np).squeeze(1)
        dxt_np = splines(t_np, 1).squeeze(1)
        title = f"Cubic Splines Interpolants ($K=${times.shape[0]})"
    plt.plot(xt[..., 0], xt[..., 1])
    plt.title(title)
    plt.show()



    for step in trange(n_epochs, desc="Training CFM", leave=False):
        cfm_optimizer.zero_grad()

        t = torch.rand(batch_size, 1, device=device)

        if interpolant == 'linear':
            # Find the interval each t falls into
            # bucketize returns index of the *right* bin edge
            idx = torch.bucketize(t, times) - 1
            idx = idx.clamp(0, Xt.shape[0] - 2)

            # there are only single samples associated with each time stamp
            x_k = Xt[idx].squeeze(1)
            x_kp1 = Xt[idx + 1].squeeze(1)
            xt, ut = get_ot_interpolant_generalized(x_k, x_kp1, times[idx], times[idx + 1], t)
        else:
            splines = interpolate.CubicSpline(times.cpu().numpy(), Xt.cpu().numpy(), axis=0)
            t_np = t.cpu().numpy()
            xt = torch.tensor(splines(t_np), device=device, dtype=torch.float32).squeeze(1)
            ut = torch.tensor(splines(t_np, 1), device=device, dtype=torch.float32).squeeze(1)


        vt = cfm_model(torch.cat([xt, t], dim=-1))

        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        cfm_optimizer.step()
        losses.append(loss.item())
    return cfm_model, losses


def main(distribution):
    seed = 0
    size = 30
    batch_size = 128
    ot_fm = "independent"
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
    # X0 = torch.tensor(data[0][np.random.choice(np.arange(0, data[0].shape[0]), 64)], dtype=torch.float32).to(device)
    X0 = torch.tensor(data[1][0], dtype=torch.float32).to(device).unsqueeze(0)
    with torch.no_grad():
        cfm_traj = node.trajectory(X0, t_span=torch.linspace(0, 1, num_int_steps + 1),
        )
        cfm_traj = cfm_traj.cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.plot(cfm_traj[..., 0], cfm_traj[..., 1], color='blue', alpha=0.2)
    #plt.scatter(data[0][:, 0], data[0][:, 1], color='red', alpha=0.5, s=1)
    plt.scatter(data[1][:, 0], data[1][:, 1], color='red', label='real data', alpha=0.5, s=1)
    #plt.scatter(data[2][:, 0], data[2][:, 1], color='red', alpha=0.5, s=1)
    titles = {"linear": "OT-CFM Trajectories", "cubic": "OT-MMFM Trajectories"}
    plt.title(titles[interpolant])
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()



if __name__ == '__main__':
    main("knot")

