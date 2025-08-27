from matplotlib import pyplot as plt
from knot_distribution import loop_distribution
from trimodal_distribution import generate_marginals
import numpy as np
from train_ali import train as train_interpolants
from train_ali import plot_knot
from learnable_interpolants import CorrectionInterpolant
import torch
from tqdm.auto import trange
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
import os
from torchcfm.optimal_transport import OTPlanSampler
from ST.utils import mmot_couple_marginals


class MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, w), torch.nn.SELU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def generate_data(seed, distribution, size):
    np.random.seed(seed)
    if distribution == 'knot':
        X0, Xt, X1, times = loop_distribution(size, std=0.1)
    else:
        X0, Xt, X1 = generate_marginals(size)
        times = np.ones(size) * 0.5
    data = [X0, Xt, X1]
    return data, times


def train_ali_cfm(data, interpolant, cfm_model, cfm_optimizer, batch_size, n_epochs, device, ot=False):
    X0, Xt, X1 = (torch.tensor(X, dtype=torch.float32).to(device) for X in data)
    losses = []

    if ot:
        otplan = OTPlanSampler('exact')
        pi = otplan.get_map(X0, X1)
    else:
        otplan = None

    for step in trange(n_epochs, desc="Training CFM", leave=False):
        cfm_optimizer.zero_grad()

        x0 = X0[np.random.choice(np.arange(0, X0.shape[0]), batch_size)]
        x1 = X1[np.random.choice(np.arange(0, X1.shape[0]), batch_size)]
        if ot == 'ot':
            i, j = otplan.sample_map(pi, batch_size, replace=True)
            x0, x1 = X0[i], X1[j]
        elif ot == 'mmot':
            xt = Xt[np.random.choice(np.arange(0, Xt.shape[0]), batch_size)]
            x0, x1 = mmot_couple_marginals(x0, x1, xt, otplan)

        t = torch.rand(x0.shape[0], 1, device=device)

        xt = interpolant(x0, x1, t).detach()
        ut = interpolant.dI_dt(x0, x1, t).detach()

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
    train_ali = False
    ot_gan = "ot"
    gan_loss = 'vanilla'
    ot_fm = "ot"

    data, times = generate_data(seed, distribution, size)
    dims = data[0].shape[-1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.manual_seed(seed)
    g_hidden = 128
    interpolant = CorrectionInterpolant(dims, g_hidden).to(device)

    d_hidden = 128
    discriminator = torch.nn.Sequential(
        torch.nn.Linear(dims + 1, d_hidden), torch.nn.ELU(),
        torch.nn.Linear(d_hidden, d_hidden), torch.nn.ELU(),
        torch.nn.Linear(d_hidden, 1)
    ).to(device)

    gan_optimizer_G = torch.optim.Adam(interpolant.parameters(), lr=1e-3)
    gan_optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-3)

    if train_ali:
        interpolant = train_interpolants(interpolant, discriminator, data, train_timesteps=times,
                                         gan_optimizer_D=gan_optimizer_D, gan_optimizer_G=gan_optimizer_G,
                                         n_epochs=100_001, seed=seed, batch_size=batch_size, correct_coeff=1.,
                                         device=device, distribution=distribution, ot=ot_gan, gan_loss=gan_loss)
        torch.save(interpolant.state_dict(), "interpolant_models_toy_data/" + ot_gan + "-" +
               distribution + str(seed) + '.pth')
    else:
        interpolant.load_state_dict(torch.load("interpolant_models_toy_data/" + ot_fm + "-" + distribution + str(seed) + '.pth'))
        X0, Xt, X1 = (torch.tensor(X, dtype=torch.float32).to(device) for X in data)
        if ot_gan == 'ot':
            from torchcfm.optimal_transport import OTPlanSampler
            otplan = OTPlanSampler('exact')
            pi = otplan.get_map(X0, X1)
        else:
            otplan, pi = None, None
        plot_knot(X0, Xt, X1, times, device, interpolant, None, pi, otplan)

    cfm_model = MLP(dims, time_varying=True, w=64).to(device)
    cfm_optimizer = torch.optim.Adam(cfm_model.parameters(), 1e-3)
    cfm_model, losses = train_ali_cfm(data, interpolant, cfm_model, cfm_optimizer, batch_size, n_epochs=2_001,
                                      device=device, ot=ot_fm)
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

    if distribution == 'knot':
        plt.figure(figsize=(5, 5))
        plt.rcParams.update({'font.size': 15})
        plt.plot(cfm_traj[..., 0], cfm_traj[..., 1], color='blue', alpha=0.2)
        plt.scatter(data[0][:, 0], data[0][:, 1], color='red', alpha=0.5, s=1)
        xt = data[1].reshape(-1, 2)
        labels = np.tile(np.arange(size - 2) % 10, 10)

        plt.scatter(data[1][..., 0], data[1][..., 1], color='red', alpha=0.5, s=1)
        # plt.scatter(xt[..., 0], xt[..., 1], c=labels, alpha=0.5, s=1, cmap='tab10')
        # plt.scatter(data[2][:, 0], data[2][:, 1], color='red', alpha=0.5, s=1)
        # plt.title(f"ALI-CFM Trajectories")
        plt.xlim(-3.5, 3.5)
        plt.tight_layout()
        # plt.legend(loc='upper right')

        save_dir = f"{distribution}_ali_frames"
        filename = os.path.join(save_dir, f"ALI_CFM.png")
        plt.savefig(filename, dpi=150)
        plt.show()
    else:
        plot_trimodal_cfm(data[0], data[1], data[2], cfm_traj)


def plot_trimodal_cfm(X0, Xt, X1, cfm_traj):
    t_marginal_times = [0.0, 0.5, 1.0]
    marginal_data = [X0, Xt, X1]
    marginal_colors = ['red', 'green', 'blue']
    marginal_labels = ['t=0', 't=0.5', 't=1']

    fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))

    ax1.plot(torch.linspace(0, 1, cfm_traj.shape[0]), cfm_traj.squeeze(-1), 'b-', alpha=0.4, linewidth=1.5)

    for t_val, data, color, label in zip(t_marginal_times, marginal_data, marginal_colors, marginal_labels):
        # Create histogram
        hist, bin_edges = np.histogram(data, bins=25, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Scale and offset histogram to show as vertical distribution
        hist_scaled = hist * 0.02  # Scale factor for visibility

        # Plot as filled area
        for j in range(len(bin_centers)):
            ax1.fill_betweenx([bin_centers[j] - (bin_edges[1] - bin_edges[0]) / 2,
                               bin_centers[j] + (bin_edges[1] - bin_edges[0]) / 2],
                              t_val, t_val + hist_scaled[j],
                              alpha=0.7, color=color)

        # Add vertical line at time point
        ax1.axvline(t_val, color=color, linestyle='--', alpha=0.8, linewidth=2, label=label)

    ax1.set_xlabel('time t', fontsize=12)
    ax1.set_ylabel('position x', fontsize=12)
    ax1.set_title(f'OT-ALI-CFM with marginal distributions', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(f"trimodal_ali_plots/OT-ALI-CFM.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main("knot")

