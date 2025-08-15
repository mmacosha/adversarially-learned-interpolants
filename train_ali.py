import numpy as np
import torch
from tqdm.auto import trange
from torch import nn
import matplotlib.pyplot as plt
import os
import imageio
import glob
from torchcfm.optimal_transport import OTPlanSampler


def create_gif(save_dir='frames', output_file='distribution.gif', fps=5):
    frame_paths = sorted(glob.glob(os.path.join(save_dir, 'frame_*.png')))
    frames = [imageio.v2.imread(path) for path in frame_paths]
    imageio.mimsave(output_file, frames, fps=fps)


def sample_gan_batch(X, batch_size, times):
    # X shape (size, dims)
    idx = np.random.choice(np.arange(0, X.shape[0]), batch_size)
    return times[idx], X[idx]



def train(interpolant, discriminator, data,
    gan_optimizer_G, gan_optimizer_D, n_epochs,
    batch_size, correct_coeff, train_timesteps, seed, distribution, plot_frequency=10_000, device='cpu', ot=False):

    torch.manual_seed(seed)

    X0, Xt, X1 = (torch.tensor(X, dtype=torch.float32).to(device) for X in data)
    train_timesteps_np = train_timesteps.copy()
    train_timesteps = torch.tensor(train_timesteps, dtype=torch.float32).to(device).view(-1, 1)

    if ot:
        otplan = OTPlanSampler('exact')
        pi = otplan.get_map(X0, X1)
    else:
        pi, otplan = None, None


    curr_epoch = 0
    for epoch in trange(curr_epoch, n_epochs,
                        desc="Training GAN Interpolant", leave=False):
        if epoch > 40_000:
            gan_optimizer_G.param_groups[0]['lr'] = 1e-5
            gan_optimizer_D.param_groups[0]['lr'] = 5e-5
        elif epoch > 100_000:
            gan_optimizer_G.param_groups[0]['lr'] = 1e-6
            gan_optimizer_D.param_groups[0]['lr'] = 5e-6

        curr_epoch += 1

        t, xt = sample_gan_batch(Xt, batch_size, train_timesteps)
        if ot:
            i, j = otplan.sample_map(pi, batch_size, replace=True)
            x0, x1 = X0[i], X1[j]
        else:
            x0 = X0[np.random.choice(np.arange(0, X0.shape[0]), batch_size)]
            x1 = X1[np.random.choice(np.arange(0, X1.shape[0]), batch_size)]
        xt_fake = interpolant(x0, x1, t)

        real_inputs = torch.cat([xt, t], dim=-1)
        fake_inputs = torch.cat([xt_fake.detach(), t], dim=-1)

        real_proba = discriminator(real_inputs)
        fake_proba = discriminator(fake_inputs)

        # Train discriminator
        gan_optimizer_D.zero_grad()
        d_real_loss = nn.functional.softplus(-real_proba).mean()
        d_fake_loss = nn.functional.softplus(fake_proba).mean()
        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        gan_optimizer_D.step()

        # Train generator
        gan_optimizer_G.zero_grad()
        fake_inputs = torch.cat([xt_fake, t], dim=-1)
        fake_proba = discriminator(fake_inputs)

        g_loss_ = nn.functional.softplus(-fake_proba).mean()
        reg_weight_loss = interpolant.regularizing_term(x0, x1, t, xt_fake)
        g_loss = g_loss_ + correct_coeff * reg_weight_loss

        g_loss.backward()
        gan_optimizer_G.step()

        if epoch % plot_frequency == 0:
            if distribution == 'knot':
                plot_knot(X0, Xt, X1, train_timesteps_np, device, interpolant, epoch)
            else:
                plot_trimodal(X0, Xt, X1, device, interpolant, epoch, pi, otplan)

    return interpolant


def plot_trimodal(X0, Xt, X1, device, interpolant, epoch, pi, otplan):
    plt_bs = 64
    with torch.no_grad():
        if pi is not None:
            i, j = otplan.sample_map(pi, plt_bs, replace=True)
            x0, x1 = X0[i], X1[j]
        else:
            x0 = X0[np.random.choice(np.arange(0, X0.shape[0]), plt_bs)]
            x1 = X1[np.random.choice(np.arange(0, X1.shape[0]), plt_bs)]

        t = torch.linspace(0, 1, 100)
        t = torch.tensor(np.tile(t.reshape((-1, 1)), plt_bs), dtype=torch.float32,
                         device=device).unsqueeze(-1)
        x0_expanded = x0.unsqueeze(0).expand(t.shape[0], -1, -1)
        x1_expanded = x1.unsqueeze(0).expand(t.shape[0], -1, -1)

        xt_fake = interpolant(x0_expanded, x1_expanded, t, training=False).cpu().numpy()

        t_marginal_times = [0.0, 0.5, 1.0]
        marginal_data = [X0, Xt, X1]
        marginal_colors = ['red', 'green', 'blue']
        marginal_labels = ['t=0', 't=0.5', 't=1']

        fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))

        ax1.plot(torch.linspace(0, 1, 100), xt_fake.squeeze(-1), 'b-', alpha=0.4, linewidth=1.5)

        for t_val, data, color, label in zip(t_marginal_times, marginal_data, marginal_colors, marginal_labels):
            # Create histogram
            hist, bin_edges = np.histogram(data.cpu().numpy(), bins=25, density=True)
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
        ax1.set_title(f'Epoch {epoch} ALIs with marginal distributions', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xlim(-0.05, 1.05)

        plt.tight_layout()
        plt.savefig(f"trimodal_ali_plots/epoch{epoch}.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_knot(X0, Xt, X1, train_timesteps_np, device, interpolant, epoch):
    plt_bs = 64
    with torch.no_grad():
        x0 = X0[np.random.choice(np.arange(0, X0.shape[0]), plt_bs)]
        x1 = X1[np.random.choice(np.arange(0, X1.shape[0]), plt_bs)]
        t = torch.tensor(np.tile(train_timesteps_np.reshape((-1, 1)), plt_bs), dtype=torch.float32,
                         device=device)
        t0 = torch.zeros((1, plt_bs), device=device)
        t1 = torch.ones((1, plt_bs), device=device)
        t = torch.cat((torch.cat((t0, t), dim=0), t1), dim=0).unsqueeze(-1)

        x0_expanded = x0.unsqueeze(0).expand(t.shape[0], -1, -1)
        x1_expanded = x1.unsqueeze(0).expand(t.shape[0], -1, -1)

        xt_fake = interpolant(x0_expanded, x1_expanded, t, training=False).cpu().numpy()

        plt.figure(figsize=(5, 5))
        plt.plot(xt_fake[..., 0], xt_fake[..., 1], color='blue', alpha=0.2)
        plt.scatter(x0[:, 0].cpu().numpy(), x0[:, 1].cpu().numpy(), color='red', alpha=0.5, s=1)
        plt.scatter(x1[:, 0].cpu().numpy(), x1[:, 1].cpu().numpy(), color='red', alpha=0.5, s=1)
        plt.scatter(Xt[:, 0].cpu().numpy(), Xt[:, 1].cpu().numpy(), color='red', label='real data', alpha=0.5, s=1)
        plt.title(f"Epoch {epoch}")
        plt.tight_layout()
        plt.legend(loc='upper right')

        save_dir = "knot_ali_frames"
        filename = os.path.join(save_dir, f"frame_{epoch:06d}.png")
        plt.savefig(filename, dpi=150)
        plt.close()

if __name__ == '__main__':
    create_gif("knot_ali_frames", "knot.gif")
