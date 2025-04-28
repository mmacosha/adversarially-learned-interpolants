import torch as T
# from generate_circle_data import sample_interm
from generate_rotating_gaussians import sample_interm
import numpy as np
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from learnable_interpolants import CorrectionInterpolant, AffineTransformInterpolant, GaussianProbabilityPath
import time


device = 'cuda' if T.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

g_hidden = 256
# interpolant = CorrectionInterpolant(2, g_hidden, 'linear', correction_scale_factor=None)
# interpolant = AffineTransformInterpolant(2, g_hidden, 'linear')
interpolant = GaussianProbabilityPath(2, g_hidden, 'linear', correction_scale_factor=None)
interpolant = interpolant.to(device)

discriminator = T.nn.Sequential(
    T.nn.Linear(3, g_hidden), T.nn.ELU(),
    T.nn.Linear(g_hidden, g_hidden), T.nn.ELU(),
    T.nn.Linear(g_hidden, 1), T.nn.Sigmoid()
).to(device)

lr = 1e-3
opt_interp = T.optim.Adam(interpolant.parameters(), lr=lr)
opt_disc = T.optim.Adam(discriminator.parameters(), lr=lr)

otplan = OTPlanSampler('exact')

X0, *_ = sample_interm(2000, T.zeros(2000))
X1, *_ = sample_interm(2000, T.ones(2000))
X0, X1 = X0.to(device), X1.to(device)
pi = otplan.get_map(X0, X1)
idx_x0, idx_x1 = otplan.sample_map(pi, 4000, replace=True)

# assume we have data in the following time stamps
time_stamps = (np.arange(1, 10) / 10)[::2]
# observed time stamps are array([0.1, 0.3, 0.5, 0.7, 0.9])

losses = []
bs = 128
reg_weight = .05
lam = 0.2
for it in range(20000):

    t = np.random.choice(time_stamps, size=bs)
    t = T.tensor(t, dtype=T.float32, device=device)

    x_t, xhat_t, _ = sample_interm(bs, t.to('cpu'), lam=lam)
    x_t, xhat_t= x_t.to(device), xhat_t.to(device)

    idx = T.multinomial(T.ones(idx_x0.size), bs, replacement=False)
    x0, x1 = X0[idx_x0[idx]], X1[idx_x1[idx]]

    t = t.unsqueeze(-1)

    opt_interp.zero_grad()
    xt_fake = T.cat([interpolant(x0, x1, t), t], 1)
    disc_score_fake = (1-discriminator(xt_fake)).log() # discriminator(xt_fake).log()  #
    loss_interp = disc_score_fake.mean()
    loss_reg = interpolant.regularizing_term(x0, x1, t, xt_fake)
    (loss_interp + reg_weight * loss_reg).backward()
    opt_interp.step()

    opt_disc.zero_grad()
    xt_real = T.cat([xhat_t, t], 1)
    disc_score_real = discriminator(xt_real).log()
    disc_score_fake = (1-discriminator(xt_fake.detach())).log()
    loss_disc = - (disc_score_real.mean() + disc_score_fake.mean())
    loss_disc.backward()
    opt_disc.step()

    losses.append((loss_interp.item(), loss_reg.item(), loss_disc.item()))

    if it % 1000 == 0:
        print(it, np.array(losses)[-100:].mean(0))
        time_steps = np.arange(0, 11) / 10
        fig, axes = plt.subplots(3, 11, figsize=(15, 6), sharex=True, sharey=True)
        emd = []
        for i, t in enumerate(time_steps):
            t_scaler = t
            t = T.ones(bs, dtype=T.float32, device=device) * t
            x_t, xhat_t, y_t = sample_interm(bs, t.to('cpu'), lam=lam)

            # idx_x0, idx_x1 = otplan.sample_map(pi, bs, replace=False)
            # x0, x1, = X0[idx_x0], X1[idx_x1]
            idx = T.multinomial(T.ones(idx_x0.size), bs, replacement=False)
            x0, x1 = X0[idx_x0[idx]], X1[idx_x1[idx]]
            t = t.unsqueeze(-1)

            with T.no_grad():
                xt_fake = interpolant(x0, x1, t).detach().to('cpu')

            emd.append(wasserstein(x_t, xt_fake))

            axes[0, i].scatter(x_t[:, 0], x_t[:, 1], s=1)
            axes[0, i].set_title(f'$t$ = {time_steps[i]:.1f}')
            if t_scaler in time_stamps:
                axes[1, i].scatter(xhat_t[:, 0], xhat_t[:, 1], s=1)
            else:
                axes[1, i].set_title('Unobserved')
            axes[2, i].scatter(xt_fake[:, 0], xt_fake[:, 1], s=1)
        axes[0,0].set_ylabel(r'True Kernel, $\kappa(x|y_t)$')
        axes[1,0].set_ylabel(r'Approx Kernel, $\hat{\kappa}(x|y_t)$')
        axes[2,0].set_ylabel(r'Generator')
        print("Mean EMD:", np.mean(emd))

        for ax in axes[-1,:]:
            ax.set_xlabel('$x_1$')

        for ax_row in axes:
            for ax in ax_row:
                ax.set_aspect('equal')
                ax.grid(True)

        plt.tight_layout()
        plt.show()
