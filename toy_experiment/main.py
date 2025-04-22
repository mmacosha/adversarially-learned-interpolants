import torch as T
# from generate_circle_data import sample_interm
from generate_rotating_gaussians import sample_interm
import numpy as np
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import OTPlanSampler
from learnable_interpolants import CorrectionInterpolant, AffineTransformInterpolant


g_hidden = 64
# interpolant = CorrectionInterpolant(2, g_hidden, 'linear')
interpolant = AffineTransformInterpolant(2, g_hidden, 'linear')
opt_interp = T.optim.Adam(interpolant.parameters(), lr=1e-3)
discriminator = T.nn.Sequential(T.nn.Linear(3, 64), T.nn.ELU(), T.nn.Linear(64, 64), T.nn.ELU(), T.nn.Linear(64, 1), T.nn.Sigmoid()).type(T.float32)
opt_disc = T.optim.Adam(discriminator.parameters(), lr=1e-2)

otplan = OTPlanSampler('exact')

# assume we have data in the following time stamps
time_stamps = np.arange(1, 10) / 10

losses = []
bs = 1024 // 2
reg_weight = 0#.5
lam = 0.1
for it in range(20000):

    t = np.random.choice(time_stamps, size=bs)

    t = T.tensor(t, dtype=T.float32)
    x_t, xhat_t, y_t = sample_interm(bs, t, lam=lam)

    x0, *_ = sample_interm(bs, T.zeros_like(t))
    x1, *_ = sample_interm(bs, T.ones_like(t))

    x0, x1 = otplan.sample_plan(x0, x1)

    t = t.unsqueeze(-1)

    opt_interp.zero_grad()
    xt_fake = T.cat([interpolant(x0, x1, t), t], 1)
    disc_score_fake = discriminator(xt_fake).log()  # (1-discriminator(xt_fake)).log()
    loss_interp = - disc_score_fake.mean()
    # loss_reg = xt_fake[:, :-1] - (x1 * t + x0 * (1-t))
    # loss_reg = (loss_reg ** 2).sum(1).mean()
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
        for i, t in enumerate(time_steps):
            t = T.ones(bs, dtype=T.float32) * t
            x_t, xhat_t, y_t = sample_interm(bs, t, lam=lam)
            x0, *_ = sample_interm(bs, T.zeros_like(t))
            x1, *_ = sample_interm(bs, T.ones_like(t))
            x0, x1 = otplan.sample_plan(x0, x1)
            t = t.unsqueeze(-1)

            with T.no_grad():
                xt_fake = T.cat([interpolant(x0, x1, t), t], 1).detach()

            axes[0, i].scatter(x_t[:, 0], x_t[:, 1], s=1)
            axes[0, 5].set_title(f'$t$ = {time_stamps[4]:.1f}')
            axes[1, i].scatter(xhat_t[:, 0], xhat_t[:, 1], s=1)
            axes[2, i].scatter(xt_fake[:, 0], xt_fake[:, 1], s=1)
        axes[0,0].set_ylabel(r'True Kernel, $\kappa(x|y_t)$')
        axes[1,0].set_ylabel(r'Approx Kernel, $\hat{\kappa}(x|y_t)$')
        axes[2,0].set_ylabel(r'Generator')

        for ax in axes[-1,:]:
            ax.set_xlabel('$x_1$')

        for ax_row in axes:
            for ax in ax_row:
                ax.set_aspect('equal')
                ax.grid(True)

        plt.tight_layout()
        plt.show()
