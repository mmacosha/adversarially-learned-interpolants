import torch as T
import numpy as np
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from learnable_interpolants import CorrectionInterpolant, AffineTransformInterpolant, GaussianProbabilityPath
import scanpy as sc
from utils import sample_interm


g_hidden = 64
interpolant = CorrectionInterpolant(2, g_hidden, 'linear', correction_scale_factor='sqrt')
# interpolant = AffineTransformInterpolant(2, g_hidden, 'linear')
# interpolant = GaussianProbabilityPath(2, g_hidden, 'linear', correction_scale_factor='sqrt')
opt_interp = T.optim.Adam(interpolant.parameters(), lr=1e-4)
discriminator = T.nn.Sequential(T.nn.Linear(3, 64), T.nn.ELU(),
                                T.nn.Linear(64, 64), T.nn.ELU(),
                                T.nn.Linear(64, 1), T.nn.Sigmoid())
opt_disc = T.optim.Adam(discriminator.parameters(), lr=1e-4)

otplan = OTPlanSampler('exact')
training_data = sc.read_h5ad("../data/Drosophila/drosophila_p100.h5ad")
true_data = sc.read_h5ad("../data/Drosophila/drosophila_p100.h5ad")

slides = T.tensor(np.arange(2, 16, dtype=np.float32)[::2])  # time stamps with observed noisy data
losses = []
bs = 128
reg_weight = .5
for it in range(20000):

    idx = T.multinomial(T.ones_like(slides), bs, replacement=True)
    sampled_slides = slides[idx]

    xhat_t = sample_interm(training_data, sampled_slides)

    x0 = sample_interm(true_data, T.ones_like(sampled_slides))
    x1 = sample_interm(true_data, T.ones_like(sampled_slides) * 16)

    x0, x1 = otplan.sample_plan(x0, x1)

    # embed the slide indices to the unit line
    t = (sampled_slides - 1) / (16 - 1)
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
        all_slides = np.arange(1, 17, dtype=np.float32)
        fig, axes = plt.subplots(3, all_slides.size, figsize=(20, 8), sharex=True, sharey=True)
        emd = []
        vis_bs = 200
        for i, slide_idx in enumerate(all_slides):
            t_scaler = slide_idx
            slide_idx = T.ones(vis_bs, dtype=T.float32) * slide_idx

            x_t = sample_interm(true_data, slide_idx)
            xhat_t = sample_interm(training_data, slide_idx)

            x0 = sample_interm(true_data, T.ones_like(slide_idx))
            x1 = sample_interm(true_data, T.ones_like(slide_idx) * 16)

            x0, x1 = otplan.sample_plan(x0, x1)

            # embed the slide indices to the unit line
            t = (slide_idx - 1) / (16 - 1)
            t = t.unsqueeze(-1)

            with T.no_grad():
                xt_fake = interpolant(x0, x1, t).detach()

            if (i > 0) and (i < 15):
                emd.append(wasserstein(x_t, xt_fake))

            axes[0, i].scatter(x_t[:, 0], x_t[:, 1], s=1)
            axes[0, i].set_title(f's = {int(all_slides[i])}')
            if t_scaler in slides:
                axes[1, i].scatter(xhat_t[:, 0], xhat_t[:, 1], s=1)
            else:
                axes[1, i].set_title('Unobserved')
            axes[2, i].scatter(xt_fake[:, 0], xt_fake[:, 1], s=1)
        axes[0, 0].set_ylabel(r'True Kernel, $\kappa(x|y_t)$')
        axes[1, 0].set_ylabel(r'Approx Kernel, $\hat{\kappa}(x|y_t)$')
        axes[2, 0].set_ylabel(r'Generator')
        print("Mean EMD:", np.mean(emd))

        for ax in axes[-1, :]:
            ax.set_xlabel('$x_1$')

        for ax_row in axes:
            for ax in ax_row:
                ax.set_aspect('equal')
                ax.grid(True)

        plt.tight_layout()
        plt.show()

        """
        x0 = sample_interm(true_data, T.ones(vis_bs))
        x1 = sample_interm(true_data, T.ones(vis_bs) * 16)

        x0, x1 = otplan.sample_plan(x0, x1)
        plt.scatter(x0[:, 0], x0[:, 1], s=1, color='red', label='s = 1')
        plt.scatter(x1[:, 0], x1[:, 1], s=1, color='blue', label='s = 16')

        for t in np.linspace(0., 1., 500, dtype=np.float32):
            t = T.ones((vis_bs, 1)) * t

            with T.no_grad():
                xt_fake = interpolant(x0, x1, t).detach()
            plt.scatter(xt_fake[:, 0], xt_fake[:, 1], s=0.1, color='black', label='Interpolations', alpha=0.5)
        plt.show()
        """


