import torch as T
import numpy as np
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from learnable_interpolants import CorrectionInterpolant, AffineTransformInterpolant, GaussianProbabilityPath
from utils import sample_interm, load_data


device = 'cuda' if T.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

g_hidden = 512
interpolant = CorrectionInterpolant(2, g_hidden, 'linear', correction_scale_factor=None, interpolnet_input='')
# interpolant = AffineTransformInterpolant(2, g_hidden, 'linear')
# interpolant = GaussianProbabilityPath(2, g_hidden, 'linear', correction_scale_factor=None)
interpolant = interpolant.to(device)

discriminator = T.nn.Sequential(
    T.nn.Linear(3, g_hidden), T.nn.ELU(),
    T.nn.Linear(g_hidden, g_hidden), T.nn.ELU(),
    T.nn.Linear(g_hidden, 1), T.nn.Sigmoid()
).to(device)

opt_interp = T.optim.Adam(interpolant.parameters(), lr=1e-5)
opt_disc = T.optim.Adam(discriminator.parameters(), lr=1e-4)

scale_factor = 10
X = load_data(scale_factor)

# PRE-PROCESS MINIBATCHES
otplan = OTPlanSampler('exact')
pi = otplan.get_map(X[0], X[-1])
idx_x0, idx_x1 = otplan.sample_map(pi, 4000, replace=True)
N = min([x.shape[0] for x in X])
eval_idx_x0, eval_idx_x1 = otplan.sample_map(pi, N, replace=True)


n_observed = 3
observed_slides = T.tensor(np.linspace(0, 15, n_observed, dtype=int))
losses = []
bs = 512
reg_weight = 0.001  # best performing = 0.01 or 0.001
for it in range(40000):

    # sample intermediate slide indices
    idx = T.randint(1, n_observed - 1, (bs,))
    sampled_slides = observed_slides[idx]

    idx = T.randint(0, idx_x0.size, (bs,))
    x0, x1 = X[0][idx_x0[idx]], X[-1][idx_x1[idx]]
    x0, x1 = x0.to(device), x1.to(device)

    # embed the slide indices to the unit line
    t = sampled_slides / (16 - 1)
    t = t.unsqueeze(-1).to(device)

    opt_interp.zero_grad()
    xt_fake = T.cat([interpolant(x0, x1, t), t], 1)
    disc_score_fake =  (1-discriminator(xt_fake)).log() # discriminator(xt_fake).log()  #
    loss_interp = disc_score_fake.mean()
    loss_reg = interpolant.regularizing_term(x0, x1, t, xt_fake)
    (loss_interp + reg_weight * loss_reg).backward()
    opt_interp.step()

    xhat_t = sample_interm(X, sampled_slides)
    xhat_t = xhat_t.to(device)
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
        all_slides = np.arange(0, 16, dtype=np.float32)
        # fig, axes = plt.subplots(2, all_slides.size, figsize=(30, 4), sharex=True, sharey=True)
        emd = []
        emd_observed = []
        x0, x1 = X[0][eval_idx_x0], X[-1][eval_idx_x1]
        x0, x1 = x0.to(device), x1.to(device)
        for i, slide_idx in enumerate(all_slides):
            slide_scaler = slide_idx
            slide_idx = T.ones(N, dtype=T.int) * slide_idx

            # embed the slide indices to the unit line
            t = slide_idx / (16 - 1)
            t = t.unsqueeze(-1).to(device)

            with T.no_grad():
                xt_fake = interpolant(x0, x1, t).detach()

            if i not in observed_slides:
                emd.append(wasserstein(xt_fake * scale_factor, X[i].to(device) * scale_factor, power=1))
            else:
                if (i > 0) & (i < 15):
                    emd_observed.append(wasserstein(xt_fake * scale_factor, X[i].to(device) * scale_factor, power=1))

            """
            axes[0, i].scatter(X[i][:, 0], X[i][:, 1], s=1)
            axes[0, i].set_title(f's = {int(all_slides[i])}')
            if slide_scaler not in observed_slides:
                axes[0, i].set_title('Unobserved')
            axes[1, i].scatter(xt_fake[:, 0].to('cpu'), xt_fake[:, 1].to('cpu'), s=1)
        axes[0, 0].set_ylabel(r'True Kernel')
        axes[1, 0].set_ylabel(r'Generator')
        for ax in axes[-1, :]:
            ax.set_xlabel('$x_1$')

        for ax_row in axes:
            for ax in ax_row:
                ax.set_aspect('equal')
                ax.grid(True)

        plt.tight_layout()
        plt.show()
        """

        print("Mean EMD (Unobserved):", np.mean(emd))
        print("Mean EMD (Observed):", np.mean(emd_observed))

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


