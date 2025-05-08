import torch as T
import numpy as np
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from tqdm import tqdm
from utils import sample_interm_4_slides, Plotter, load_data
from learnable_interpolants import CorrectionInterpolant




device = 'cuda' if T.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# LOAD DATA
scale_factor = 100
X0, Xt1, Xt2, X1 = load_data(scale_factor, device)

# INIT PLOT OBJECT
pl = Plotter("../data/ST_images/ref_U5_warped_images",
             [0., 0.25, 0.75, 1.], coordinate_scaling=scale_factor)

# PRE-PROCESS MINIBATCHES
otplan = OTPlanSampler('exact')
pi = otplan.get_map(X0, X1)
idx_x0, idx_x1 = otplan.sample_map(pi, 4000, replace=True)

# INITIALIZE MODELS
g_hidden = 512
interpolant = CorrectionInterpolant(2, g_hidden, 'linear', correction_scale_factor='sqrt', interpolnet_input='reference')
# interpolant = AffineTransformInterpolant(2, g_hidden, 'linear')
# interpolant = GaussianProbabilityPath(2, g_hidden, 'linear', correction_scale_factor=None)
interpolant = interpolant.to(device)

discriminator = T.nn.Sequential(
    T.nn.Linear(3, g_hidden), T.nn.ELU(),
    T.nn.Linear(g_hidden, g_hidden), T.nn.ELU(),
    T.nn.Linear(g_hidden, 1), T.nn.Sigmoid()
).to(device)

lr = 1e-4
opt_interp = T.optim.Adam(interpolant.parameters(), lr=lr)
opt_disc = T.optim.Adam(discriminator.parameters(), lr=lr)

time_stamps = [0.75]
bs = 512
n_real_mini_batches = 5
reg_weight = .5
losses = []
for it in range(20000):

    t = np.random.choice(time_stamps, size=bs)
    t = T.tensor(t, dtype=T.float32, device=device)

    # idx = T.multinomial(T.ones(idx_x0.size), bs, replacement=False)
    idx = T.randint(0, idx_x0.size, (bs,))
    x0, x1 = X0[idx_x0[idx]], X1[idx_x1[idx]]

    t = t.unsqueeze(-1)

    opt_interp.zero_grad()
    xt_fake = T.cat([interpolant(x0, x1, t), t], 1)
    disc_score_fake = (1-discriminator(xt_fake)).log() #discriminator(xt_fake).log()  #
    loss_interp = disc_score_fake.mean()
    loss_reg = interpolant.regularizing_term(x0, x1, t, xt_fake)
    (loss_interp + reg_weight * loss_reg).backward()
    opt_interp.step()

    opt_disc.zero_grad()
    disc_score_real = 0
    for b in range(n_real_mini_batches):
        xhat_t = sample_interm_4_slides(bs, t.squeeze(-1), Xt1, Xt2)
        xt_real = T.cat([xhat_t, t], 1)
        disc_score_real += discriminator(xt_real).log() / n_real_mini_batches
    disc_score_fake = (1 - discriminator(xt_fake.detach())).log()
    loss_disc = - (disc_score_real.mean() + disc_score_fake.mean())
    loss_disc.backward()
    opt_disc.step()

    losses.append((loss_interp.item(), loss_reg.item(), loss_disc.item()))

    if it % 1000 == 0:
        print(it, np.array(losses)[-100:].mean(0))

        N = max(X0.shape[0], Xt1.shape[0], Xt2.shape[0], X1.shape[0])
        idx = T.randint(0, idx_x0.size, (N,))
        x0, x1 = X0[idx_x0[idx]], X1[idx_x1[idx]]
        pl.plot_interpolants(interpolant, [x0, Xt1, Xt2, x1], [1, 0, 1, 1])

        t = T.ones(N, dtype=T.float32, device=x0.device) * 0.25
        t = t.unsqueeze(-1)
        with T.no_grad():
            xt_fake = interpolant(x0, x1, t)
        print(wasserstein(Xt1, xt_fake, power=1))