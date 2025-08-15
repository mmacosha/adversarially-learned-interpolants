import torch as T
import numpy as np
import matplotlib.pyplot as plt
from torchcfm.optimal_transport import wasserstein
from tqdm import tqdm
import matplotlib.colors as mcolors

from generate_tme_data import sample_data
from learnable_interpolants import SpatialCorrectionInterpolant


device = 'cuda' if T.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

dataset = T.load('../data/boundaries/dataset_0.pt')
dims = dataset["true_distributions"].shape[-1]  # number of marker genes
print(dims)
g_hidden = 64 * 8
g_hidden_disc = 64 * 8

interpolant = SpatialCorrectionInterpolant(
    dims, g_hidden, 'linear', correction_scale_factor='sqrt', interpolnet_input='reference'
)
interpolant = interpolant.to(device)

discriminator = T.nn.Sequential(
    T.nn.Linear(dims + 1 + 1, g_hidden_disc), T.nn.ELU(),  # add 2 to input_dims to account for time and 1D coordinate
    T.nn.Linear(g_hidden_disc, g_hidden_disc), T.nn.ELU(),
    T.nn.Linear(g_hidden_disc, 1), T.nn.Sigmoid()
).to(device)

opt_interp = T.optim.Adam(interpolant.parameters(), lr=1e-4)
opt_disc = T.optim.Adam(discriminator.parameters(), lr=1e-4)

interm_slides = np.array([1, 2, 3])

losses = []
emds = []
emds_noisy = []
bs = 1048 * 2
reg_weight = 0.01
for it in tqdm(range(40000)):
    sampled_slides = T.tensor(np.random.choice(interm_slides, size=bs, replace=True), dtype=T.int)
    x0, x1, xhat_t, c = sample_data(bs, sampled_slides, dataset)
    x0, x1, xhat_t, c = x0.to(device), x1.to(device), xhat_t.to(device), c.to(device)

    t = sampled_slides / (len(interm_slides) + 2 - 1)
    t = t.unsqueeze(-1).to(device)
    c = c.unsqueeze(-1)

    opt_interp.zero_grad()
    xt_fake = T.cat([interpolant(x0, x1, t, c), t, c], 1)
    disc_score_fake = discriminator(xt_fake).log()  # (1-discriminator(xt_fake)).log() #
    loss_interp = -disc_score_fake.mean()
    loss_reg = interpolant.regularizing_term(x0, x1, t, xt_fake)
    (loss_interp + reg_weight * loss_reg).backward()
    opt_interp.step()

    opt_disc.zero_grad()
    xt_real = T.cat([xhat_t, t, c], 1)
    disc_score_real = discriminator(xt_real).log()
    disc_score_fake = (1 - discriminator(xt_fake.detach())).log()
    loss_disc = - (disc_score_real.mean() + disc_score_fake.mean())
    loss_disc.backward()
    opt_disc.step()

    losses.append((loss_interp.item(), loss_reg.item(), loss_disc.item()))

    if it % 1000 == 0:
        print(it, np.array(losses)[-100:].mean(0))
        slides = T.arange(0, 5, dtype=T.int)

        true_distributions = dataset['true_distributions']  # (n_spots, n_slides, n_marker_genes)
        noisy_distributions = dataset['noisy_distributions']
        spot_coordinates = dataset['spot_coordinates']  # (n_spots)
        n_spots = spot_coordinates.shape[0]
        offsets = dataset['offsets']
        boarders = dataset['boarders']

        true_marker_gene_intensities = true_distributions.sum(axis=-1)
        noisy_marker_gene_intensities = noisy_distributions.sum(axis=-1)

        emd = []
        emd_noisy = []
        predicted_intensities = T.zeros_like(true_marker_gene_intensities)
        for j, slide_id in enumerate(slides):
            c = spot_coordinates / n_spots

            t = slide_id / (len(interm_slides) + 2 - 1)
            t = T.ones(n_spots, dtype=T.float32, device=device) * t
            t = t.unsqueeze(-1).to(device)
            c = c.unsqueeze(-1).to(device)

            x0 = true_distributions[:, 0].to(device)
            x1 = true_distributions[:, -1].to(device)

            with T.no_grad():
                xt_fake = interpolant(x0, x1, t, c).to('cpu')

            x_t = true_distributions[:, j]
            xhat_t = noisy_distributions[:, j]

            if (j > 0.) and (j < 4.):
                emd.append(wasserstein(x_t, xt_fake, power=1) / dims)
                emd_noisy.append(wasserstein(xhat_t, xt_fake, power=1) / dims)

            predicted_intensities[:, slide_id] = xt_fake.sum(-1)

        print(np.mean(emd))
        sorted_idx = np.argsort(spot_coordinates)
        sorted_spot_coords = spot_coordinates[sorted_idx]
        sorted_true_intensities = true_marker_gene_intensities[sorted_idx]
        sorted_fake_intensities = predicted_intensities[sorted_idx]

        n_spots, n_slides = sorted_true_intensities.shape
        x = np.repeat(np.arange(n_slides), n_spots)  # Slide index repeated for each spot
        y = np.tile(sorted_spot_coords, n_slides)  # Spot coordinates tiled across slides
        color = sorted_true_intensities.T.flatten()  # Flattened intensity values
        color_fake = sorted_fake_intensities.T.flatten()

        vmin = min(color.min(), color_fake.min())
        vmax = max(color.max(), color_fake.max())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Plot
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(x, y, c=color, cmap='viridis_r', s=20, norm=norm)
        sc_noisy = plt.scatter(x + 0.1, y, c=color_fake, cmap='viridis_r', s=20, norm=norm)
        plt.scatter(np.arange(n_slides) - 0.08, boarders - 9, c='red', marker='_')
        plt.scatter(np.arange(n_slides) - 0.08, boarders + 9, c='red', marker='_')
        plt.scatter(np.arange(n_slides) + 0.1 + 0.08, boarders - 9, c='red', marker='_')
        plt.scatter(np.arange(n_slides) + 0.1 + 0.08, boarders + 9, c='red', marker='_')
        plt.xlabel('Slide Index')
        plt.ylabel('Spot Coordinate')
        plt.title('True vs. Generated Marker Gene Intensities per Spot and Slide')
        plt.colorbar(sc, label='Intensity')
        plt.tight_layout()
        plt.pause(0.5)
        plt.show()
