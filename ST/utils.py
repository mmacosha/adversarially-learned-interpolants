import torch as T
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from torchcfm import OTPlanSampler

from ali_cfm.data_utils import denormalize


def sample_interm_4_slides(bs, t, x_t1, x_t2):
    n_spots_t1 = x_t1.shape[0]
    n_spots_t2 = x_t2.shape[0]
    dims = x_t1.shape[-1]

    n_t1 = T.sum(t == 0.25).item()
    n_t2 = T.sum(t == 0.75).item()

    spots_t1 = T.tensor(np.random.choice(n_spots_t1, n_t1, replace=True), dtype=T.int)
    spots_t2 = T.tensor(np.random.choice(n_spots_t2, n_t2, replace=True), dtype=T.int)

    interm_data = T.zeros((bs, dims), device=x_t1.device)
    interm_data[t == 0.25] = x_t1[spots_t1]
    interm_data[t == 0.75] = x_t2[spots_t2]
    return interm_data


def sample_slides(bs, t, x0, x_t1, x_t2, x1, eta=0.001):
    n_spots_0 = x0.shape[0]
    n_spots_t1 = x_t1.shape[0]
    n_spots_t2 = x_t2.shape[0]
    n_spots_1 = x1.shape[0]
    dims = x_t1.shape[-1]

    n_0 = T.sum(t == 0).item()
    n_t1 = T.sum(t == 0.25).item()
    n_t2 = T.sum(t == 0.75).item()
    n_1 = T.sum(t == 1).item()

    spots_0 = T.tensor(np.random.choice(n_spots_0, n_0, replace=True), dtype=T.int)
    spots_t1 = T.tensor(np.random.choice(n_spots_t1, n_t1, replace=True), dtype=T.int)
    spots_t2 = T.tensor(np.random.choice(n_spots_t2, n_t2, replace=True), dtype=T.int)
    spots_1 = T.tensor(np.random.choice(n_spots_1, n_1, replace=True), dtype=T.int)

    data = T.zeros((bs, dims), device=x_t1.device)
    data[t == 0.] = x0[spots_0]  # + T.rand_like(x0[spots_0]) * eta
    data[t == 0.25] = x_t1[spots_t1]
    data[t == 0.75] = x_t2[spots_t2]
    data[t == 1] = x1[spots_1]  # + T.rand_like(x1[spots_1]) * eta
    return data


class Plotter(object):
    def __init__(self, path_to_images, time_stamps, slide_names=None, coordinate_scaling=100, ds=None):
        self.rgb_images = []
        self.slide_names = [u for u in ["U2", "U3", "U4", "U5"]] if slide_names is None else slide_names
        for u in self.slide_names:
            img_name = f"{u}_aligned_to_U5.tif"
            path_to_image = os.path.join(path_to_images, img_name)
            # reverse channel order in preparation for plotting with plt.imshow
            self.rgb_images.append(cv2.imread(path_to_image, cv2.IMREAD_UNCHANGED)[..., ::-1])
        self.time_stamps = time_stamps
        self.n_slides = len(self.time_stamps)
        self.coordinate_scaling = coordinate_scaling
        self.ds = ds

    def overlay_spot_coordinates(self, s_t, slide, ax, observed):
        # x_list is a list of coordinate matrices, e.g. containing four matrices of interpolants
        # observed is a boolean array indicating if a slide was used during training
        t = self.time_stamps[slide]
        ax.imshow(self.rgb_images[slide])
        s_t = s_t * self.ds.scale.cpu() + self.ds.shift.cpu() if self.ds is not None else s_t
        x = s_t[:, 1] * self.coordinate_scaling
        y = s_t[:, 0] * self.coordinate_scaling
        ax.scatter(x, y, alpha=0.2, color='red')
        #ax.set_title(f""
        #             f"{self.slide_names[slide]}" +
        #             " (Observed;" * observed +
        #             " (Unobserved;" * (1 - observed) +
        #             f" t = {t})"
        #             )
        ax.axis("off")

    def plot_fn(self, interpolant, epoch, seed, t_max, data, ot_sampler, device, metric_prefix, train_timesteps, wandb, min_max):
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        x0 = denormalize(data[0], min_max)
        x1 = denormalize(data[-1], min_max)
        N = x0.shape[0]
        for slide, s_t in enumerate(data):
            self.overlay_spot_coordinates(s_t.to('cpu'), slide, axes[0, slide], data[slide])

            t = self.time_stamps[slide]
            t = T.ones(N, dtype=T.float32, device=x0.device) * t
            t = t.unsqueeze(-1)
            if isinstance(interpolant, list):
                xt_fake = interpolant[slide]
            else:
                with T.no_grad():
                    xt_fake = interpolant(x0, x1, t)
            self.overlay_spot_coordinates(xt_fake.to('cpu'), slide, axes[1, slide], observed_list[slide])
        plt.tight_layout()
        if wandb is not None:
            wandb.log({f"{metric_prefix}/interpolants": wandb.Image(fig), f"{metric_prefix}_step": epoch})
            plt.close(fig)
        else:
            plt.show()

    def plot_interpolants(self, interpolant, X, observed_list, mmot_interpolants=None):
        # if plotting mmot-based interpolants, like cubic splines, OT-CFM or MFM, input a list of interpolants
        # as the mmot_interpolants argument
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        x0 = X[0]
        x1 = X[-1]
        N = x0.shape[0]
        for slide, s_t in enumerate(X):
            self.overlay_spot_coordinates(s_t.to('cpu'), slide, axes[0, slide], observed_list[slide])

            t = self.time_stamps[slide]
            t = T.ones(N, dtype=T.float32, device=x0.device) * t
            t = t.unsqueeze(-1)
            if mmot_interpolants is not None:
                xt_fake = mmot_interpolants[slide]
            else:
                with T.no_grad():
                    xt_fake = interpolant(x0, x1, t)
            self.overlay_spot_coordinates(xt_fake.to('cpu'), slide, axes[1, slide], observed_list[slide])
        plt.tight_layout()
        plt.show()

    def plot_data(self, X):
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        for slide, s_t in enumerate(X):
            ax = axes[1, slide]
            t = self.time_stamps[slide]
            ax.imshow(self.rgb_images[slide])
            x = s_t[:, 1] * self.coordinate_scaling
            y = s_t[:, 0] * self.coordinate_scaling
            ax.scatter(x, y, alpha=0.2, color='red')
            ax.axis("off")

            ax = axes[0, slide]
            ax.imshow(self.rgb_images[slide])
            ax.set_title(f""
                         f"{self.slide_names[slide]}" +
                         f" (t = {t})", fontsize=20
                         )
            ax.axis("off")
        plt.tight_layout()
        plt.show()

def pre_compute_OT_minibatches(X0, X1, bs, n_batches):
    coupled_indices = T.zeros((n_batches, bs, 2), device=X0.device, dtype=T.int)
    otplan = OTPlanSampler('exact')
    for batch in range(n_batches):
        x0_indices = T.randint(0, X0.size(0), (bs,), device=X0.device)
        x0 = X0[x0_indices]
        x1_indices = T.randint(0, X1.size(0), (bs,), device=X0.device)
        x1 = X1[x1_indices]

        pi = otplan.get_map(x0, x1)

        probs = T.from_numpy(pi[T.arange(bs)])  # (bs, bs)
        probs = probs / T.sum(probs, 1, keepdim=True)
        idxs = T.multinomial(probs, num_samples=1).squeeze(1)

        # idx_x0, idx_x1 = otplan.sample_map(pi, bs, replace=False)
        coupled_indices[batch, :, 0], coupled_indices[batch, :, 1] = x0_indices, idxs
    return coupled_indices


def pad_a_like_b(a, b):
    """Pad a to have the same number of dimensions as b."""
    if isinstance(a, float | int):
        return a
    return a.reshape(-1, *([1] * (b.dim() - 1)))


class Dataset:
    def __init__(self, data, timesteps, normalize=True, normalization_type="minmax", device='cuda:0'):
        self.shift = 0.0
        self.scale = 1.0

        if normalize and normalization_type == "minmax":
            min_ = np.stack([x.min(0) for x in data]).min(0)
            max_ = np.stack([x.max(0) for x in data]).max(0)

            self.shift = min_
            self.scale = max_ - min_
            data = [self.normalize(data_t) for data_t in data]

        # if normalize and normalization_type == "scale":
        #     self.shift = 0.0
        #     self.scale = math.sqrt(12)
        #     data = [self.normalize(data_t) for data_t in data]

        self.data = data
        self.timesteps = timesteps
        self.shift = T.tensor(self.shift, dtype=T.float32).to(device)
        self.scale = T.tensor(self.scale, dtype=T.float32).to(device)

    def __getitem__(self, index):
        return self.data[index]

    def normalize(self, datapoints):
        return (datapoints - self.shift) / self.scale

    def denormalize(self, datapoints):
        return datapoints * self.scale + self.shift

    def denormalize_gradfield(self, gradfield):
        return gradfield * self.scale


def mmot_couple_marginals(X0, X1, Xt, otplan):
    """
        Samples bs triplets (x0, xt, x1) from the factorized joint
           π*(x0, xt, x1) ∝ π1*(x0, xt) · π2*(xt, x1) / μt(xt)
        assuming μt is uniform.

        Args:
            minibatch samples from the three marginals
            X0: (bs, d)
            X1: (bs, d)
            Xt: (bs, d)
            bs: number of samples to draw (with replacement)

        Returns:
            aligned: Tensor of shape (bs, 3, d)
        """
    bs, d = X0.shape

    device = X0.device

    # 1) compute the two pairwise plans as numpy arrays
    pi1_np = otplan.get_map(X0, Xt)  # shape (n0, nt)
    pi2_np = otplan.get_map(Xt, X1)  # shape (nt, n1)


    # 2) convert to torch and move to device
    pi1 = T.from_numpy(pi1_np).to(device)  # (n0, nt)
    pi2 = T.from_numpy(pi2_np).to(device)  # (nt, n1)

    idx_t = T.tensor(np.arange(0, bs), dtype=T.int, device=device)

    # 4) sample x0 | xt  using columns of pi1
    probs0 = pi1[:, idx_t].t()
    probs0 = probs0 / probs0.sum(dim=1, keepdim=True)
    idx_0 = T.multinomial(probs0, num_samples=1, replacement=True).squeeze(1)

    # 5) sample x1 | xt using rows of pi2
    probs2 = pi2[idx_t, :]  # (bs, n1)
    probs2 = probs2 / probs2.sum(dim=1, keepdim=True)
    idx_1 = T.multinomial(probs2, num_samples=1, replacement=True).squeeze(1)

    # 6) return the coupled points
    return X0[idx_0], X1[idx_1]




if __name__ == '__main__':
    pass
    # X = load_data(scale_factor=100, device='cpu')
    # pl = Plotter("/home/oskar/phd/interpolnet/Mixture-FMLs/data/ST_images/ref_U5_warped_images",
    #              [0, 0.25, 0.75, 1])
    # pl.plot_data(X)

