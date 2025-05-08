import torch as T
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def load_data(scale_factor, device):
    df_u2 = pd.read_csv('../data/ST_images/aligned_spots/U2_tumor_coordinates.csv')
    df_u3 = pd.read_csv('../data/ST_images/aligned_spots/U3_tumor_coordinates.csv')
    df_u4 = pd.read_csv('../data/ST_images/aligned_spots/U4_tumor_coordinates.csv')
    df_u5 = pd.read_csv('../data/ST_images/aligned_spots/U5_tumor_coordinates.csv')

    X0 = T.tensor(df_u2.iloc[:, -2:].values, dtype=T.float32,
                  device=device) / scale_factor  # scale down to stabilize training
    Xt1 = T.tensor(df_u3.iloc[:, -2:].values, dtype=T.float32, device=device) / scale_factor
    Xt2 = T.tensor(df_u4.iloc[:, -2:].values, dtype=T.float32, device=device) / scale_factor
    X1 = T.tensor(df_u5.iloc[:, -2:].values, dtype=T.float32, device=device) / scale_factor
    return X0, Xt1, Xt2, X1

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


class Plotter(object):
    def __init__(self, path_to_images, time_stamps, slide_names=None, coordinate_scaling=100):
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

    def overlay_spot_coordinates(self, s_t, slide, ax, observed):
        # x_list is a list of coordinate matrices, e.g. containing four matrices of interpolants
        # observed is a boolean array indicating if a slide was used during training
        t = self.time_stamps[slide]
        ax.imshow(self.rgb_images[slide])
        x = s_t[:, 1] * self.coordinate_scaling
        y = s_t[:, 0] * self.coordinate_scaling
        ax.scatter(x, y, alpha=0.2, color='red')
        ax.set_title(f""
                     f"{self.slide_names[slide]}" +
                     " (Observed)" * observed +
                     " (Unobserved)" * (1 - observed) +
                     f"t = {t}"
                     )
        ax.axis("off")

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

