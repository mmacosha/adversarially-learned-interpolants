import matplotlib.pyplot as plt
import numpy as np

from cell_tracking.data import denormalize
from cell_tracking_utils import CellOverlayViewer
import torch
from ali_cfm.data_utils import get_dataset


def plot_centroids(inferred_masks):
    inferred_masks = inferred_masks.numpy()
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, 115))

    fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    centroids = []
    for i, color in enumerate(colors):
        xs, ys = inferred_masks[i][..., 0], inferred_masks[i][..., 1]
        centroid = (np.mean(xs), np.mean(ys))
        centroids.append(centroid)
        axes.scatter(centroid[0], centroid[1], color=color, s=60, edgecolor='k', zorder=2)
        if i > 0:
            axes.plot([centroids[i-1][0], centroid[0]], [centroids[i-1][1], centroid[1]],
                      color=color, linewidth=2, zorder=1)
    axes.set_xlabel('$x$', fontsize=20)
    axes.set_ylabel('$y$', fontsize=20)
    axes.tick_params(labelsize=20)


    axes.set_ylim(275, 475)
    axes.set_xlim(240, 450)
    axes.invert_yaxis()
    # axes.axis('equal')
    plt.show()




def main_plotting(cfm_traj, data, min_max, subset_data, path_to_images="", method='ali', ext=".tif", animate=True):
    """
    Plot trajectories over raw images.

    Args:
        cfm_traj (torch.Tensor): Shape (T, N, 2) tensor of (x,y) coords over time.
        path_to_images (str): Path to directory with raw image frames.
        method (str): Method name for saving the GIF.
        ext (str): File extension to filter images (default: .tif).
        animate (bool): Whether to create an animated GIF or static plots.
    """
    frame_indices = [0, 25, 50, 75, 100, 114]
    inferred_masks = torch.tensor(np.array([cfm_traj[t] for t in range(len(cfm_traj))]))



    cov = CellOverlayViewer(image_dir=path_to_images, method=method, ext=ext, animate=animate)
    plot_centroids(denormalize(inferred_masks, min_max))
    cov.plot_fn(inferred_masks, epoch=None, seed=42, t_max=114, data=data, device='cpu',
                ot_sampler=None, metric_prefix=None, train_timesteps=None, wandb=None, min_max=min_max, method=method)

    if animate:
        cov.overlay_masks(inferred_masks, cmap="viridis", point_size=6, alpha=1.)
    else:
        path = f"/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/scatter_overlayed/{method}"
        cov.overlay_masks_subplot(inferred_masks, cmap="viridis", point_size=10, alpha=1.,
                          frame_indices=frame_indices, min_max=min_max, path=path)


if __name__ == '__main__':
    data, min_max = get_dataset("cell_tracking", 2,
                                True, False)
    np.random.seed(42)
    subset_data = [x[np.random.choice(np.arange(0, x.shape[0]), 10, replace=False)] for x in data]

    path_to_images = '/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/data/PhC-C2DH-U373/01/'

    # main_plotting(subset_data, data, min_max, subset_data, path_to_images=path_to_images, method='data', animate=False)

    ckpt_path = "/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/traj_ckpts/mfm_traj.pt"
    cfm_traj =  torch.load(ckpt_path, map_location="cpu")["trajectory"]
    cfm_traj_np = cfm_traj.cpu().numpy()
    main_plotting(cfm_traj, data, None, subset_data, path_to_images=path_to_images, method='mfm', animate=False)

    # ckpt_path = "/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/traj_ckpts/ali_cfm_traj.pt"
    # cfm_traj =  torch.load(ckpt_path, map_location="cpu")["trajectory"]
    # cfm_traj_np = cfm_traj.cpu().numpy()
    # main_plotting(cfm_traj, data, None, subset_data, path_to_images=path_to_images, method='ali_cfm', animate=False)

    # ckpt_path = "/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/traj_ckpts/linear_traj.pt"
    # cfm_traj = torch.load(ckpt_path, map_location="cpu")["trajectory"]
    # cfm_traj_np = cfm_traj.cpu().numpy()
    # main_plotting(cfm_traj, data, None, subset_data, path_to_images=path_to_images, method='ot_cfm', animate=False)