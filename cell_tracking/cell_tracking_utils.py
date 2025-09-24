import numpy as np

import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from ali_cfm.training.training_utils import sample_gan_batch
from ali_cfm.data_utils import denormalize
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


class CellOverlayViewer:
    def __init__(self, image_dir, method='ali', ext=".tif", animate=True):
        """
        Load raw images when initializing.

        Args:
            image_dir (str): Path to directory with raw image frames.
            ext (str): File extension to filter images (default: .tif).
        """
        image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(ext))
        if not image_files:
            raise ValueError(f"No image files with extension {ext} found in {image_dir}")

        self.images = [imageio.imread(os.path.join(image_dir, f)) for f in image_files]
        self.num_frames = len(self.images)
        self.save_base_path = f"/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/inferred_gifs/interpolants/"
        self.save_path = self.save_base_path + f"{method}_inferred.gif"
        self.animate = animate

    def overlay_masks(self, inferred_masks, cmap="viridis", point_size=6, alpha=0.7):
        """
        Overlay inferred masks on the raw images, static plots.

        Args:
            inferred_masks (list[np.ndarray]): Each entry is (N,2) numpy array of (x,y) coords.
            cmap (str): Colormap for scatter points.
            point_size (int): Scatter marker size.
            alpha (float): Transparency of scatter markers.
        """
        if len(inferred_masks) != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} masks, got {len(inferred_masks)}"
            )

        for i, (img, mask_coords) in enumerate(zip(self.images, inferred_masks)):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img, cmap="gray")

            if mask_coords is not None and len(mask_coords) > 0:
                xs, ys = mask_coords[:, 0], mask_coords[:, 1]
                sc = ax.scatter(xs, ys, c=np.arange(len(xs)), cmap=cmap,
                                s=point_size, alpha=alpha)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.05)
                cbar = plt.colorbar(sc, cax=cax)
                cbar.set_label("mask index")

            ax.set_title(f"Frame {i}")
            ax.axis("off")
            plt.tight_layout()
            plt.show()

    def animate_masks(self, inferred_masks, data, interval=200, point_size=6, alpha=0.01):
        """
        Create an animation of masks overlaid on raw images.

        Args:
            inferred_masks np.ndarray]: (n_frames, N,2) arrays of coords.
            interval (int): Delay between frames in ms.
        """
        if inferred_masks.shape[0] != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} masks, got {len(inferred_masks)}"
            )

        fig, ax = plt.subplots(figsize=(6, 6))
        im_raw = ax.imshow(self.images[0], cmap="gray")
        scatter_data = ax.scatter([], [], color='red',
                                      s=point_size, alpha=alpha)
        scatter_inferred = ax.scatter([], [], color='blue',
                             s=point_size, alpha=alpha)
        ax.axis("off")

        def update(frame):
            im_raw.set_data(self.images[frame])

            coords_dat = data[frame]
            scatter_data.set_offsets(coords_dat)

            coords_inf = inferred_masks[frame]
            scatter_inferred.set_offsets(coords_inf)

            ax.set_title(f"Frame {frame}")
            return im_raw, scatter_data, scatter_inferred

        anim = animation.FuncAnimation(
            fig, update, frames=self.num_frames, interval=interval, blit=False
        )

        anim.save(self.save_path, writer="imagemagick")
        plt.close()


    def plot_fn(self, interpolant, epoch, seed, t_max, data, ot_sampler, device, metric_prefix,
                train_timesteps, wandb, min_max, method="ali", animate=None):
        animate = self.animate if animate is None else animate
        if method == "ali":
            with torch.no_grad():
                batch = sample_gan_batch(data, data[0].shape[0], divisor=t_max, ot_sampler=ot_sampler, ot="border",
                                         times=train_timesteps)
                x0, x1, _, _ = (x.to(device) for x in batch)
                inferred_mask = torch.zeros((len(train_timesteps), x0.shape[0], 2), device=device)
                for i, t_ in enumerate(train_timesteps):
                    t = t_ / t_max * torch.ones((x0.shape[0], 1), device=device)
                    xt = interpolant(x0, x1, t, training=False)
                    inferred_mask[i] = denormalize(xt, min_max)
        elif method == "ali-cfm":
            inferred_mask = interpolant.to(device)
        else:
            inferred_mask = denormalize(interpolant, min_max).to(device)

        if animate:
            try:
                self.save_path = self.save_base_path + f"{method}.gif"
                self.animate_masks(inferred_mask.cpu().numpy(), [denormalize(d, min_max) for d in data])
            except Exception as e:
                print(f"Animation failed: {e}. Proceeding with static plots.")


        n_steps, K, _ = inferred_mask.shape

        pts = inferred_mask.reshape(-1, 2).cpu().numpy()
        tvals = torch.repeat_interleave(torch.tensor(np.arange(n_steps), device=device) / (n_steps - 1), K).cpu().numpy()

        # 2D scatter plot of inferred masks
        fig, ax = plt.subplots(figsize=(6, 5))
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=tvals, cmap="viridis", norm=norm,
                        s=6, alpha=0.5)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = plt.colorbar(sc, cax=cax)
        cbar.set_label(r"$t$")
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_ylim(200, 500)
        ax.set_xlim(200, 500)
        plt.tight_layout()
        if wandb is not None:
            wandb.log({f"{metric_prefix}/scatter": wandb.Image(fig), f"{metric_prefix}_step": epoch})
            plt.close(fig)
        else:
            plt.show()

        # 3D scatter plot of inferred masks

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(pts[:, 0], pts[:, 1], tvals, c=tvals, cmap="viridis", norm=norm, s=6, alpha=0.5)
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label(r"$t$")
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("t")
        ax.set_xlim(200, 500)
        ax.set_ylim(200, 500)
        ax.set_zlim(0, 1)
        plt.tight_layout()
        if wandb is not None:
            wandb.log({f"{metric_prefix}/scatter3d": wandb.Image(fig), f"{metric_prefix}_step": epoch})
            plt.close(fig)
        else:
            plt.show()

        # T = len(data)
        # points = np.concatenate(data, axis=0)
        # labels = np.concatenate([np.full(len(arr), i) for i, arr in enumerate(data)])
        # tvals = labels / (T - 1)
        #
        # fig, ax = plt.subplots(figsize=(6, 5))
        # norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        # sc = ax.scatter(points[:, 0], points[:, 1], c=tvals, cmap="viridis", norm=norm, s=6, alpha=0.3)
        #
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="3%", pad=0.05)
        # cbar = plt.colorbar(sc, cax=cax)
        # cbar.set_label("time")
        # cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        # cbar.set_ticklabels(["start", "¼", "½", "¾", "end"])
        #
        # plt.tight_layout()
        # if wandb is not None:
        #     wandb.log({f"{metric_prefix}/scatter_real": wandb.Image(fig), f"{metric_prefix}_step": epoch})
        #     plt.close(fig)
        # else:
        #     plt.show()
