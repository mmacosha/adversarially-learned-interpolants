import matplotlib.pyplot as plt
import torch
import numpy as np

from ali_cfm.training.training_utils import sample_gan_batch
from ali_cfm.data_utils import denormalize


PLOT_TIMESTAMPS = np.arange(17, dtype=np.float32) / 16.

def plot_fn(interpolant, epoch, seed, t_max, data, ot_sampler, device, metric_prefix, train_timesteps, wandb, min_max):
    with torch.no_grad():
        batch = sample_gan_batch(data, data[0].shape[0], divisor=t_max, ot_sampler=ot_sampler, ot="border",
                                 times=train_timesteps)
        x0, x1, _, _ = (x.to(device) for x in batch)
        inferred_mask = torch.zeros((len(PLOT_TIMESTAMPS), x0.shape[0], x0.shape[-1]), device=device)
        fig, axes = plt.subplots(2, len(PLOT_TIMESTAMPS) // 2, figsize=(15, 3))
        row, col = 0, 0
        for i, t_ in enumerate(PLOT_TIMESTAMPS[:-1]):
            t = t_ / max(PLOT_TIMESTAMPS) * torch.ones((x0.shape[0], 1), device=device)
            xt = interpolant(x0, x1, t, training=False)
            inferred_mask[i] = denormalize(xt, min_max)
            dim = int(np.sqrt(x0.shape[-1]))
            axes[row, col].imshow(inferred_mask[i][0].reshape(dim, dim).cpu().numpy(), cmap='gray')
            axes[row, col].set_title(f"t={360 * PLOT_TIMESTAMPS[i]:.0f}°")
            axes[row, col].axis('off')
            col += 1
            if col >= axes.shape[1]:
                col = 0
                row += 1
        if wandb is not None:
            wandb.log({f"{metric_prefix}/interpolants": wandb.Image(fig), f"{metric_prefix}_step": epoch})
            plt.close(fig)
        else:
            plt.show()
