import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# --- User parameters ---
# Path to image sequence directory (raw images for each frame)
IMAGE_DIR = '/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/data/PhC-C2DH-U373/01'  # adjust as needed
# Path to corresponding labeled masks (TRA folder)
MASK_DIR = '/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/data/PhC-C2DH-U373/01_ST/SEG/'  # adjust as needed

# Which frame index to visualize (zero-based)
# Select 5 frame indices to visualize
frame_indices = np.linspace(0, 114, 6, dtype=int)

cell_label = 4

fig, axes = plt.subplots(2, 3, figsize=(10, 10))
plt.rcParams.update({'font.size': 15})

centroids = []

# First row: single frames with segmentation masks
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')])
col = 0
row = 0
# for i, frame_idx in enumerate(frame_indices):
for i in range(115):
    # Load raw image
    raw_image_path = os.path.join(IMAGE_DIR, image_files[i])
    raw = imageio.imread(raw_image_path)

    # Load mask
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.tif')])
    mask_path = os.path.join(MASK_DIR, mask_files[i])
    mask = imageio.imread(mask_path)

    # Extract cell coordinates
    ys, xs = np.where(mask == cell_label)
    coords = np.vstack([xs, ys]).T

    # Compute centroid
    if len(xs) > 0:
        centroid = (np.mean(xs), np.mean(ys))
    else:
        centroid = (np.nan, np.nan)
    centroids.append(centroid)

    # Plot
    if i in frame_indices:
        ax = axes[row, col % 3]
        ax.imshow(raw, cmap='gray')
        ax.scatter(xs, ys, s=1, c='red', alpha=0.6)
        ax.set_title(f'Frame {i} (114)', fontsize=15)
        ax.axis('off')
        col += 1
        row += int(col % 3 == 0)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 1, figsize=(8, 8))
plt.rcParams.update({'font.size': 15})
# Second row: trajectory of centroid
centroids = np.array(centroids)
cmap = plt.get_cmap('viridis')
colors = cmap(np.linspace(0, 1, len(centroids)))

ax_traj = axes  # Center subplot for trajectory
for i, (centroid, color) in enumerate(zip(centroids, colors)):
    if not np.isnan(centroid[0]):
        ax_traj.scatter(centroid[0], centroid[1], color=color, s=60, edgecolor='k', zorder=2)
        if i > 0 and not np.isnan(centroids[i-1][0]):
            ax_traj.plot([centroids[i-1][0], centroid[0]], [centroids[i-1][1], centroid[1]], color=color, linewidth=2, zorder=1)

ax_traj.set_title('Cell centroid trajectory', fontsize=15)
ax_traj.set_xlabel('X', fontsize=15)
ax_traj.set_ylabel('Y', fontsize=15)
ax_traj.invert_yaxis()
ax_traj.axis('equal')

plt.tight_layout()
plt.show()