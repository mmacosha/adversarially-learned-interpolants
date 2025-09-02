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
for frame_idx in range(1, 2):


    # Specific cell label to overlay (integer)
    cell_label = 4

    # --- Load the raw image ---
    # Assumes frame files are consistently named, e.g. 'img000.tif', 'img001.tif', ...
    image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')])
    raw_image_path = os.path.join(IMAGE_DIR, image_files[frame_idx])
    raw = imageio.imread(raw_image_path)

    # --- Load corresponding labeled mask image ---
    mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.tif')])
    mask_path = os.path.join(MASK_DIR, mask_files[frame_idx])
    mask = imageio.imread(mask_path)

    # --- Extract coordinates of the specific cell ---
    ys, xs = np.where(mask == cell_label)
    coords = np.vstack([xs, ys]).T  # shape: (num_pixels, 2)

    # --- Overlay and visualize ---
    plt.figure(figsize=(8, 8))
    plt.imshow(raw, cmap='gray')
    plt.scatter(xs, ys, s=1, c='red', alpha=0.6)  # overlay cell pixels as red dots
    plt.title(f'Frame {frame_idx}, Cell label {cell_label} ({len(xs)} pixels)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
