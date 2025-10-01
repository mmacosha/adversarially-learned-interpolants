import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- User params ---
IMAGE_DIR = '/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/data/PhC-C2DH-U373/01/'   # Raw image frames
MASK_DIR = '/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/data/PhC-C2DH-U373/01_ST/SEG/'    # Labeled segmentation masks
cell_label = 4                             # The cell ID you want to animate

# Gather and sort filenames
image_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif'))
mask_files = sorted(f for f in os.listdir(MASK_DIR) if f.endswith('.tif'))
num_frames = 114

# Setup figure
fig, ax = plt.subplots(figsize=(6, 6))
raw = imageio.imread(os.path.join(IMAGE_DIR, image_files[0]))
mask = imageio.imread(os.path.join(MASK_DIR, mask_files[0])) == cell_label

# Initial display
im_raw = ax.imshow(raw, cmap='gray')
im_mask = ax.imshow(np.ma.masked_where(~mask, mask), cmap='jet', alpha=0.5)
ax.axis('off')
ax.set_title(f'Cell {cell_label} — Frame 0')

# Update function for animation
def update(frame):
    raw = imageio.imread(os.path.join(IMAGE_DIR, image_files[frame]))
    mask = imageio.imread(os.path.join(MASK_DIR, mask_files[frame])) == cell_label
    im_raw.set_data(raw)
    im_mask.set_data(np.ma.masked_where(~mask, mask))
    ax.set_title(f'Cell {cell_label} — Frame {frame}')
    return im_raw, im_mask

# Create animation
anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=200, blit=True)

# Display
plt.show()

# Optional: save to file
# anim.save('cell_animation.mp4', writer='ffmpeg')
anim.save('cell_animation.gif', writer='imagemagick')
