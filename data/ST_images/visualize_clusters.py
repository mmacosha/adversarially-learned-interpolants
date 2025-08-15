import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import cv2
import math

u = "U3"
rgb_image = cv2.imread(f"downscaled_images/20201103_ST_HT206B1-S1Fc1{u}_10p.tif", cv2.IMREAD_UNCHANGED)

# Load clustering result
h5ad_u = sc.read_h5ad(f'annotations/HT206B1-S1Fc1{u}Z1B1-SeuratObj.h5ad')
clusters = np.unique(h5ad_u.obs.seurat_clusters)

# Layout for subplots
n_clusters = len(clusters)
cols = min(4, n_clusters)  # up to 4 columns
rows = math.ceil(n_clusters / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
if rows == 1 and cols == 1:
    axes = np.array([[axes]])
elif rows == 1 or cols == 1:
    axes = axes.reshape(rows, cols)

# Load spot coordinates once
coords_u = pd.read_csv(f'spot_coordinates/{u}_coordinates.csv', header=None)

for idx, cluster in enumerate(clusters):
    row, col = divmod(idx, cols)
    ax = axes[row, col]

    # Extract barcodes for this cluster
    c = h5ad_u[(h5ad_u.obs.seurat_clusters == cluster) & (h5ad_u.obs.Tumor_purity_estimate > 0.8)]
    c_barcodes = c.obs_names.values

    # Subset spots belonging to this cluster
    coords_u_c = coords_u[np.isin(coords_u[0].values, c_barcodes)]
    x_spots = coords_u_c.iloc[:, -1].values.astype('float32') / 10
    y_spots = coords_u_c.iloc[:, -2].values.astype('float32') / 10

    # Plot
    ax.imshow(rgb_image[..., ::-1])
    ax.scatter(x_spots, y_spots, alpha=0.2, color='red')
    ax.set_title(f"Cluster {cluster}")
    ax.axis("off")

# Hide unused axes if any
for i in range(n_clusters, rows * cols):
    row, col = divmod(i, cols)
    axes[row, col].axis("off")

plt.tight_layout()
plt.show()

