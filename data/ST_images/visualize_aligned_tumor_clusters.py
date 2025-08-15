import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd
import cv2
import math


slides = ["U2", "U3", "U4", "U5"]
n = len(slides)
cols = min(3, n)
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))

# Flatten axes array for easy indexing
axes = axes.flatten() if n > 1 else [axes]
coords_by_slide = {}
for i, u in enumerate(slides):

    rgb_image = cv2.imread(f"ref_U5_warped_images/{u}_aligned_to_U5.tif", cv2.IMREAD_UNCHANGED)
    # rgb_image = cv2.imread(f"downscaled_images/20201103_ST_HT206B1-S1Fc1{u}_10p.tif", cv2.IMREAD_UNCHANGED)

    # Load clustering result
    h5ad_u = sc.read_h5ad(f'annotations/HT206B1-S1Fc1{u}Z1B1-SeuratObj.h5ad')
    if u == "U2":
        clusters = [0, 2, 8, 12]
    elif u == "U3":
        clusters = [4, 7, 8, 9, 2]
    elif u == "U4":
        clusters = [1, 5, 6, 8, 10, 13]
    else:
        clusters = [3, 4, 5, 6, 10, 13]

    # Load spot coordinates once
    coords_u = pd.read_csv(f'aligned_spots/{u}_aligned_to_U5.csv', header=None)
    # coords_u = pd.read_csv(f'spot_coordinates/{"U5"}_coordinates.csv', header=None)

    ax = axes[i]
    ax.imshow(rgb_image[..., ::-1])

    slide_dfs = []
    for idx, cluster in enumerate(clusters):

        # Extract barcodes for this cluster
        # c = h5ad_u[(h5ad_u.obs.seurat_clusters == cluster) & (h5ad_u.obs.Tumor_purity_estimate > 0.8)]
        c = h5ad_u[(h5ad_u.obs.Tumor_purity_estimate > 0.8)]

        # c = h5ad_u[(h5ad_u.obs.seurat_clusters == cluster)]
        c_barcodes = c.obs_names.values

        # Subset spots belonging to this cluster
        coords_u_c = coords_u[np.isin(coords_u[0].values, c_barcodes)]
        slide_dfs.append(coords_u_c)
        x_spots = coords_u_c.iloc[:, -1].values.astype('float32')
        y_spots = coords_u_c.iloc[:, -2].values.astype('float32')

        ax.scatter(x_spots, y_spots, alpha=0.2, color='red')
        break  # break only if using tumor purity as a filter
    # coords = pd.read_csv(f"spot_coordinates/{u}_tumor_coordinates.csv", header=None)
    # ax.scatter(coords.iloc[:, -1], coords.iloc[:, -2], alpha=0.2, color='red')
    ax.set_title(f"{u}")
    ax.axis("off")
    coords_by_slide[u] = pd.concat(slide_dfs, ignore_index=True)

    # Hide unused axes if any
for j in range(len(slides), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()

for slide, df in coords_by_slide.items():
    df.to_csv(f"aligned_spots/{slide}_tumor_coordinates.csv", index=False, header=False)

