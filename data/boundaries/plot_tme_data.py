import torch as T
import numpy as np
import matplotlib.pyplot as plt


dataset = T.load('dataset_0.pt')

"""
{
   'true_distributions': true_distributions,
   'noisy_distributions': noisy_distributions,
   'spot_coordinates': spot_coordinates,
   'boarders': boarders
}
"""

true_distributions = dataset['true_distributions']  # (n_spots, n_slides, n_marker_genes)
noisy_distributions = dataset['noisy_distributions']
spot_coordinates = dataset['spot_coordinates']  # (n_spots)
boarders = dataset['boarders']  # (n_slides)
true_marker_gene_intensities = true_distributions.sum(axis=-1)
noisy_marker_gene_intensities = noisy_distributions.sum(axis=-1)

sorted_idx = np.argsort(spot_coordinates)
sorted_spot_coords = spot_coordinates[sorted_idx]
sorted_true_intensities = true_marker_gene_intensities[sorted_idx]
sorted_noisy_intensities = noisy_marker_gene_intensities[sorted_idx]

n_spots, n_slides = sorted_true_intensities.shape
x = np.repeat(np.arange(n_slides), n_spots)  # Slide index repeated for each spot
y = np.tile(sorted_spot_coords, n_slides)    # Spot coordinates tiled across slides
c = sorted_true_intensities.T.flatten()           # Flattened intensity values
c_noisy = sorted_noisy_intensities.T.flatten()

# Plot
plt.figure(figsize=(10, 6))
sc = plt.scatter(x, y, c=c, cmap='viridis_r', s=20)
sc_noisy = plt.scatter(x + 0.1, y, c=c_noisy, cmap='viridis_r', s=20)
plt.scatter(np.arange(n_slides) - 0.08, boarders - 9, c='red', marker='_')
plt.scatter(np.arange(n_slides) - 0.08, boarders + 9, c='red', marker='_')
plt.scatter(np.arange(n_slides) + 0.1 + 0.08, boarders - 9, c='red', marker='_')
plt.scatter(np.arange(n_slides) + 0.1 + 0.08, boarders + 9, c='red', marker='_')
plt.xlabel('Slide Index')
plt.ylabel('Spot Coordinate')
plt.title('True vs. Noisy Marker Gene Intensities per Spot and Slide')
plt.colorbar(sc, label='Intensity')
plt.tight_layout()
plt.show()