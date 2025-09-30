import torch
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from torch.utils.data import Subset
import numpy as np

from typing import List
import random


def create_dataset(digit=0, B=500, seed=42, test=False):
    """
    Creates timestamped, non-overlapping rotated MNIST subsets for a given digit.

    Returns:
        train_dataset, test_dataset: Lists of arrays (one for each t),
            each of shape [B, 28, 28]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    TIMESTAMPS = np.arange(17) / 16 if test else np.arange(9) / 8

    # --- Transform (resize to 14x14, pad to 28x28) ---
    base_transform = v2.Compose([
        v2.Resize((14, 14)),
        v2.Pad(padding=1)  # Pad to 28x28,
        # v2.Pad(padding=0)
    ])

    # --- Load full datasets (no transform yet, we'll apply manually) ---
    raw_train = datasets.MNIST(root='./data', train=True, download=True)
    raw_test = datasets.MNIST(root='./data', train=False, download=True)

    def extract_digit_images(dataset):
        idxs = [i for i, (_, label) in enumerate(dataset) if label == digit]
        return [dataset[i][0] for i in idxs]

    # Filter only images with specified digit
    train_imgs = extract_digit_images(raw_train)
    test_imgs = extract_digit_images(raw_test)

    # Make sure we have enough for all timestamps
    min_n = min(len(train_imgs), len(test_imgs))
    n_per_time = B
    total_needed = n_per_time * len(TIMESTAMPS)
    assert total_needed <= min_n, f"Not enough images of digit {digit} to sample {n_per_time}×{len(TIMESTAMPS)} without replacement."

    # Shuffle and split without replacement
    train_imgs = random.sample(train_imgs, total_needed)
    test_imgs = random.sample(test_imgs, total_needed)

    # --- Rotate and transform ---
    def process_images(imgs: List, test = False) -> List[np.ndarray]:
        arrays = []
        for i, t in enumerate(TIMESTAMPS):
            chunk = imgs[i*B:(i+1)*B]
            angle = t * 360
            processed = []
            for img in chunk:
                img = TF.rotate(img, angle=angle)
                img = base_transform(img)
                img_np = 2 * (np.array(img) / 255.0) - 1  # Normalize to [-1, 1]
                processed.append(img_np.reshape(img_np.shape[1] ** 2))
            arrays.append(np.stack(processed))
        return arrays

    train_dataset = process_images(train_imgs, False)
    test_dataset = process_images(test_imgs, True)

    return train_dataset, test_dataset

if __name__ == '__main__':
    TIMESTAMPS = np.arange(9) / 8
    train_data, test_data = create_dataset(digit=3, B=100, seed=42)
    print("Train data shapes (t=0..8):", [d.shape for d in train_data])
    print("Test data shapes (t=0..8):", [d.shape for d in test_data])
    # Example: visualize first image of each timestamp
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(train_data), figsize=(15, 3))
    for i, ax in enumerate(axes):
        dim = int(np.sqrt(train_data[i][0].shape[-1]))
        ax.imshow(train_data[i][0].reshape(dim, dim), cmap='gray')
        ax.set_title(f"t={360 * TIMESTAMPS[i]:.0f}°")
        ax.axis('off')
    plt.tight_layout()
    plt.show()