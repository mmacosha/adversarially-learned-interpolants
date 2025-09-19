from pathlib import Path

import math
import torch
import numpy as np
import scanpy as sc
from typing import List
import pandas as pd

from sklearn.preprocessing import StandardScaler
from rotating_MNIST.create_dataset import create_dataset


# DATA_PATH = Path("/home/oskar/phd/interpolnet/Mixture-FMLs/Mixture-FMLs-kirill_single_cell_experiments/data")
DATA_PATH = Path("./data/single_cell")


class Dataset:
    def __init__(self, data, timesteps, normalize=True, normalization_type="minmax"):
        self.shift = 0.0
        self.scale = 1.0

        if normalize and normalization_type == "minmax":
            min_ = np.stack([x.min(0) for x in data]).min(0)
            max_ = np.stack([x.max(0) for x in data]).max(0)
            
            self.shift = min_
            self.scale = max_ - min_
            data = [self.normalize(data_t) for data_t in data]

        if normalize and normalization_type == "scale":
            self.shift = 0.0
            self.scale = math.sqrt(12)
            data = [self.normalize(data_t) for data_t in data]
        
        self.data = data
        self.timesteps = timesteps

    def __getitem__(self, index):
        return self.data[index]

    def normalize(self, datapoints):
        return (datapoints - self.shift) / self.scale

    def denormalize(self, datapoints):
        return datapoints * self.scale + self.shift

    def denormalize_gradfield(self, gradfield):
        return gradfield * self.scale


def normalize(x, min_max):
    if min_max is None:
        return x
    min_, max_ = (m.to(x.device) for m in min_max)
    return (x - min_) / (max_ - min_)


def denormalize(x, min_max):
    if min_max is None:
        return x
    min_, max_ = (m.to(x.device) for m in min_max)
    return x * (max_ - min_) + min_


def denormalize_gradfield(x, min_max):
    if min_max is None:
        return x
    min_, max_ = (m.to(x.device) for m in min_max)
    return x * (max_ - min_)


def get_dataset(
        name: str, 
        n_data_dims, 
        normalize: bool = True, 
        whiten: bool = False
    ) -> List[np.ndarray]:
    if name == "cite":
        adata = sc.read_h5ad(DATA_PATH / "op_cite_inputs_0.h5ad")
        
        times = adata.obs['day'].astype('category')
        unique_times = times.cat.categories
        
        data = adata.obsm['X_pca'][:, :n_data_dims]
        if whiten:
            data = StandardScaler().fit_transform(data)
        
        X = [data[times == t, :n_data_dims] for t in unique_times]

    elif name == "multi":
        adata = sc.read_h5ad(DATA_PATH / "op_train_multi_targets_0.h5ad")
        times = adata.obs['day'].unique()

        data = adata.obsm['X_pca'][:, :n_data_dims]
        if whiten:
            data = StandardScaler().fit_transform(data)

        X = [data[adata.obs['day'] == t,  :n_data_dims] for t in times]
    
    elif name=="eb":
        adata = np.load(DATA_PATH / "eb_velocity_v5.npz")
        times = np.unique(adata['sample_labels'])

        data = adata['pcs'][:, :n_data_dims]
        
        if whiten:
            data = StandardScaler().fit_transform(data)
        
        X = [data[adata['sample_labels'] == t] for t in times]

    elif name == "eb-2d":
        adata = sc.read_h5ad(DATA_PATH / "ebdata_v3.h5ad")
        n_times = len(adata.obs["sample_labels"].unique())

        data = adata.obsm["X_phate"]
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        
        X = [
            data[adata.obs["sample_labels"].cat.codes == t] for t in range(n_times)
        ]


    elif name == 'cell_tracking':

        boolean_masks = np.load(
            "/home/oskar/phd/interpolnet/Mixture-FMLs/cell_tracking/exports/Cell4_masks/mask_cell4_stack.npy")

        X = []

        for i in range(boolean_masks.shape[0]):
            coords = np.argwhere(boolean_masks[i])  # (y, x)

            coords = coords[:, [1, 0]]  # swap -> (x, y)

            X.append(coords)

    elif name == 'RotatingMNIST_train':
        X, _ = create_dataset(3, B=100, test=False)

    elif name == 'RotatingMNIST_test':
        _, X = create_dataset(3, B=10, test=True)

    elif name == "ST":
        ST_img_path = "/Users/oskarkviman/Documents/phd/mixture_FM_loss/data/ST_images"
        df_u2 = pd.read_csv(f'{ST_img_path}/aligned_spots/U2_tumor_coordinates.csv')
        df_u3 = pd.read_csv(f'{ST_img_path}/aligned_spots/U3_tumor_coordinates.csv')
        df_u4 = pd.read_csv(f'{ST_img_path}/aligned_spots/U4_tumor_coordinates.csv')
        df_u5 = pd.read_csv(f'{ST_img_path}/aligned_spots/U5_tumor_coordinates.csv')

        X0 = np.array(df_u2.iloc[:, -2:].values, dtype=np.float32)
        Xt1 = np.array(df_u3.iloc[:, -2:].values, dtype=np.float32)
        Xt2 = np.array(df_u4.iloc[:, -2:].values, dtype=np.float32)
        X1 = np.array(df_u5.iloc[:, -2:].values, dtype=np.float32)

        X = [X0, Xt1, Xt2, X1]

    else:
        raise ValueError(f"Unknown dataset {name}")
    
    X = [torch.from_numpy(x).float() for x in X]

    if normalize:
        min_ = torch.stack([x.min(0).values for x in X]).min(0).values
        max_ = torch.stack([x.max(0).values for x in X]).max(0).values

        # min_, max_ = math.sqrt(12), 0
        Xn = [(x - min_) / (max_ - min_) for x in X]
        return Xn, (min_, max_)
    else:
        return X, None



