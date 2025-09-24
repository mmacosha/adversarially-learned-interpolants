#!/usr/bin/env python
"""Regenerate trajectories and plots from a saved MFM checkpoint."""

import argparse
from pathlib import Path

import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ali_cfm.data_utils import denormalize, get_dataset
from rotating_MNIST.train_mfm import (
    TrainConfig,
    FlowMLPWrapper,
    FlowAdapter,
    load_cell_tracking_stack,
    plot_cell_trajectories_3d,
)
from mfm.networks.utils import flow_model_torch_wrapper
from torchdyn.core import NeuralODE


def build_dataset(cfg: TrainConfig):
    dataset_name = cfg.dataset.lower()

    if dataset_name == "rotating_mnist":
        data, min_max = get_dataset(
            "RotatingMNIST_train", cfg.n_data_dims, normalize=cfg.normalize_dataset
        )
        test_data, _ = get_dataset(
            "RotatingMNIST_test", cfg.n_data_dims, normalize=cfg.normalize_dataset
        )
    elif dataset_name == "cell_tracking":
        if cfg.cell_stack_path is None:
            raise ValueError("checkpoint config missing cell_stack_path")
        data, min_max = load_cell_tracking_stack(
            cfg.cell_stack_path,
            subset_size=cfg.cell_subset_size,
            seed=cfg.cell_subset_seed,
            normalize=cfg.normalize_dataset,
        )
        if cfg.cell_test_stack_path:
            test_data, _ = load_cell_tracking_stack(
                cfg.cell_test_stack_path,
                subset_size=cfg.cell_subset_size,
                seed=cfg.cell_subset_seed,
                normalize=cfg.normalize_dataset,
            )
        else:
            test_data = [frame.clone() for frame in data]
    elif dataset_name == "st":
        base_dir = (
            Path(cfg.st_data_dir).expanduser()
            if cfg.st_data_dir is not None
            else Path("data/ST/nicola_data")
        )
        file_order = [
            "U2_tumor_coordinates.csv",
            "U3_tumor_coordinates.csv",
            "U4_tumor_coordinates.csv",
            "U5_tumor_coordinates.csv",
        ]
        frames = []
        frames = []
        for name in file_order:
            csv_path = base_dir / name
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing ST CSV file: {csv_path}")
            arr = pd.read_csv(csv_path).iloc[:, -2:].to_numpy(dtype=np.float32)
            frames.append(torch.from_numpy(arr))
        if cfg.normalize_dataset:
            stacked = torch.cat(frames, dim=0)
            min_ = stacked.min(dim=0).values
            max_ = stacked.max(dim=0).values
            scale = (max_ - min_).clamp_min(1e-8)
            data = [(frame - min_) / scale for frame in frames]
            min_max = (min_, max_)
        else:
            data = frames
            min_max = None
        test_data = [frame.clone() for frame in data]
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    return data, test_data, min_max


def main():
    parser = argparse.ArgumentParser(description="Recompute trajectories from checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to saved checkpoint .pt")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts_local",
        help="Where to store regenerated trajectories and plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for integration (cpu/cuda/mps)",
    )
    parser.add_argument(
        "--st-data-dir",
        type=str,
        default=None,
        help="Optional override for ST dataset directory",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config_dict = ckpt.get("config", {})

    cfg = TrainConfig()
    for key, value in config_dict.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    if args.st_data_dir is not None:
        cfg.st_data_dir = args.st_data_dir
    cfg.device = args.device
    cfg.use_wandb = False

    device = torch.device(cfg.device)

    data, test_data, min_max = build_dataset(cfg)
    if min_max is None and ckpt.get("min_max") is not None:
        min_max = ckpt["min_max"]

    times = ckpt.get("times")
    if times is None:
        print("times is None")
        times = torch.linspace(0, 1, cfg.eval_num_timepoints)
    times = times.to(device)
    
    print(times.shape)
    print(times)

    X0 = test_data[0].to(device).float()
    if min_max is not None:
        denorm_min, denorm_max = min_max
        denorm_min = denorm_min.to(device)
        denorm_max = denorm_max.to(device)
        X0_denorm = denormalize(X0, (denorm_min, denorm_max))
    else:
        X0_denorm = X0

    dim = X0.shape[-1]

    flow_net = FlowMLPWrapper(dim=dim, width=cfg.cell_flow_width).to(device)
    flow_net.load_state_dict(ckpt["flow"])
    flow_net.eval()

    flow_wrapper = flow_model_torch_wrapper(FlowAdapter(flow_net)).to(device)
    node = NeuralODE(flow_wrapper, solver="dopri5", sensitivity="adjoint").to(device)

    with torch.no_grad():
        traj = node.trajectory(X0_denorm, t_span=times)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pt = out_dir / (Path(args.checkpoint).stem + "_recomputed_traj.pt")
    torch.save(
        {
            "trajectory": traj.cpu(),
            "t_eval": times.cpu(),
            "seed": ckpt.get("config", {}).get("seed_list", [None])[0],
            "config": config_dict,
        },
        out_pt,
    )

    views = [
        (30.0, -60.0),
        (45.0, -45.0),
        (20.0, 45.0),
    ]
    for idx, (elev, azim) in enumerate(views):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        plot_cell_trajectories_3d(ax, traj.cpu(), times.cpu(), elev=elev, azim=azim)
        fig.tight_layout()
        fig.savefig(out_dir / f"{out_pt.stem}_view{idx}.png", dpi=200)
        plt.close(fig)

    print(f"Saved recomputed trajectories to {out_pt}")


if __name__ == "__main__":
    main()
