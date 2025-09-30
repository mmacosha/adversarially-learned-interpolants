#!/usr/bin/env python
"""Metric Flow Matching training script for ST data (held-out marginal protocol)."""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torchdyn.core import NeuralODE
from tqdm.auto import trange

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
EXTERNAL_DIR = REPO_ROOT / "external" / "metric-flow-matching"
if EXTERNAL_DIR.exists() and str(EXTERNAL_DIR) not in sys.path:
    sys.path.append(str(EXTERNAL_DIR))

import wandb

from ali_cfm.data_utils import denormalize, get_dataset
from ali_cfm.training.training_utils import sample_x_batch
from ali_cfm.loggin_and_metrics import compute_emd
from mfm.flow_matchers.models.mfm import MetricFlowMatcher
from mfm.geo_metrics.metric_factory import DataManifoldMetric
from ali_cfm.nets import TrainableInterpolant, MLP
from torchcfm.utils import torch_wrapper
from torchcfm.conditional_flow_matching import OTPlanSampler


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_endpoint_pairs(start: torch.Tensor, end: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = sample_x_batch(start.detach().cpu(), batch_size).to(start.device)
    x1 = sample_x_batch(end.detach().cpu(), batch_size).to(end.device)
    return x0, x1


# -----------------------------------------------------------------------------
# Training loops
# -----------------------------------------------------------------------------

def train_geopath(
    flow_matcher: MetricFlowMatcher,
    optimizer: torch.optim.Optimizer,
    frames: Sequence[torch.Tensor],
    times: torch.Tensor,
    steps: int,
    batch_size: int,
    ot_sampler: Optional[OTPlanSampler],
    data_metric: DataManifoldMetric,
    *,
    metric_samples: torch.Tensor,
    piecewise: bool,
    wandb_run,
    seed: int,
    removed_t: Optional[int],
    log_interval: int = 250,
) -> float:
    if piecewise:
        print('Running in piecewise mode')
    
    geopath_net: nn.Module = flow_matcher.geopath_net
    geopath_net.train()

    num_segments = len(frames) - 1
    metric_segments: Optional[List[torch.Tensor]] = None
    if piecewise:
        metric_segments = [
            torch.cat([frames[i], frames[i + 1]], dim=0).detach()
            for i in range(num_segments)
        ]
    else:
        metric_samples = metric_samples.detach()

    total_loss = 0.0
    progress = trange(steps, desc="GeoPath", leave=False)
    for step_idx in progress:
        optimizer.zero_grad()
        if piecewise:
            seg_idx = random.randrange(num_segments)
            start_frame = frames[seg_idx]
            end_frame = frames[seg_idx + 1]
            t_min = times[seg_idx]
            t_max = times[seg_idx + 1]
            seg_metric = metric_segments[seg_idx]
        else:
            start_frame = frames[0]
            end_frame = frames[-1]
            t_min = times[0]
            t_max = times[-1]
            seg_metric = metric_samples

        x0, x1 = sample_endpoint_pairs(start_frame, end_frame, batch_size)
        if ot_sampler is not None:
            x0, x1 = ot_sampler.sample_plan(x0, x1, replace=True)

        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(
            x0,
            x1,
            t_min,
            t_max,
            training_geopath_net=True,
        )

        velocity = data_metric.calculate_velocity(xt, ut, seg_metric, timestep=0)
        loss = torch.mean(velocity ** 2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (step_idx + 1) % log_interval == 0 or step_idx + 1 == steps:
            avg_loss = total_loss / (step_idx + 1)
            progress.set_postfix({"loss": avg_loss})
            wandb_run.log(
                {
                    "train/geopath_loss_step": avg_loss,
                    "train/geopath_step": step_idx + 1,
                    "train/seed": seed,
                    "train/removed_t": removed_t,
                }
            )

    return total_loss / max(1, steps)


def train_flow(
    flow_matcher: MetricFlowMatcher,
    flow_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    frames: Sequence[torch.Tensor],
    times: torch.Tensor,
    steps: int,
    batch_size: int,
    ot_sampler: Optional[OTPlanSampler],
    *,
    piecewise: bool,
    wandb_run,
    seed: int,
    removed_t: Optional[int],
    log_interval: int = 250,
) -> float:
    flow_net.train()
    total_loss = 0.0
    num_segments = len(frames) - 1

    progress = trange(steps, desc="Flow", leave=False)
    for step_idx in progress:
        optimizer.zero_grad()
        if piecewise:
            seg_idx = random.randrange(num_segments)
            start_frame = frames[seg_idx]
            end_frame = frames[seg_idx + 1]
            t_min = times[seg_idx]
            t_max = times[seg_idx + 1]
        else:
            start_frame = frames[0]
            end_frame = frames[-1]
            t_min = times[0]
            t_max = times[-1]

        x0, x1 = sample_endpoint_pairs(start_frame, end_frame, batch_size)
        if ot_sampler is not None:
            x0, x1 = ot_sampler.sample_plan(x0, x1, replace=True)

        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1, t_min, t_max)
        if t.dim() == 1:
            t = t[:, None]

        vt = flow_net(torch.cat([xt, t], dim=-1))
        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (step_idx + 1) % log_interval == 0 or step_idx + 1 == steps:
            avg_loss = total_loss / (step_idx + 1)
            progress.set_postfix({"loss": avg_loss})
            wandb_run.log(
                {
                    "train/flow_loss_step": avg_loss,
                    "train/flow_step": step_idx + 1,
                    "train/seed": seed,
                    "train/removed_t": removed_t,
                }
            )

    return total_loss / max(1, steps)


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metric Flow Matching on ST data (leave-one-out)")
    parser.add_argument("--st-data-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--geopath-epochs", type=int, default=20)
    parser.add_argument("--geopath-steps", type=int, default=2500)
    parser.add_argument("--geopath-lr", type=float, default=1e-4)
    parser.add_argument("--flow-epochs", type=int, default=20)
    parser.add_argument("--flow-steps", type=int, default=2500)
    parser.add_argument("--flow-lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--rho", type=float, default=5e-4)
    parser.add_argument("--alpha-metric", type=float, default=1.0)
    parser.add_argument("--piecewise-training", action="store_true")
    parser.add_argument("--wandb-name", type=str, default="nicola_mfm_cst_new")
    parser.add_argument("--wandb-project", type=str, default="mixture-fmls")
    parser.add_argument("--wandb-entity", type=str, default="mixtures-all-the-way")
    parser.add_argument("--normalize-dataset", action="store_true", default=True)
    parser.add_argument("--eval-segment-points", type=int, default=101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    data_frames, min_max = get_dataset(
        "ST",
        n_data_dims=2,
        normalize=args.normalize_dataset,
        nicola_path=args.st_data_dir,
    )
    data_frames = [frame.to(device) for frame in data_frames]
    timesteps_list = list(range(len(data_frames)))
    max_time_index = timesteps_list[-1]
    normalized_times = torch.tensor(timesteps_list, dtype=torch.float32, device=device) / max_time_index

    metric_args = SimpleNamespace(
        gamma_current=args.gamma,
        rho=args.rho,
        velocity_metric="land",
        n_centers=100,
        kappa=0.5,
        metric_epochs=0,
        metric_patience=10,
        metric_lr=1e-3,
        alpha_metric=args.alpha_metric,
        data_type="trajectory",
        accelerator="cpu",
    )
    data_metric = DataManifoldMetric(metric_args, skipped_time_points=None, datamodule=None)
    ot_sampler = OTPlanSampler(method="exact")

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or f"mfm_st_{int(time.time())}",
        config=vars(args),
    )

    geopath_steps = args.geopath_epochs * args.geopath_steps
    flow_steps = args.flow_epochs * args.flow_steps

    for seed in args.seeds:
        set_seed(seed)
        emd_results = []
        saved_segments: List[dict] = []

        for removed_t in timesteps_list[1:-1]:
            current_indices = [t for t in timesteps_list if t != removed_t]
            train_frames = [data_frames[idx] for idx in current_indices]
            train_times = torch.tensor(current_indices, dtype=torch.float32, device=device) / max_time_index

            geopath_net = TrainableInterpolant(
                dim=train_frames[0].shape[1],
                h_dim=128,
                t_smooth=0.0,
                time_varying=True,
            ).to(device)
            setattr(geopath_net, "time_geopath", True)
            flow_net = MLP(dim=train_frames[0].shape[1], time_varying=True, w=128).to(device)
            flow_matcher = MetricFlowMatcher(geopath_net=geopath_net, sigma=0.0, alpha=1)

            geopath_optimizer = Adam(geopath_net.parameters(), lr=args.geopath_lr)
            metric_samples_all = torch.cat(train_frames, dim=0)

            if geopath_steps > 0:
                geopath_loss = train_geopath(
                    flow_matcher,
                    geopath_optimizer,
                    train_frames,
                    train_times,
                    geopath_steps,
                    args.batch_size,
                    ot_sampler,
                    data_metric,
                    metric_samples=metric_samples_all,
                    piecewise=args.piecewise_training,
                    wandb_run=wandb_run,
                    seed=seed,
                    removed_t=removed_t,
                )
                wandb_run.log(
                    {
                        "train/geopath_loss": geopath_loss,
                        "train/seed": seed,
                        "train/removed_t": removed_t,
                    }
                )

            geopath_net.eval()
            for param in geopath_net.parameters():
                param.requires_grad_(False)

            flow_optimizer = Adam(flow_net.parameters(), lr=args.flow_lr)
            if flow_steps > 0:
                flow_loss = train_flow(
                    flow_matcher,
                    flow_net,
                    flow_optimizer,
                    train_frames,
                    train_times,
                    flow_steps,
                    args.batch_size,
                    ot_sampler,
                    piecewise=args.piecewise_training,
                    wandb_run=wandb_run,
                    seed=seed,
                    removed_t=removed_t,
                )
                wandb_run.log(
                    {
                        "train/flow_loss": flow_loss,
                        "train/seed": seed,
                        "train/removed_t": removed_t,
                    }
                )

            flow_wrapper = torch_wrapper(flow_net).to(device)
            node = NeuralODE(flow_wrapper, solver="dopri5", sensitivity="adjoint").to(device)

            start_state = data_frames[removed_t - 1]
            target_state = data_frames[removed_t]
            
            span = torch.linspace(
                float(normalized_times[removed_t - 1].item()),
                float(normalized_times[removed_t].item()),
                args.eval_segment_points,
                device=device,
            )
            
            with torch.no_grad():
                seg_traj = node.trajectory(start_state, t_span=span)
            pred_state = seg_traj[-1]

            target_eval = denormalize(target_state, min_max) if args.normalize_dataset else target_state
            pred_eval = denormalize(pred_state, min_max) if args.normalize_dataset else pred_state
            emd_val = float(compute_emd(target_eval, pred_eval))
            emd_results.append(emd_val)

            saved_segments.append(
                {
                    "removed_t": removed_t,
                    "seed": seed,
                    "span": span.detach().cpu(),
                    "start_state": start_state.detach().cpu(),
                    "pred_state": pred_state.detach().cpu(),
                    "target_state": target_state.detach().cpu(),
                    "emd": emd_val,
                }
            )

            wandb_run.log(
                {
                    "eval/emd": emd_val,
                    "eval/removed_t": removed_t,
                    "eval/seed": seed,
                }
            )

        emd_array = np.array(emd_results, dtype=float)
        print(
            f"[seed={seed}] mean EMD={np.nanmean(emd_array):.6f} std={np.nanstd(emd_array):.6f}"
        )
        wandb_run.log(
            {
                "eval/emd_mean": float(np.nanmean(emd_array)),
                "eval/emd_std": float(np.nanstd(emd_array)),
                "eval/seed": seed,
            }
        )

        if saved_segments:
            artifact_dir = Path(wandb_run.dir) / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"{(wandb_run.name or 'mfm_st')}_seed{seed}_segments.pt"
            torch.save(saved_segments, artifact_path)
            wandb_run.save(str(artifact_path), policy="now")

    wandb_run.finish()


# -----------------------------------------------------------------------------
# Simple MLP with time concatenation
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
