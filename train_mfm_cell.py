#!/usr/bin/env python
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

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from ali_cfm.data_utils import denormalize, denormalize_gradfield, get_dataset
from ali_cfm.training.training_utils import sample_x_batch
from ali_cfm.loggin_and_metrics import compute_emd
from mfm.flow_matchers.models.mfm import MetricFlowMatcher
from mfm.geo_metrics.metric_factory import DataManifoldMetric
from ali_cfm.nets import TrainableInterpolant, MLP
from mfm.networks.utils import flow_model_torch_wrapper
from torchcfm.conditional_flow_matching import OTPlanSampler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_endpoint_pairs(start: torch.Tensor, end: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = sample_x_batch(start.detach().cpu(), batch_size).to(start.device)
    x1 = sample_x_batch(end.detach().cpu(), batch_size).to(end.device)
    return x0, x1


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
    log_interval: int = 250,
) -> float:
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
    denormalize_inputs: bool,
    min_max,
    wandb_run,
    seed: int,
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

        if denormalize_inputs and min_max is not None:
            xt = denormalize(xt, min_max)
            ut = denormalize_gradfield(ut, min_max)

        vt = flow_net(t, xt)
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
                }
            )

    return total_loss / max(1, steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Metric Flow Matching on cell-tracking data")
    parser.add_argument("--cell-stack-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seeds", type=int, nargs="*", default=[42])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--geopath-epochs", type=int, default=30)
    parser.add_argument("--geopath-steps", type=int, default=2500)
    parser.add_argument("--geopath-lr", type=float, default=1e-4)
    parser.add_argument("--flow-epochs", type=int, default=30)
    parser.add_argument("--flow-steps", type=int, default=2500)
    parser.add_argument("--flow-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--rho", type=float, default=1e-3)
    parser.add_argument("--alpha-metric", type=float, default=1.0)
    parser.add_argument("--cell-subset-size", type=int, default=10)
    parser.add_argument("--cell-subset-seed", type=int, default=None)
    parser.add_argument("--piecewise-training", action="store_true")
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="mfm-cell")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--normalize-dataset", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    full_data, min_max = get_dataset(
        "cell_tracking",
        n_data_dims=2,
        normalize=args.normalize_dataset,
        nicola_path=args.cell_stack_path,
        whiten=args.whiten,
    )
    full_data = [frame.to(device) for frame in full_data]

    time_indices = torch.arange(len(full_data), device=device, dtype=torch.float32)
    times = time_indices / time_indices[-1]

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
        name=args.wandb_name or f"mfm_cell_{int(time.time())}",
        config=vars(args),
    )

    geopath_steps = args.geopath_epochs * args.geopath_steps
    flow_steps = args.flow_epochs * args.flow_steps
    requires_denorm_flow = args.normalize_dataset and min_max is not None

    for seed in args.seeds:
        set_seed(seed)
        if args.cell_subset_size:
            rng = np.random.default_rng(args.cell_subset_seed)
            train_frames = []
            for frame in full_data:
                if frame.shape[0] <= args.cell_subset_size:
                    train_frames.append(frame)
                else:
                    idx = rng.choice(frame.shape[0], args.cell_subset_size, replace=False)
                    train_frames.append(frame[idx])
        else:
            train_frames = full_data

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
                times,
                geopath_steps,
                args.batch_size,
                ot_sampler,
                data_metric,
                metric_samples=metric_samples_all,
                piecewise=args.piecewise_training,
                wandb_run=wandb_run,
                seed=seed,
            )
            wandb_run.log({"train/geopath_loss": geopath_loss, "train/seed": seed})

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
                times,
                flow_steps,
                args.batch_size,
                ot_sampler,
                piecewise=args.piecewise_training,
                denormalize_inputs=requires_denorm_flow,
                min_max=min_max,
                wandb_run=wandb_run,
                seed=seed,
            )
            wandb_run.log({"train/flow_loss": flow_loss, "train/seed": seed})

        flow_wrapper = flow_model_torch_wrapper(FlowAdapter(flow_net)).to(device)
        node = NeuralODE(flow_wrapper, solver="dopri5", sensitivity="adjoint").to(device)

        start_state = full_data[0]
        model_input = denormalize(start_state, min_max) if requires_denorm_flow else start_state
        with torch.no_grad():
            traj = node.trajectory(model_input, t_span=times)

        emd_values: List[float] = []
        saved_payload = {
            "seed": seed,
            "times": times.detach().cpu(),
        }
        for idx_time in range(1, len(full_data)):
            target_state = denormalize(full_data[idx_time], min_max) if args.normalize_dataset else full_data[idx_time]
            pred_state = traj[idx_time]
            pred_eval = pred_state if requires_denorm_flow else (denormalize(pred_state, min_max) if args.normalize_dataset else pred_state)
            emd_val = float(compute_emd(target_state, pred_eval))
            emd_values.append(emd_val)
            if args.normalize_dataset and np.isnan(emd_val):
                print(f"[warn] seed={seed} t={idx_time}: EMD is NaN")

        emd_array = np.array(emd_values, dtype=float)
        print(
            f"[seed={seed}] EMD mean={np.nanmean(emd_array):.6f} std={np.nanstd(emd_array):.6f}"
        )
        if wandb_run:
            wandb_run.log(
                {
                    "eval/emd_mean": float(np.nanmean(emd_array)),
                    "eval/emd_std": float(np.nanstd(emd_array)),
                    "eval/seed": seed,
                }
            )

        saved_payload["trajectory"] = traj.detach().cpu()
        if wandb_run:
            artifact_dir = Path(wandb_run.dir) / "artifacts"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"{(wandb_run.name or 'mfm_cell')}_seed{seed}_traj.pt"
            torch.save(saved_payload, artifact_path)
            wandb_run.save(str(artifact_path), policy="now")

    if wandb_run is not None:
        wandb_run.finish()


class FlowAdapter(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t[:, None]
        return self.base(t, x)


if __name__ == "__main__":
    main()
