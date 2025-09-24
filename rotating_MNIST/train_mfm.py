import argparse
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from torch import nn
from torch.optim import Adam
from torchdyn.core import NeuralODE
from tqdm.auto import trange

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


# Ensure the external Metric Flow Matching repo is importable when running the script directly.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

EXTERNAL_DIR = os.path.join(REPO_ROOT, "external", "metric-flow-matching")
if EXTERNAL_DIR not in sys.path:
    sys.path.append(EXTERNAL_DIR)

NUMBA_CACHE_DIR = os.path.join(REPO_ROOT, ".numba_cache")
os.makedirs(NUMBA_CACHE_DIR, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", NUMBA_CACHE_DIR)


from ali_cfm.data_utils import denormalize, denormalize_gradfield, get_dataset
from ali_cfm.nets import CorrectionUNet, MLP, TrainableInterpolant, TrainableInterpolantMNIST
from mfm.flow_matchers.models.mfm import MetricFlowMatcher
from mfm.geo_metrics.metric_factory import DataManifoldMetric
from mfm.networks.geopath_networks.mlp import GeoPathMLP
from mfm.networks.utils import flow_model_torch_wrapper

from torchcfm.conditional_flow_matching import OTPlanSampler


def default_device() -> str:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")
            return "mps"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_emd(p1: torch.Tensor, p2: torch.Tensor, device: Union[torch.device, str] = "cpu") -> float:
    a_weights = torch.ones((p1.shape[0],), device=device) / p1.shape[0]
    b_weights = torch.ones((p2.shape[0],), device=device) / p2.shape[0]

    M = pot.dist(p1, p2).sqrt()
    return float(pot.emd2(a_weights, b_weights, M, numItermax=1e7))


def fix_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    seed_list: Sequence[int] = field(default_factory=lambda: [42])
    device: str = field(default_factory=default_device)
    n_data_dims: int = 256
    normalize_dataset: bool = True

    dataset: str = "cell_tracking"  # or "cell_tracking"

    batch_size: int = 128
    metric_batch_size: int = 512
    geopath_epochs: int = 5
    geopath_steps_per_epoch: int = 200
    geopath_lr: float = 1e-4

    flow_epochs: int = 10
    flow_steps_per_epoch: int = 500
    flow_lr: float = 1e-4

    gamma: float = 0.4
    rho: float = 1e-3
    alpha_metric: float = 1.0

    alpha: float = 1.0
    sigma: float = 0.0
    ot_method: Optional[str] = "exact"

    geopath_hidden_dims: Sequence[int] = field(default_factory=lambda: [512, 512, 512])
    activation_geopath: str = "swish"
    activation_flow: str = "swish"

    eval_num_timepoints: int = 101
    verbose: bool = False
    save_plot: Optional[str] = None
    skip_eval: bool = False

    cell_stack_path: Optional[str] = None
    cell_test_stack_path: Optional[str] = None
    cell_subset_size: Optional[int] = 10
    cell_subset_seed: Optional[int] = None

    velocity_metric: str = "land"
    metric_n_centers: int = 100
    metric_kappa: float = 0.5
    metric_epochs: int = 0
    metric_patience: int = 10
    metric_lr: float = 1e-3 #not used in land

    mnist_unet_base: int = 32
    mnist_interpolant_hidden: int = 512
    cell_flow_width: int = 128
    cell_interpolant_hidden: int = 512
    whiten: bool = False

    use_wandb: bool = True
    wandb_project: str = "mixture-fmls"
    wandb_entity: str = "mixtures-all-the-way"
    wandb_name: Optional[str] = "nicola_mfm_celltrack"

    save_checkpoints: bool = True
    checkpoint_dir: Optional[str] = None
    st_data_dir: Optional[str] = None

def _sample_endpoint_pairs(
    start: torch.Tensor,
    end: torch.Tensor,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample batches from the first and last timesteps for endpoint-only training."""
    if start.device != end.device:
        raise ValueError("Start and end tensors must share the same device")
    if start.shape[0] == 0 or end.shape[0] == 0:
        raise ValueError("Endpoint tensors must contain at least one sample")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    replace_start = batch_size > start.shape[0]
    replace_end = batch_size > end.shape[0]

    if replace_start:
        idx_start = torch.randint(0, start.shape[0], (batch_size,), device=start.device)
    else:
        idx_start = torch.randperm(start.shape[0], device=start.device)[:batch_size]

    if replace_end:
        idx_end = torch.randint(0, end.shape[0], (batch_size,), device=end.device)
    else:
        idx_end = torch.randperm(end.shape[0], device=end.device)[:batch_size]

    return start.index_select(0, idx_start), end.index_select(0, idx_end)


def train_geopath(
    flow_matcher: MetricFlowMatcher,
    optimizer: torch.optim.Optimizer,
    data: Sequence[torch.Tensor],
    times: torch.Tensor,
    gamma: float,
    rho: float,
    steps: int,
    batch_size: int,
    ot_sampler: Optional[OTPlanSampler],
    data_metric: DataManifoldMetric,
    *,
    metric_samples: torch.Tensor,
    log_shapes: bool = False,
    wandb_run: Optional[Any] = None,
    log_interval: int = 500,
    seed: Optional[int] = None,
) -> float:
    geopath_net: nn.Module = flow_matcher.geopath_net
    geopath_net.train()

    if len(data) < 2:
        raise ValueError("GeoPath training requires at least two timesteps of data")

    total_loss = 0.0
    start_data = data[0]
    end_data = data[-1]
    t_min = times[0]
    t_max = times[-1]
    metric_samples = metric_samples.detach()

    for step_idx in trange(steps, desc="Training GeoPath", leave=False):
        optimizer.zero_grad()

        x0, x1 = _sample_endpoint_pairs(start_data, end_data, batch_size)

        if ot_sampler is not None:
            x0, x1 = ot_sampler.sample_plan(x0, x1, replace=True)

        x0 = x0.to(start_data.device)
        x1 = x1.to(end_data.device)

        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(
            x0,
            x1,
            t_min,
            t_max,
            training_geopath_net=True,
        )

        if log_shapes and step_idx == 0:
            print(
                "[GeoPath] pair 0 ->",
                len(data) - 1,
                "| x0",
                tuple(x0.shape),
                "x1",
                tuple(x1.shape),
                "xt",
                tuple(xt.shape),
                "ut",
                tuple(ut.shape),
                "metric",
                tuple(metric_samples.shape),
            )

        velocity = data_metric.calculate_velocity(
            xt,
            ut,
            metric_samples,
            timestep=0, # doesn't matter for land metric 
        )

        loss = torch.mean(velocity ** 2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if wandb_run and (step_idx == 0 or (step_idx + 1) % max(1, log_interval) == 0):
            log_payload = {
                "train/geopath_loss_step": float(loss.item()),
                "train/geopath_step": step_idx + 1,
            }
            if seed is not None:
                log_payload["train/seed"] = seed
            wandb_run.log(log_payload)

    return total_loss / max(1, steps)


def train_flow(
    flow_matcher: MetricFlowMatcher,
    flow_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    data: Sequence[torch.Tensor],
    times: torch.Tensor,
    steps: int,
    batch_size: int,
    ot_sampler: Optional[OTPlanSampler],
    *,
    log_shapes: bool = False,
    wandb_run: Optional[Any] = None,
    log_interval: int = 500,
    seed: Optional[int] = None,
    denormalize_inputs: bool = False,
    min_max: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> float:
    flow_net.train()

    total_loss = 0.0
    if len(data) < 2:
        raise ValueError("Flow training requires at least two timesteps of data")

    for step_idx in trange(steps, desc="Training Flow", leave=False):
        optimizer.zero_grad()

        x0, x1 = _sample_endpoint_pairs(data[0], data[-1], batch_size)

        if ot_sampler is not None:
            x0, x1 = ot_sampler.sample_plan(x0, x1, replace=True)

        x0 = x0.to(data[0].device)
        x1 = x1.to(data[-1].device)

        t_min = times[0]
        t_max = times[-1]
        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(
            x0,
            x1,
            t_min,
            t_max,
        )

        if log_shapes and step_idx == 0:
            print(
                "[Flow ] pair 0 ->",
                len(data) - 1,
                "| x0",
                tuple(x0.shape),
                "x1",
                tuple(x1.shape),
                "xt",
                tuple(xt.shape),
                "ut",
                tuple(ut.shape),
            )

        t_batch = t
        if t_batch.dim() == 1:
            t_batch = t_batch[:, None]
        xt_batch = xt
        ut_batch = ut

        if denormalize_inputs and min_max is not None:
            xt_batch = denormalize(xt_batch, min_max)
            ut_batch = denormalize_gradfield(ut_batch, min_max)

        try:
            vt = flow_net(t_batch, xt_batch)
        except TypeError:
            tb = t_batch
            if tb.dim() == 1:
                tb = tb[:, None]
            vt = flow_net(torch.cat([xt_batch, tb], dim=-1))
        loss = torch.mean((vt - ut_batch) ** 2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if wandb_run and (step_idx == 0 or (step_idx + 1) % max(1, log_interval) == 0):
            log_payload = {
                "train/flow_loss_step": float(loss.item()),
                "train/flow_step": step_idx + 1,
            }
            if seed is not None:
                log_payload["train/seed"] = seed
            wandb_run.log(log_payload)

    return total_loss / max(1, steps)


class FlowMLPWrapper(nn.Module):
    def __init__(self, dim: int, width: int):
        super().__init__()
        self.model = MLP(dim=dim, w=width, time_varying=True)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return self.model(torch.cat([x, t], dim=-1))


class FlowAdapter(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        tb = t
        if tb.dim() == 0:
            tb = tb.unsqueeze(0)
        if tb.dim() == 1:
            tb = tb.unsqueeze(-1)
        if tb.shape[0] != x.shape[0]:
            tb = tb.expand(x.shape[0], -1)

        argcount = self.base.forward.__code__.co_argcount
        if argcount >= 3:
            return self.base(tb, x)
        return self.base(torch.cat([x, tb], dim=-1))



def _scatter_cell_points(
    ax,
    samples: torch.Tensor,
    *,
    color: str,
    marker_size: float,
    alpha: float,
    label: Optional[str] = None,
) -> None:
    pts = samples.detach().cpu().numpy()
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=marker_size,
        alpha=alpha,
        c=color,
        edgecolors="none",
        label=label,
    )


def plot_cell_samples(
    ax,
    samples: torch.Tensor,
    title: str,
    *,
    color: str = "#1f77b4",
    label: Optional[str] = None,
    marker_size: float = 10.0,
    alpha: float = 0.85,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    _scatter_cell_points(
        ax,
        samples,
        color=color,
        marker_size=marker_size,
        alpha=alpha,
        label=label,
    )
    if bounds is not None:
        (min_xy, max_xy) = bounds
        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])
        ax.invert_yaxis()
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if label is not None:
        ax.legend(loc="upper right", frameon=False, handlelength=1.0, fontsize="small")


def plot_cell_overlay(
    ax,
    prediction: torch.Tensor,
    target: torch.Tensor,
    title: str,
    *,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    show_legend: bool = False,
    marker_size: float = 10.0,
    alpha_target: float = 0.4,
    alpha_pred: float = 0.6,
    color: Optional[str] = None,
) -> None:
    _scatter_cell_points(
        ax,
        target,
        color="#d62728" if color is None else color,
        marker_size=marker_size,
        alpha=alpha_target,
        label="target" if show_legend else None,
    )
    _scatter_cell_points(
        ax,
        prediction,
        color="#1f77b4" if color is None else color,
        marker_size=marker_size,
        alpha=alpha_pred,
        label="prediction" if show_legend else None,
    )
    if bounds is not None:
        (min_xy, max_xy) = bounds
        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])
        ax.invert_yaxis()
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if show_legend:
        ax.legend(loc="upper right", frameon=False, handlelength=1.0, fontsize="small")


def plot_cell_trajectories(
    ax,
    trajectories: torch.Tensor,
    times: torch.Tensor,
    *,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    cmap_name: str = "viridis",
    marker_size: float = 8.0,
    line_width: float = 1.2,
) -> None:
    traj = trajectories.detach().cpu().numpy()  # (T, N, 2)
    num_steps, num_points, _ = traj.shape
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.0, 1.0, num_points))

    for idx in range(num_points):
        path = traj[:, idx]
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=colors[idx],
            linewidth=line_width,
            alpha=0.9,
        )
        ax.scatter(
            path[:, 0],
            path[:, 1],
            s=marker_size,
            color=colors[idx],
            alpha=0.9,
        )

    if bounds is not None:
        (min_xy, max_xy) = bounds
        ax.set_xlim(min_xy[0], max_xy[0])
        ax.set_ylim(min_xy[1], max_xy[1])
    ax.invert_yaxis()
    ax.set_title("Flow trajectories")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_cell_trajectories_3d(
    ax,
    trajectories: torch.Tensor,
    times: torch.Tensor,
    *,
    cmap_name: str = "viridis",
    elev: float = 30.0,
    azim: float = -60.0,
    title: str = "Flow trajectories (3D)",
) -> None:
    traj = trajectories.detach().cpu().numpy()  # (T, N, 2)
    times_np = times.detach().cpu().numpy()      # (T,)
    num_steps, num_points, _ = traj.shape

    xs = traj[:, :, 0].reshape(-1)
    ys = traj[:, :, 1].reshape(-1)
    ts = np.repeat(times_np, num_points)
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=ts.min(), vmax=ts.max())
    colors = cmap(norm(ts))

    ax.scatter(xs, ys, ts, c=colors, s=14.0, alpha=0.9, linewidths=0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, pad=0.10, fraction=0.04, label="t")


def _save_model_checkpoint(
    directory: Path,
    prefix: str,
    seed: int,
    geopath_net: nn.Module,
    flow_net: nn.Module,
    cfg: TrainConfig,
    times: torch.Tensor,
    min_max: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    ckpt_path = directory / f"{prefix}_seed{seed}.pt"
    payload = {
        "geopath": geopath_net.state_dict(),
        "flow": flow_net.state_dict(),
        "config": asdict(cfg),
        "times": times.detach().cpu(),
    }
    if min_max is not None:
        payload["min_max"] = (min_max[0].detach().cpu(), min_max[1].detach().cpu())
    torch.save(payload, ckpt_path)
    return ckpt_path


def main(cfg: TrainConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    warnings.filterwarnings("ignore")

    try:
        device = torch.device(cfg.device)
        torch.tensor(0, device=device)
    except Exception as exc:
        print(f"[warn] failed to use device '{cfg.device}': {exc}. Falling back to CPU.")
        device = torch.device("cpu")

    if cfg.verbose:
        print(
            f"Config: batch_size={cfg.batch_size}, metric_batch_size={cfg.metric_batch_size}, "
            f"geopath_steps={cfg.geopath_steps_per_epoch}, geopath_epochs={cfg.geopath_epochs}, "
            f"flow_steps={cfg.flow_steps_per_epoch}, flow_epochs={cfg.flow_epochs}, "
            f"geopath_lr={cfg.geopath_lr}, flow_lr={cfg.flow_lr}, seeds={list(cfg.seed_list)}"
        )

    wandb_run = None
    try:
        if cfg.use_wandb:
            if wandb is None:
                raise ImportError(
                    "wandb is not installed but use_wandb=True. Install wandb or run with --no-wandb"
                )
            default_name = f"mfm_{cfg.dataset}_{int(time.time())}"
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_name or default_name,
                config=asdict(cfg),
            )

        dataset_name = cfg.dataset.lower()
        if dataset_name == "rotating_mnist":
            data, min_max = get_dataset(
                "RotatingMNIST_train", cfg.n_data_dims, normalize=cfg.normalize_dataset
            )
            test_data, _ = get_dataset(
                "RotatingMNIST_test", cfg.n_data_dims, normalize=cfg.normalize_dataset
            )
        elif dataset_name == "cell_tracking":
            dataset_key = "cell_tracking"
            dataset_path = getattr(cfg, "cell_stack_path", None)
            extra_kwargs = {"whiten": getattr(cfg, "whiten", False)}
            print(dataset_path)
            try:
                data, min_max = get_dataset(
                    dataset_key,
                    cfg.n_data_dims,
                    normalize=cfg.normalize_dataset,
                    **extra_kwargs,
                    nicola_path=dataset_path,
                )
            except FileNotFoundError:
                raise
            import copy
            test_data = copy.deepcopy(data)
        elif dataset_name == "st":
            dataset_path = getattr(cfg, "st_data_dir", None)
            dataset_key = "ST"
            data, min_max = get_dataset(
                dataset_key,
                cfg.n_data_dims,
                normalize=cfg.normalize_dataset,
                nicola_path=dataset_path,
            )
            import copy
            test_data = copy.deepcopy(data)
        else:
            raise ValueError(f"Unknown dataset '{cfg.dataset}'")

        print("Using dataset: ", cfg.dataset)

        times = torch.linspace(0.0, 1.0, len(data), device=device)

        dim = data[0].shape[1]

        original_min_batch = min(x.shape[0] for x in data)

        print(len(data))

        print(data[0].shape)
        print(data[1].shape)

        print(
            f"Loaded {cfg.dataset} with {len(data)} timesteps; "
            f"min batch {original_min_batch}; feature dim {dim}"
        )
        if cfg.verbose:
            for idx, frame in enumerate(data):
                print(f"  train t={idx}: {tuple(frame.shape)}")

        if cfg.batch_size > original_min_batch:
            print(
                f"[info] Reducing batch size from {cfg.batch_size} to {original_min_batch} to match available samples"
            )
            cfg.batch_size = original_min_batch

        if wandb_run:
            wandb_run.log({
                "data/num_timesteps": len(data),
                "data/min_batch": original_min_batch,
                "data/feature_dim": dim,
            })

        ot_sampler = None
        if cfg.ot_method is not None and cfg.ot_method.lower() != "none":
            ot_sampler = OTPlanSampler(method=cfg.ot_method)

        metric_args = SimpleNamespace(
            gamma_current=cfg.gamma,
            rho=cfg.rho,
            velocity_metric=cfg.velocity_metric,
            n_centers=cfg.metric_n_centers,
            kappa=cfg.metric_kappa,
            metric_epochs=cfg.metric_epochs,
            metric_patience=cfg.metric_patience,
            metric_lr=cfg.metric_lr,
            alpha_metric=cfg.alpha_metric,
            data_type="trajectory",
            accelerator="cpu",
        )

        data_metric = DataManifoldMetric(
            args=metric_args, skipped_time_points=None, datamodule=None
        )

        for seed in cfg.seed_list:
            fix_seed(seed)

            train_frames = data

            if dataset_name == "cell_tracking" and cfg.cell_subset_size not in (None, 0):
                subset_size = cfg.cell_subset_size
                subset_seed = cfg.cell_subset_seed
                if subset_size is None or subset_size <= 0:
                    subset_size = original_min_batch
                frame_min = min(frame.shape[0] for frame in data)
                effective_subset = min(subset_size, frame_min)

                rng = None
                if subset_seed is not None:
                    rng = np.random.RandomState(subset_seed)

                subset_frames = []
                for frame in data:
                    if effective_subset >= frame.shape[0]:
                        subset_frames.append(frame)
                        continue

                    if rng is None:
                        idx_np = np.random.choice(frame.shape[0], effective_subset, replace=False)
                    else:
                        idx_np = rng.choice(frame.shape[0], effective_subset, replace=False)
                    idx = torch.from_numpy(idx_np).to(frame.device)
                    subset_frames.append(frame.index_select(0, idx))

                train_frames = subset_frames

                if cfg.verbose:
                    print(
                        f"[debug] Using cell subset of size {effective_subset} per timestep for seed {seed}"
                    )
                if wandb_run:
                    wandb_run.log(
                        {
                            "data/cell_subset_size": effective_subset,
                            "train/seed": seed,
                        }
                    )

            train_frames_device = [frame.to(device) for frame in train_frames]
            metric_samples_device = torch.cat(train_frames_device, dim=0)

            train_min_batch = min(frame.shape[0] for frame in train_frames)
            current_batch_size = min(cfg.batch_size, train_min_batch)

            if dataset_name == "rotating_mnist":
                geopath_net = TrainableInterpolantMNIST(
                    dim=dim,
                    h_dim=cfg.mnist_interpolant_hidden,
                    t_smooth=0.0,
                    time_varying=True,
                    # regulariser="linear",
                ).to(device)
                setattr(geopath_net, "time_geopath", True)
                flow_base = cfg.mnist_unet_base
                flow_net = CorrectionUNet(
                    in_ch=2, base=flow_base, interpolant=False
                ).to(device)
            else:
                geopath_net = TrainableInterpolant(
                    dim=dim,
                    h_dim=cfg.cell_interpolant_hidden,
                    t_smooth=0.0,
                    time_varying=True,
                    # regulariser="linear",
                ).to(device)
                setattr(geopath_net, "time_geopath", True)

                flow_net = FlowMLPWrapper(
                    dim=dim,
                    width=cfg.cell_flow_width,
                ).to(device)

            if cfg.verbose:
                print(
                    f"[debug] geopath params={sum(p.numel() for p in geopath_net.parameters())}, "
                    f"flow params={sum(p.numel() for p in flow_net.parameters())}"
                )

            if cfg.verbose:
                num_geo_params = sum(p.numel() for p in geopath_net.parameters())
                num_flow_params = sum(p.numel() for p in flow_net.parameters())
                print(f"GeoPath parameters: {num_geo_params}")
                print(f"Flow parameters: {num_flow_params}")

            flow_matcher = MetricFlowMatcher(
                geopath_net=geopath_net,
                alpha=float(cfg.alpha),
                sigma=float(cfg.sigma),
            )

            if cfg.verbose:
                num_geo_params = sum(p.numel() for p in geopath_net.parameters())
                print(f"[debug] number of GeoPath parameters = {num_geo_params}")

            geopath_optimizer = Adam(
                geopath_net.parameters(),
                lr=cfg.geopath_lr,
            )

            total_geopath_steps = cfg.geopath_epochs * cfg.geopath_steps_per_epoch
            geopath_loss = None
            if total_geopath_steps > 0 and len(list(geopath_net.parameters())) > 0:
                if cfg.verbose:
                    print(
                        f"[debug] Starting GeoPath training for {total_geopath_steps} steps (batch={current_batch_size})"
                    )
                geopath_loss = train_geopath(
                    flow_matcher,
                    geopath_optimizer,
                    train_frames_device,
                    times,
                    cfg.gamma,
                    cfg.rho,
                    total_geopath_steps,
                    current_batch_size,
                    ot_sampler,
                    data_metric,
                    metric_samples=metric_samples_device,
                    log_shapes=cfg.verbose,
                    wandb_run=wandb_run,
                    seed=seed,
                )
                print(f"[seed={seed}] GeoPath loss: {geopath_loss:.6f}")
                if wandb_run and geopath_loss is not None:
                    wandb_run.log({
                        "train/geopath_loss": geopath_loss,
                        "train/seed": seed,
                    })

            for param in geopath_net.parameters():
                param.requires_grad_(False)
            geopath_net.eval()

            flow_optimizer = Adam(flow_net.parameters(), lr=cfg.flow_lr)

            total_flow_steps = cfg.flow_epochs * cfg.flow_steps_per_epoch
            flow_loss = None
            if total_flow_steps > 0:
                if cfg.verbose:
                    print(
                        f"[debug] Starting flow training for {total_flow_steps} steps (batch={current_batch_size})"
                    )
                denorm_flow = bool(cfg.normalize_dataset and min_max is not None)

                flow_loss = train_flow(
                    flow_matcher,
                    flow_net,
                    flow_optimizer,
                    train_frames_device,
                    times,
                    total_flow_steps,
                    current_batch_size,
                    ot_sampler,
                    log_shapes=cfg.verbose,
                    wandb_run=wandb_run,
                    seed=seed,
                    denormalize_inputs=denorm_flow,
                    min_max=min_max,
                )
                print(f"[seed={seed}] Flow loss: {flow_loss:.6f}")
                if wandb_run and flow_loss is not None:
                    wandb_run.log({
                        "train/flow_loss": flow_loss,
                        "train/seed": seed,
                    })

            if cfg.skip_eval:
                continue

            flow_wrapper = flow_model_torch_wrapper(FlowAdapter(flow_net)).to(device)
            node = NeuralODE(
                flow_wrapper, solver="dopri5", sensitivity="adjoint"
            ).to(device)

            # respect dataset time grid when available
            if dataset_name in {"cell_tracking", "st"}:
                t_eval = times
            else:
                t_eval = torch.linspace(0, 1, cfg.eval_num_timepoints, device=device)
                
            X0 = test_data[0].to(device)

            with torch.no_grad():
                if cfg.normalize_dataset:
                    traj = node.trajectory(denormalize(X0, min_max), t_span=t_eval)
                else:
                    traj = node.trajectory(X0, t_span=t_eval)

            emd_values: List[float] = []
            if dataset_name == "cell_tracking":
                # 3D plot
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

                trajectory_figures = []
                views = [
                    (30.0, -60.0),
                    (45.0, -45.0),
                    (20.0, 45.0),
                ]
                for idx, (elev, azim) in enumerate(views):
                    fig3d = plt.figure(figsize=(6, 5))
                    ax3d = fig3d.add_subplot(111, projection="3d")
                    plot_cell_trajectories_3d(
                        ax3d,
                        traj,
                        t_eval,
                        elev=elev,
                        azim=azim,
                    )
                    fig3d.tight_layout()
                    trajectory_figures.append((fig3d, f"view{idx}"))

                if wandb_run:
                    wandb_run.log(
                        {
                            "eval/trajectories_3d": [
                                wandb.Image(fig3d, caption=view_name)
                                for fig3d, view_name in trajectory_figures
                            ]
                        }
                    )
                if cfg.save_plot:
                    traj_path = Path(cfg.save_plot)
                    traj_path.parent.mkdir(parents=True, exist_ok=True)
                    for fig3d, view_name in trajectory_figures:
                        fig3d.savefig(
                            traj_path.with_name(
                                f"{traj_path.stem}_trajectories3d_{view_name}{traj_path.suffix}"
                            ),
                            dpi=200,
                        )
                for fig3d, _ in trajectory_figures:
                    plt.close(fig3d)

                interpolant_figures_norm: List[Tuple[plt.Figure, str]] = []
                interpolant_figures_denorm: List[Tuple[plt.Figure, str]] = []
                endpoint_figures_norm: List[plt.Figure] = []
                endpoint_figures_denorm: List[plt.Figure] = []
                if flow_matcher.geopath_net is not None and cfg.alpha != 0:
                    X1 = test_data[-1].to(device)
                    expected_pairs = 10
                    if dataset_name == "cell_tracking":
                        if X0.shape[0] != expected_pairs or X1.shape[0] != expected_pairs:
                            raise AssertionError(
                                f"cell_tracking interpolant viz expects exactly {expected_pairs} samples, "
                                f"found {X0.shape[0]} and {X1.shape[0]}"
                            )
                    x0_samples = X0
                    x1_samples = X1
                    if ot_sampler is not None:
                        x0_samples, x1_samples = ot_sampler.sample_plan(
                            x0_samples,
                            x1_samples,
                            replace=True,
                        )

                    t_min = times[0]
                    t_max = times[-1]
                    interp_points: List[torch.Tensor] = []
                    for t_val in t_eval:
                            t_vec = (
                                t_val.expand(x0_samples.shape[0])
                                .clone()
                                .to(device=device, dtype=x0_samples.dtype)
                                .requires_grad_(True)
                            )
                            mu_t = flow_matcher.compute_mu_t(
                                x0_samples,
                                x1_samples,
                                t_vec,
                                t_min,
                                t_max,
                            )
                            interp_points.append(mu_t.detach())
                        interpolant_traj = torch.stack(interp_points, dim=0)
                        interpolant_traj_norm = interpolant_traj.detach()
                        interpolant_traj_denorm = None
                        if cfg.normalize_dataset and min_max is not None:
                            interpolant_traj_denorm = denormalize(
                                interpolant_traj, min_max
                            ).detach()
                        interpolant_traj_norm = interpolant_traj_norm.cpu()
                        if interpolant_traj_denorm is not None:
                            interpolant_traj_denorm = interpolant_traj_denorm.cpu()

                        times_cpu = t_eval.detach().cpu()

                        # Plot endpoints (normalized)
                        x0_norm = x0_samples.detach().cpu()
                        x1_norm = x1_samples.detach().cpu()
                        fig_endpoints_norm, axes_endpoints_norm = plt.subplots(
                            1, 2, figsize=(8, 4)
                        )
                        axes_endpoints_norm = np.atleast_1d(axes_endpoints_norm)
                        plot_cell_samples(
                            axes_endpoints_norm[0],
                            x0_norm,
                            "X0 samples (norm)",
                            color="#1f77b4",
                        )
                        plot_cell_samples(
                            axes_endpoints_norm[1],
                            x1_norm,
                            "X1 samples (norm)",
                            color="#d62728",
                        )
                        fig_endpoints_norm.tight_layout()
                        endpoint_figures_norm.append(fig_endpoints_norm)

                        if interpolant_traj_denorm is not None:
                            x0_denorm = denormalize(x0_samples, min_max).detach().cpu()
                            x1_denorm = denormalize(x1_samples, min_max).detach().cpu()
                            fig_endpoints_denorm, axes_endpoints_denorm = plt.subplots(
                                1, 2, figsize=(8, 4)
                            )
                            axes_endpoints_denorm = np.atleast_1d(axes_endpoints_denorm)
                            plot_cell_samples(
                                axes_endpoints_denorm[0],
                                x0_denorm,
                                "X0 samples (denorm)",
                                color="#1f77b4",
                            )
                            plot_cell_samples(
                                axes_endpoints_denorm[1],
                                x1_denorm,
                                "X1 samples (denorm)",
                                color="#d62728",
                            )
                            fig_endpoints_denorm.tight_layout()
                            endpoint_figures_denorm.append(fig_endpoints_denorm)

                        for idx, (elev, azim) in enumerate(views):
                            fig_norm = plt.figure(figsize=(6, 5))
                            ax_norm = fig_norm.add_subplot(111, projection="3d")
                            plot_cell_trajectories_3d(
                                ax_norm,
                                interpolant_traj_norm,
                                times_cpu,
                                cmap_name="plasma",
                                elev=elev,
                                azim=azim,
                                title="Interpolant trajectories (3D, normalized)",
                            )
                            fig_norm.tight_layout()
                            interpolant_figures_norm.append((fig_norm, f"view{idx}"))

                        if interpolant_traj_denorm is not None:
                            for idx, (elev, azim) in enumerate(views):
                                fig_denorm = plt.figure(figsize=(6, 5))
                                ax_denorm = fig_denorm.add_subplot(111, projection="3d")
                                plot_cell_trajectories_3d(
                                    ax_denorm,
                                    interpolant_traj_denorm,
                                    times_cpu,
                                    cmap_name="magma",
                                    elev=elev,
                                    azim=azim,
                                    title="Interpolant trajectories (3D, denormalized)",
                                )
                                fig_denorm.tight_layout()
                                interpolant_figures_denorm.append((fig_denorm, f"view{idx}"))

                if interpolant_figures_norm:
                    if wandb_run:
                        wandb_run.log(
                            {
                                "eval/interpolants3d_norm": [
                                    wandb.Image(fig_norm, caption=view_name)
                                    for fig_norm, view_name in interpolant_figures_norm
                                ]
                            }
                        )
                    if cfg.save_plot:
                        interp_path = Path(cfg.save_plot)
                        interp_path.parent.mkdir(parents=True, exist_ok=True)
                        for fig_norm, view_name in interpolant_figures_norm:
                            fig_norm.savefig(
                                interp_path.with_name(
                                    f"{interp_path.stem}_interpolants3d_norm_{view_name}{interp_path.suffix}"
                                ),
                                dpi=200,
                            )
                    for fig_norm, _ in interpolant_figures_norm:
                        plt.close(fig_norm)

                if interpolant_figures_denorm:
                    if wandb_run:
                        wandb_run.log(
                            {
                                "eval/interpolants3d_denorm": [
                                    wandb.Image(fig_denorm, caption=view_name)
                                    for fig_denorm, view_name in interpolant_figures_denorm
                                ]
                            }
                        )
                    if cfg.save_plot:
                        interp_path = Path(cfg.save_plot)
                        interp_path.parent.mkdir(parents=True, exist_ok=True)
                        for fig_denorm, view_name in interpolant_figures_denorm:
                            fig_denorm.savefig(
                                interp_path.with_name(
                                    f"{interp_path.stem}_interpolants3d_denorm_{view_name}{interp_path.suffix}"
                                ),
                                dpi=200,
                            )
                    for fig_denorm, _ in interpolant_figures_denorm:
                        plt.close(fig_denorm)

                if endpoint_figures_norm:
                    if wandb_run:
                        wandb_run.log(
                            {
                                "eval/endpoints_norm": [
                                    wandb.Image(fig) for fig in endpoint_figures_norm
                                ]
                            }
                        )
                    if cfg.save_plot:
                        endpoint_path = Path(cfg.save_plot)
                        endpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        for idx, fig_end in enumerate(endpoint_figures_norm):
                            fig_end.savefig(
                                endpoint_path.with_name(
                                    f"{endpoint_path.stem}_endpoints_norm_{idx}{endpoint_path.suffix}"
                                ),
                                dpi=200,
                            )
                    for fig_end in endpoint_figures_norm:
                        plt.close(fig_end)

                if endpoint_figures_denorm:
                    if wandb_run:
                        wandb_run.log(
                            {
                                "eval/endpoints_denorm": [
                                    wandb.Image(fig) for fig in endpoint_figures_denorm
                                ]
                            }
                        )
                    if cfg.save_plot:
                        endpoint_path = Path(cfg.save_plot)
                        endpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        for idx, fig_end in enumerate(endpoint_figures_denorm):
                            fig_end.savefig(
                                endpoint_path.with_name(
                                    f"{endpoint_path.stem}_endpoints_denorm_{idx}{endpoint_path.suffix}"
                                ),
                                dpi=200,
                            )
                    for fig_end in endpoint_figures_denorm:
                        plt.close(fig_end)
            else:
                # 2D plot
                max_panels = min(6, len(test_data) - 1)
                panel_indices = (
                    np.linspace(1, len(test_data) - 1, num=max_panels, dtype=int)
                    if max_panels > 0
                    else np.array([], dtype=int)
                )
                panel_indices = np.unique(panel_indices)
                if len(panel_indices) == 0:
                    panel_indices = np.array([len(test_data) - 1], dtype=int)

                fig_overlay, axes_overlay = plt.subplots(
                    1, len(panel_indices), figsize=(4 * len(panel_indices), 4)
                )
                axes_overlay = np.atleast_1d(axes_overlay)

                for col, idx_time in enumerate(panel_indices):
                    t_target = times[idx_time]
                    idx = torch.argmin(torch.abs(t_eval - t_target)).item()
                    pred = traj[idx].float().cpu()
                    if cfg.normalize_dataset:
                        target = denormalize(test_data[idx_time], min_max).cpu()
                    else:
                        target = test_data[idx_time].cpu()
                    emd_val = float(compute_emd(target.to(device), pred.to(device), device=device))
                    emd_values.append(emd_val)

                    ax = axes_overlay[col]
                    ax.scatter(target[:, 0], target[:, 1], s=10, c="red", alpha=0.4, label="target" if col == 0 else None)
                    ax.scatter(pred[:, 0], pred[:, 1], s=10, c="blue", alpha=0.7, label="prediction" if col == 0 else None)
                    ax.set_title(f"t={t_target.item():.2f}, EMD={emd_val:.3f}")
                    ax.set_aspect("equal")
                    ax.axis("off")
                if len(panel_indices) > 0:
                    axes_overlay[0].legend(frameon=False, loc="upper right")

                fig_overlay.tight_layout()

                if wandb_run:
                    wandb_run.log({"eval/overlay": wandb.Image(fig_overlay)})
                if cfg.save_plot:
                    overlay_path = Path(cfg.save_plot)
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    fig_overlay.savefig(
                        overlay_path.with_name(f"{overlay_path.stem}_overlay{overlay_path.suffix}"),
                        dpi=200,
                    )
                plt.close(fig_overlay)

            if wandb_run and emd_values:
                wandb_run.log({
                    "eval/emd_mean": float(np.mean(emd_values)),
                    "eval/emd_std": float(np.std(emd_values)),
                    "eval/seed": seed,
                })

            if wandb_run is not None:
                artifact_dir = Path(wandb_run.dir) / "artifacts"
                artifact_dir.mkdir(parents=True, exist_ok=True)
                traj_artifact_path = artifact_dir / f"{(cfg.wandb_name or 'mfm_model')}_seed{seed}_traj.pt"
                torch.save(
                    {
                        "trajectory": traj.detach().cpu(),
                        "t_eval": t_eval.detach().cpu(),
                        "times": times.detach().cpu(),
                        "seed": seed,
                        "config": asdict(cfg),
                    },
                    traj_artifact_path,
                )
                wandb_run.save(str(traj_artifact_path), policy="now")

            if cfg.save_checkpoints:
                if cfg.checkpoint_dir is not None:
                    ckpt_dir = Path(cfg.checkpoint_dir)
                elif wandb_run is not None:
                    ckpt_dir = Path(wandb_run.dir) / "checkpoints"
                else:
                    ckpt_dir = Path("checkpoints")

                prefix = cfg.wandb_name or "mfm_model"
                ckpt_path = _save_model_checkpoint(
                    ckpt_dir,
                    prefix,
                    seed,
                    geopath_net,
                    flow_net,
                    cfg,
                    times,
                    min_max,
                )
                if cfg.verbose:
                    print(f"[info] Saved checkpoint to {ckpt_path}")
                if wandb_run is not None:
                    wandb_run.save(str(ckpt_path), policy="now")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def parse_config() -> TrainConfig:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Metric Flow Matching on rotating MNIST")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu/cuda/mps)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use: rotating_mnist or cell_tracking")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--metric-batch-size", type=int, default=None, help="Batch size for metric samples")
    parser.add_argument("--geopath-epochs", type=int, default=None, help="Number of GeoPath epochs")
    parser.add_argument("--geopath-steps", type=int, default=None, help="GeoPath steps per epoch")
    parser.add_argument("--geopath-lr", type=float, default=None, help="Learning rate for GeoPath optimiser")
    parser.add_argument("--flow-epochs", type=int, default=None, help="Number of flow epochs")
    parser.add_argument("--flow-steps", type=int, default=None, help="Flow steps per epoch")
    parser.add_argument("--flow-lr", type=float, default=None, help="Learning rate for flow optimiser")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha weighting for geopath corrections")
    parser.add_argument("--sigma", type=float, default=None, help="Sigma for flow matcher noise")
    parser.add_argument("--gamma", type=float, default=None, help="Gamma parameter for Land metric")
    parser.add_argument("--rho", type=float, default=None, help="Rho regulariser for Land metric")
    parser.add_argument("--alpha-metric", type=float, default=None, help="Alpha for metric weighting")
    parser.add_argument("--metric-velocity", type=str, default=None, help="Velocity metric type (land or rbf)")
    parser.add_argument("--metric-n-centers", type=int, default=None, help="Number of centers for RBF metric")
    parser.add_argument("--metric-kappa", type=float, default=None, help="Kappa parameter for RBF metric")
    parser.add_argument("--metric-epochs", type=int, default=None, help="Training epochs for metric network")
    parser.add_argument("--metric-patience", type=int, default=None, help="Early stopping patience for metric training")
    parser.add_argument("--metric-lr", type=float, default=None, help="Learning rate for metric network")
    parser.add_argument("--n-data-dims", type=int, default=None, help="Number of pixel dimensions to keep")
    parser.add_argument("--normalize-dataset", action="store_true", help="Enable dataset normalization")
    parser.add_argument("--no-normalize-dataset", dest="normalize_dataset", action="store_false")
    parser.add_argument("--ot-method", type=str, default=None, help="OT sampling method (exact / None)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="List of random seeds")
    parser.add_argument("--verbose", action="store_true", help="Print tensor shapes during training")
    parser.add_argument("--save-plot", type=str, default=None, help="If provided, save final comparison image to this path")
    parser.add_argument("--eval-num-timepoints", type=int, default=None, help="Number of evaluation time points for NeuralODE trajectory")
    parser.add_argument("--skip-eval", action="store_true", help="Skip NeuralODE evaluation phase")
    parser.add_argument("--cell-stack-path", type=str, default=None, help="Path to cell-tracking boolean stack .npy")
    parser.add_argument("--cell-test-stack-path", type=str, default=None, help="Optional test stack for evaluation")
    parser.add_argument("--cell-subset-size", type=int, default=None, help="Random subset size per timestep for cell data")
    parser.add_argument("--cell-subset-seed", type=int, default=None, help="Random seed for cell subsampling")
    parser.add_argument("--mnist-unet-base", type=int, default=None, help="Base channel count for MNIST UNet models")
    parser.add_argument("--mnist-interpolant-hidden", type=int, default=None, help="Hidden width for TrainableInterpolantMNIST")
    parser.add_argument("--cell-flow-width", type=int, default=None, help="Hidden width for cell-tracking flow MLP")
    parser.add_argument("--cell-interpolant-hidden", type=int, default=None, help="Hidden width for cell-tracking interpolant")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-name", type=str, default=None, help="wandb run name")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to store final checkpoints")
    parser.add_argument("--save-checkpoints", dest="save_checkpoints", action="store_true", help="Save model checkpoints")
    parser.add_argument("--no-save-checkpoints", dest="save_checkpoints", action="store_false", help="Skip saving model checkpoints")
    parser.add_argument("--st-data-dir", type=str, default=None, help="Directory containing ST CSV files")
    parser.set_defaults(
        normalize_dataset=cfg.normalize_dataset,
        save_checkpoints=cfg.save_checkpoints,
    )

    args = parser.parse_args()

    if args.device is not None:
        cfg.device = args.device
    if args.dataset is not None:
        cfg.dataset = args.dataset
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.metric_batch_size is not None:
        cfg.metric_batch_size = args.metric_batch_size
    if args.geopath_epochs is not None:
        cfg.geopath_epochs = args.geopath_epochs
    if args.geopath_steps is not None:
        cfg.geopath_steps_per_epoch = args.geopath_steps
    if args.geopath_lr is not None:
        cfg.geopath_lr = args.geopath_lr
    if args.flow_epochs is not None:
        cfg.flow_epochs = args.flow_epochs
    if args.flow_steps is not None:
        cfg.flow_steps_per_epoch = args.flow_steps
    if args.flow_lr is not None:
        cfg.flow_lr = args.flow_lr
    if args.alpha is not None:
        cfg.alpha = args.alpha
    if args.sigma is not None:
        cfg.sigma = args.sigma
    if args.gamma is not None:
        cfg.gamma = args.gamma
    if args.rho is not None:
        cfg.rho = args.rho
    if args.alpha_metric is not None:
        cfg.alpha_metric = args.alpha_metric
    if args.metric_velocity is not None:
        cfg.velocity_metric = args.metric_velocity
    if args.metric_n_centers is not None:
        cfg.metric_n_centers = args.metric_n_centers
    if args.metric_kappa is not None:
        cfg.metric_kappa = args.metric_kappa
    if args.metric_epochs is not None:
        cfg.metric_epochs = args.metric_epochs
    if args.metric_patience is not None:
        cfg.metric_patience = args.metric_patience
    if args.metric_lr is not None:
        cfg.metric_lr = args.metric_lr
    if args.n_data_dims is not None:
        cfg.n_data_dims = args.n_data_dims
    if args.ot_method is not None:
        cfg.ot_method = args.ot_method
    if args.seeds is not None and len(args.seeds) > 0:
        cfg.seed_list = args.seeds
    cfg.normalize_dataset = args.normalize_dataset
    cfg.verbose = args.verbose
    cfg.save_plot = args.save_plot
    if args.eval_num_timepoints is not None:
        cfg.eval_num_timepoints = args.eval_num_timepoints
    cfg.skip_eval = args.skip_eval
    if args.cell_stack_path is not None:
        cfg.cell_stack_path = args.cell_stack_path
    if args.cell_test_stack_path is not None:
        cfg.cell_test_stack_path = args.cell_test_stack_path
    if args.cell_subset_size is not None:
        cfg.cell_subset_size = args.cell_subset_size
    if args.cell_subset_seed is not None:
        cfg.cell_subset_seed = args.cell_subset_seed
    if args.mnist_unet_base is not None:
        cfg.mnist_unet_base = args.mnist_unet_base
    if args.mnist_interpolant_hidden is not None:
        cfg.mnist_interpolant_hidden = args.mnist_interpolant_hidden
    if args.cell_flow_width is not None:
        cfg.cell_flow_width = args.cell_flow_width
    if args.cell_interpolant_hidden is not None:
        cfg.cell_interpolant_hidden = args.cell_interpolant_hidden
    if args.no_wandb:
        cfg.use_wandb = False
    if args.wandb_name is not None:
        cfg.wandb_name = args.wandb_name
    if args.wandb_project is not None:
        cfg.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        cfg.wandb_entity = args.wandb_entity
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    cfg.save_checkpoints = args.save_checkpoints
    if args.st_data_dir is not None:
        cfg.st_data_dir = args.st_data_dir
    
    print(cfg.cell_stack_path)
        
    return cfg


if __name__ == "__main__":
    config = parse_config()
    main(config)
