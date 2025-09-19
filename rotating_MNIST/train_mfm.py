import argparse
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
from torch import nn
from torch.optim import Adam
from torchdyn.core import NeuralODE
from tqdm.auto import trange


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


from ali_cfm.data_utils import denormalize, get_dataset
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

    dataset: str = "rotating_mnist"  # or "cell_tracking"

    batch_size: int = 128
    metric_batch_size: int = 512
    geopath_epochs: int = 5
    geopath_steps_per_epoch: int = 200
    geopath_lr: float = 5e-4

    flow_epochs: int = 10
    flow_steps_per_epoch: int = 500
    flow_lr: float = 1e-3

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
    cell_subset_size: Optional[int] = None
    cell_subset_seed: Optional[int] = 42

    velocity_metric: str = "land"
    metric_n_centers: int = 100
    metric_kappa: float = 0.5
    metric_epochs: int = 0
    metric_patience: int = 10
    metric_lr: float = 1e-3

    mnist_unet_base: int = 32
    mnist_interpolant_hidden: int = 512
    mnist_interpolant_hidden: int = 512
    cell_flow_width: int = 128
    cell_interpolant_hidden: int = 512
    

def _sample_batches(
    per_timestep_data: Sequence[torch.Tensor],
    batch_size: int,
) -> List[torch.Tensor]:
    device = per_timestep_data[0].device
    batches = []
    for tensor in per_timestep_data:
        idx = torch.randint(0, tensor.shape[0], (batch_size,), device=device)
        batches.append(tensor.index_select(0, idx))
    return batches


def train_geopath(
    flow_matcher: MetricFlowMatcher,
    optimizer: torch.optim.Optimizer,
    data: Sequence[torch.Tensor],
    times: torch.Tensor,
    gamma: float,
    rho: float,
    metric_batch_size: int,
    steps: int,
    batch_size: int,
    ot_sampler: Optional[OTPlanSampler],
    data_metric: DataManifoldMetric,
    log_shapes: bool = False,
) -> float:
    flow_matcher.train()
    geopath_net: nn.Module = flow_matcher.geopath_net
    geopath_net.train()

    total_loss = 0.0

    for _ in trange(steps, desc="Training GeoPath", leave=False):
        optimizer.zero_grad()

        main_batch = _sample_batches(data, batch_size)
        metric_batch = _sample_batches(data, metric_batch_size)

        velocities: List[torch.Tensor] = []

        for i in range(len(data) - 1):
            x0 = main_batch[i]
            x1 = main_batch[i + 1]

            metric_samples = torch.cat([metric_batch[i], metric_batch[i + 1]], dim=0)

            if ot_sampler is not None:
                x0, x1 = ot_sampler.sample_plan(x0, x1, replace=True)

            t_min = times[i]
            t_max = times[i + 1]
            t, xt, ut = flow_matcher.sample_location_and_conditional_flow(
                x0,
                x1,
                t_min,
                t_max,
                training_geopath_net=True,
            )

            if log_shapes:
                print(
                    "[GeoPath] pair", i,
                    "| x0", tuple(x0.shape),
                    "x1", tuple(x1.shape),
                    "xt", tuple(xt.shape),
                    "ut", tuple(ut.shape),
                    "metric", tuple(metric_samples.shape),
                )

            vel = data_metric.calculate_velocity(
                xt,
                ut,
                metric_samples,
                timestep=i,
            )
            velocities.append(vel)

        loss = torch.mean(torch.cat(velocities) ** 2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

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
    log_shapes: bool = False,
) -> float:
    flow_matcher.train()
    flow_net.train()

    total_loss = 0.0

    for _ in trange(steps, desc="Training Flow", leave=False):
        optimizer.zero_grad()

        main_batch = _sample_batches(data, batch_size)

        ts, xts, uts = [], [], []

        for i in range(len(data) - 1):
            x0 = main_batch[i]
            x1 = main_batch[i + 1]

            if ot_sampler is not None:
                x0, x1 = ot_sampler.sample_plan(x0, x1, replace=True)

            t_min = times[i]
            t_max = times[i + 1]
            t, xt, ut = flow_matcher.sample_location_and_conditional_flow(
                x0,
                x1,
                t_min,
                t_max,
            )

            if log_shapes and not ts:
                print(
                    "[Flow ] pair", i,
                    "| x0", tuple(x0.shape),
                    "x1", tuple(x1.shape),
                    "xt", tuple(xt.shape),
                    "ut", tuple(ut.shape),
                )

            ts.append(t[:, None])
            xts.append(xt)
            uts.append(ut)

        t_batch = torch.cat(ts, dim=0)
        xt_batch = torch.cat(xts, dim=0)
        ut_batch = torch.cat(uts, dim=0)

        vt = flow_net(t_batch, xt_batch)
        loss = torch.mean((vt - ut_batch) ** 2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, steps)


class FlowMLPWrapper(nn.Module):
    def __init__(self, dim: int, width: int):
        super().__init__()
        self.model = MLP(dim=dim, w=width, time_varying=True)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t[:, None]
        return self.model(torch.cat([x, t], dim=-1))


def load_cell_tracking_stack(
    stack_path: Union[str, Path],
    subset_size: Optional[int] = None,
    seed: Optional[int] = None,
    normalize: bool = True,
) -> Tuple[List[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    stack_path = Path(stack_path)
    if not stack_path.exists():
        raise FileNotFoundError(f"Cell-tracking stack not found: {stack_path}")

    stack = np.load(stack_path)
    if stack.ndim != 3:
        raise ValueError(
            f"Expected stacked boolean masks with shape (T, H, W), got {stack.shape}"
        )

    rng = np.random.default_rng(seed)
    frames: List[torch.Tensor] = []
    for t in range(stack.shape[0]):
        coords = np.argwhere(stack[t])  # (K, 2) -> (y, x)
        if coords.size == 0:
            raise ValueError(f"Frame {t} in {stack_path} contains no active pixels.")
        coords = coords[:, [1, 0]].astype(np.float32)  # swap -> (x, y)

        if subset_size is not None and coords.shape[0] > subset_size:
            idx = rng.choice(coords.shape[0], subset_size, replace=False)
            coords = coords[idx]

        frames.append(torch.from_numpy(coords))

    if not normalize:
        return frames, None

    stacked = torch.cat(frames, dim=0)
    min_ = stacked.min(0).values
    max_ = stacked.max(0).values
    scale = (max_ - min_).clamp_min(1e-8)
    frames = [(frame - min_) / scale for frame in frames]
    return frames, (min_, max_)


def plot_cell_samples(ax, samples: torch.Tensor, title: str) -> None:
    pts = samples.detach().cpu().numpy()
    ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.7)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


def main(cfg: TrainConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"
    warnings.filterwarnings("ignore")

    try:
        device = torch.device(cfg.device)
        torch.tensor(0, device=device)
    except Exception as exc:
        print(f"[warn] failed to use device '{cfg.device}': {exc}. Falling back to CPU.")
        device = torch.device("cpu")

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
            raise ValueError(
                "--cell-stack-path must be provided when using dataset='cell_tracking'"
            )
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
    else:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'")
    
    print("Using dataset: ", cfg.dataset)

    data = [x.to(device).float() for x in data]
    test_data = [x.to(device).float() for x in test_data]

    times = torch.linspace(0.0, 1.0, len(data), device=device)
    dim = data[0].shape[1]

    print(
        f"Loaded {cfg.dataset} with {len(data)} timesteps; "
        f"min batch {min(x.shape[0] for x in data)}; feature dim {dim}"
    )
    for idx, frame in enumerate(data):
        print(f"  train t={idx}: {tuple(frame.shape)}")

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
    
    data_metric = DataManifoldMetric(args=metric_args, skipped_time_points=None, datamodule=None)

    for seed in cfg.seed_list:
        fix_seed(seed)

        if dataset_name == "rotating_mnist":
            geopath_net = TrainableInterpolantMNIST(
                dim=dim,
                h_dim=cfg.mnist_interpolant_hidden,
                t_smooth=0.0,
                time_varying=True,
                regulariser="linear",
            ).to(device)
            flow_base = cfg.mnist_unet_base
            flow_net = CorrectionUNet(in_ch=2, base=flow_base, interpolant=False).to(device)
        else:
            geopath_net = TrainableInterpolant(
                dim=dim,
                h_dim=cfg.cell_interpolant_hidden,
                t_smooth=0.0,
                time_varying=True,
                regulariser="linear",
            ).to(device)

            flow_net = FlowMLPWrapper(
                dim=dim,
                width=cfg.cell_flow_width,
            ).to(device)

        flow_matcher = MetricFlowMatcher(
            geopath_net=geopath_net,
            alpha=float(cfg.alpha),
            sigma=float(cfg.sigma),
        )

        geopath_optimizer = Adam(
            geopath_net.parameters(),
            lr=cfg.geopath_lr,
        )

        total_geopath_steps = cfg.geopath_epochs * cfg.geopath_steps_per_epoch
        geopath_loss = None
        if total_geopath_steps > 0 and len(list(geopath_net.parameters())) > 0:
            geopath_loss = train_geopath(
                flow_matcher,
                geopath_optimizer,
                data,
                times,
                cfg.gamma,
                cfg.rho,
                cfg.metric_batch_size,
                total_geopath_steps,
                cfg.batch_size,
                ot_sampler,
                data_metric,
                log_shapes=cfg.verbose,
            )
            print(f"[seed={seed}] GeoPath loss: {geopath_loss:.6f}")

        for param in geopath_net.parameters():
            param.requires_grad_(False)
        geopath_net.eval()

        flow_optimizer = Adam(flow_net.parameters(), lr=cfg.flow_lr)

        total_flow_steps = cfg.flow_epochs * cfg.flow_steps_per_epoch
        flow_loss = None
        if total_flow_steps > 0:
            flow_loss = train_flow(
                flow_matcher,
                flow_net,
                flow_optimizer,
                data,
                times,
                total_flow_steps,
                cfg.batch_size,
                ot_sampler,
                log_shapes=cfg.verbose,
            )
            print(f"[seed={seed}] Flow loss: {flow_loss:.6f}")

        if cfg.skip_eval:
            continue

        flow_wrapper = flow_model_torch_wrapper(flow_net).to(device)
        node = NeuralODE(flow_wrapper, solver="dopri5", sensitivity="adjoint").to(device)

        t_eval = torch.linspace(0, 1, cfg.eval_num_timepoints, device=device)
        X0 = test_data[0]

        with torch.no_grad():
            traj = node.trajectory(denormalize(X0, min_max), t_span=t_eval)

        if dataset_name == "rotating_mnist":
            img_dim = int(np.sqrt(dim))
            fig, axes = plt.subplots(2, len(test_data) - 1, figsize=(20, 4))
            for i in range(1, len(test_data)):
                t_target = times[i]
                idx = torch.argmin(torch.abs(t_eval - t_target)).item()
                pred = traj[idx].float().to(device)
                target = denormalize(test_data[i], min_max).to(device)
                emd_val = float(compute_emd(target, pred, device=device))
                axes[0, i - 1].imshow(pred[0].view(img_dim, img_dim).cpu(), cmap="gray")
                axes[0, i - 1].set_title(f"t={360 * t_target.item():.0f}°, EMD={emd_val:.3f}")
                axes[0, i - 1].axis("off")
                axes[1, i - 1].imshow(target[0].view(img_dim, img_dim).cpu(), cmap="gray")
                axes[1, i - 1].axis("off")
        else:
            fig, axes = plt.subplots(2, len(test_data) - 1, figsize=(14, 6))
            for i in range(1, len(test_data)):
                t_target = times[i]
                idx = torch.argmin(torch.abs(t_eval - t_target)).item()
                pred = traj[idx].float().to(device)
                target = denormalize(test_data[i], min_max).to(device)
                emd_val = float(compute_emd(target, pred, device=device))
                plot_cell_samples(
                    axes[0, i - 1], pred, f"t={t_target.item():.2f}, EMD={emd_val:.3f}"
                )
                plot_cell_samples(axes[1, i - 1], target, "target")

        plt.tight_layout()
        if cfg.save_plot:
            fig.savefig(cfg.save_plot, dpi=200)
        plt.close(fig)


def parse_config() -> TrainConfig:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Metric Flow Matching on rotating MNIST")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu/cuda/mps)")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to use: rotating_mnist or cell_tracking")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--metric-batch-size", type=int, default=None, help="Batch size for metric samples")
    parser.add_argument("--geopath-epochs", type=int, default=None, help="Number of GeoPath epochs")
    parser.add_argument("--geopath-steps", type=int, default=None, help="GeoPath steps per epoch")
    parser.add_argument("--flow-epochs", type=int, default=None, help="Number of flow epochs")
    parser.add_argument("--flow-steps", type=int, default=None, help="Flow steps per epoch")
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
    parser.add_argument("--mnist-interpolant-t-smooth", type=float, default=None, help="t_smooth for TrainableInterpolantMNIST")
    parser.add_argument("--mnist-interpolant-regulariser", type=str, default=None, help="Regulariser for TrainableInterpolantMNIST")
    parser.add_argument("--cell-flow-width", type=int, default=None, help="Hidden width for cell-tracking flow MLP")
    parser.add_argument("--cell-interpolant-hidden", type=int, default=None, help="Hidden width for cell-tracking interpolant")
    parser.add_argument("--cell-interpolant-t-smooth", type=float, default=None, help="t_smooth for cell-tracking interpolant")
    parser.add_argument("--cell-interpolant-regulariser", type=str, default=None, help="Regulariser for cell-tracking interpolant")
    parser.set_defaults(normalize_dataset=cfg.normalize_dataset)

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
    if args.flow_epochs is not None:
        cfg.flow_epochs = args.flow_epochs
    if args.flow_steps is not None:
        cfg.flow_steps_per_epoch = args.flow_steps
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
    if args.mnist_interpolant_t_smooth is not None:
        cfg.mnist_interpolant_t_smooth = args.mnist_interpolant_t_smooth
    if args.mnist_interpolant_regulariser is not None:
        cfg.mnist_interpolant_regulariser = args.mnist_interpolant_regulariser
    if args.cell_flow_width is not None:
        cfg.cell_flow_width = args.cell_flow_width
    if args.cell_interpolant_hidden is not None:
        cfg.cell_interpolant_hidden = args.cell_interpolant_hidden
    if args.cell_interpolant_t_smooth is not None:
        cfg.cell_interpolant_t_smooth = args.cell_interpolant_t_smooth
    if args.cell_interpolant_regulariser is not None:
        cfg.cell_interpolant_regulariser = args.cell_interpolant_regulariser
    return cfg


if __name__ == "__main__":
    config = parse_config()
    main(config)
