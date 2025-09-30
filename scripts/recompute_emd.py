#!/usr/bin/env python
"""Recompute EMD metrics for a saved MFM trajectory artifact."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from ali_cfm.data_utils import get_dataset, denormalize
from ali_cfm.loggin_and_metrics import compute_emd


def infer_artifact_path(run_dir: Path, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    artifacts_dir = run_dir / "files" / "artifacts"
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"No artifacts directory found under {run_dir}")
    candidates = sorted(artifacts_dir.glob("*_traj.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No '*_traj.pt' files found under {artifacts_dir}. Specify one via --artifact-path."
        )
    if len(candidates) > 1:
        print("[info] Multiple trajectory artifacts found; using the most recent.")
    return candidates[-1]


def needs_denormalized_inputs(cfg: dict, min_max_available: bool) -> bool:
    dataset_name = cfg.get("dataset", "").lower()
    return dataset_name == "cell_tracking" and cfg.get("normalize_dataset", False) and min_max_available


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute EMD metrics for a saved MFM run")
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to the wandb run directory (e.g. wandb/run-...-ph1zcdri)",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default=None,
        help="Optional explicit path to the *_traj.pt artifact",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    artifact_path = infer_artifact_path(run_dir, args.artifact_path)
    print(f"[info] Loading trajectory artifact: {artifact_path}")
    payload = torch.load(artifact_path, map_location="cpu")

    traj = payload["trajectory"].float()
    times = payload.get("times")
    config = payload.get("config", {})

    dataset_name = config.get("dataset", "").lower()
    if not dataset_name:
        raise KeyError("Dataset name missing in saved config; cannot load reference data.")

    if dataset_name == "st":
        data, min_max = get_dataset(
            "ST",
            config.get("n_data_dims"),
            normalize=config.get("normalize_dataset", True),
            nicola_path=config.get("st_data_dir"),
        )
    elif dataset_name == "cell_tracking":
        data, min_max = get_dataset(
            "cell_tracking",
            config.get("n_data_dims"),
            normalize=config.get("normalize_dataset", True),
            nicola_path=config.get("cell_stack_path"),
            whiten=config.get("whiten", False),
        )
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}' for recomputation script")

    if len(data) != traj.shape[0]:
        raise ValueError(
            f"Mismatch between trajectory length ({traj.shape[0]}) and dataset timesteps ({len(data)})"
        )

    min_max_available = min_max is not None
    denorm_inputs = needs_denormalized_inputs(config, min_max_available)

    emd_values = []
    for idx, (target_frame, pred_frame) in enumerate(zip(data, traj)):
        if config.get("normalize_dataset", False):
            target_eval = denormalize(target_frame, min_max)
            if denorm_inputs or dataset_name == "st":
                pred_eval = pred_frame
            else:
                pred_eval = denormalize(pred_frame, min_max)
        else:
            target_eval = target_frame
            pred_eval = pred_frame

        emd = compute_emd(
            target_eval.to(torch.float32),
            pred_eval.to(torch.float32),
        )
        emd_values.append(float(emd))
        print(f"timestep {idx}: EMD={emd_values[-1]:.6f}")

    emd_array = np.array(emd_values, dtype=float)
    print("--- summary ---")
    print(f"mean={np.mean(emd_array):.6f}")
    print(f"std={np.std(emd_array):.6f}")

    if times is not None:
        print(f"times tensor shape: {times.shape}")


if __name__ == "__main__":
    main()
