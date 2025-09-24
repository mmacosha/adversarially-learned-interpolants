# Metric Flow Matching Scripts

## 1️. Spatial Transcriptomics (`train_mfm_st.py`)

```
python train_mfm_st.py \
  --st-data-dir data/ST/nicola_data \
  --seeds 42 \
  --geopath-epochs 20 --geopath-steps 2500 \
  --flow-epochs 20   --flow-steps 2500 \
  --gamma 0.20 --rho 5e-4 --alpha-metric 1.0 \
  --piecewise-training \
  --wandb-name mfm_st_leaveout
```

### Key options
- `--st-data-dir`: Root directory containing the ST CSV files.
- `--seeds`: One or more random seeds. The script loops over each and performs the full leave-one-out sweep.
- `--gamma`, `--rho`, `--alpha-metric`: Land-metric hyperparameters (same defaults as the ALI baseline).
- `--geopath-*`, `--flow-*`: Epoch/step counts and learning rates for the geodesic interpolant and flow network.
- `--piecewise-training`: Sample adjacent observed timesteps (multiple-constraints regime). Leave unset to train only on the global endpoints.

### Outputs
- Per-gap EMD values stream to WandB (and are printed) during evaluation; the History panel shows both live loss curves and final summaries.
- For each seed, the script stores the reconstructed segment data under `wandb/run-.../files/artifacts/<run-name>_seed<seed>_segments.pt` for offline inspection.

## 2️. Cell Tracking (`train_mfm_cell.py`)

```
python train_mfm_cell.py \
  --cell-stack-path cell_tracking/exports/Cell4_masks/mask_cell4_stack.npy \
  --seeds 42 \
  --cell-subset-size 10 \
  --geopath-epochs 30 --geopath-steps 2500 \
  --flow-epochs 30   --flow-steps 2500 \
  --gamma 0.4 --rho 1e-3 --alpha-metric 1.0 \
  --piecewise-training \
  --wandb-name mfm_cell_subset
```

### Key options
- `--cell-stack-path`: Path to the boolean mask stack (same format used by `cell_tracking/train_ali.py`).
- `--cell-subset-size`, `--cell-subset-seed`: Optional per-timestep subsampling (defaults to 10 points, matching the ALI baseline). Set the size to 0 to disable subsampling.
- `--whiten`: Apply whitening before normalization (mirrors the ALI option).
- Metric and optimizer flags mirror the ST script.

### Outputs
- Prints and logs the EMD for each successive timestep transition and reports mean/std per seed.
- Saves the full predicted trajectory (normalized/denormalized as appropriate) to `wandb/.../files/artifacts/<run-name>_seed<seed>_traj.pt`.

## Implementation Notes
- Both scripts instantiate `TrainableInterpolant` for the GeoPath module and the ALI `MLP` (time-varying) for the flow network, matching the architectures used in `train_ali.py`.
- Minibatching relies on `sample_x_batch`, ensuring identical sampling behaviour to the ALI reference code.
- Piecewise training samples adjacent observed timesteps; when disabled the models train on the global endpoints.
- WandB logging is required; loss curves and evaluation metrics are updated every 250 training steps.
