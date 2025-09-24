#!/usr/bin/env bash
# Launch multiple cell-tracking MFM sweeps in parallel with variant hyperparameters.
# Uses the active environment by default; set PYTHON_CMD to override.

set -euo pipefail

# Choose python executable (falls back to current env)
if [[ -z "${PYTHON_CMD:-}" ]]; then
  if command -v conda >/dev/null 2>&1 && conda info --envs | awk '{print $1}' | grep -qx "mixture-fmls"; then
    PYTHON_CMD=(conda run -n mixture-fmls python)
  else
    PYTHON_CMD=(python)
  fi
else
  # shellcheck disable=SC2206
  PYTHON_CMD=(${PYTHON_CMD})
fi

COMMON_ARGS=(
  rotating_MNIST/train_mfm.py
  --dataset cell_tracking
  --cell-stack-path cell_tracking/exports/Cell4_masks/mask_cell4_stack.npy
  --cell-subset-size 10
  --geopath-epochs 30 --geopath-steps 2500
  --flow-epochs 30   --flow-steps 2500
  --device mps
  --save-checkpoints
)

run_job() {
  local seed="$1" gamma="$2" geopath_lr="$3" flow_lr="$4" name_suffix="$5" extra_args="$6"
  local wandb_name="mfm_celltrack_s${seed}_${name_suffix}"
  local log_dir="logs"
  local out_dir="outputs/${wandb_name}"
  local ckpt_dir="checkpoints/${wandb_name}"

  mkdir -p "$log_dir" "$out_dir" "$ckpt_dir"

  MPLCONFIGDIR=${MPLCONFIGDIR:-./.mplconfig} \
  "${PYTHON_CMD[@]}" "${COMMON_ARGS[@]}" \
    --gamma "$gamma" \
    --geopath-lr "$geopath_lr" \
    --flow-lr "$flow_lr" \
    --seeds "$seed" \
    --wandb-name "$wandb_name" \
    --save-plot "${out_dir}/cell_mfm_eval.png" \
    --checkpoint-dir "$ckpt_dir" \
    ${extra_args} \
    >"${log_dir}/run_${wandb_name}.log" 2>&1 &
}

# Example sweeps; uncomment the variants you need.
run_job 42 0.4 1e-4 1e-4 "gamma_0.4_land" "--metric-velocity land --piecewise-training"
run_job 42 0.4 1e-4 1e-4 "gamma_0.4_land" "--metric-velocity land"

wait
echo "All MFM jobs finished."

# Single run:

# python rotating_MNIST/train_mfm.py --dataset cell_tracking --cell-stack-path cell_tracking/exports/Cell4_masks/mask_cell4_stack.npy --cell-subset-size 10 --geopath-epochs 30 --geopath-steps 2500 --flow-epochs 30 --flow-steps 2500 --gamma 0.4 --geopath-lr 1e-4 --flow-lr 1e-4 --seeds 42 --metric-velocity land --piecewise-training --wandb-name mfm_celltrack_s42_gamma_0.4_land --save-plot outputs/mfm_celltrack_s42_gamma_0.4_land/cell_mfm_eval.png --checkpoint-dir checkpoints/mfm_celltrack_s42_gamma_0.4_land --device mps
