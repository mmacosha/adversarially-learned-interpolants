#!/usr/bin/env bash
# Launch multiple ST dataset MFM runs in parallel.

set -euo pipefail

if [[ -z "${PYTHON_CMD:-}" ]]; then
  if command -v conda >/dev/null 2>&1 && conda info --envs | awk '{print $1}' | grep -qx "mixture-fmls"; then
    PYTHON_CMD=(conda run -n mixture-fmls python)
  else
    PYTHON_CMD=(python)
  fi
else
  PYTHON_CMD=(${PYTHON_CMD})
fi

COMMON_ARGS=(
  rotating_MNIST/train_mfm.py
  --dataset st
  --st-data-dir data/ST/nicola_data
  --geopath-epochs 20 --geopath-steps 2500
  --flow-epochs 20   --flow-steps 2500
  --device mps
  --save-checkpoints
)

run_job() {
  local seed="$1" gamma="$2" rho="$3" geopath_lr="$4" flow_lr="$5" suffix="$6" extra_args="$7"
  local wandb_name="mfm_st_s${seed}_${suffix}"
  local log_dir="logs"
  local out_dir="outputs/${wandb_name}"
  local ckpt_dir="checkpoints/${wandb_name}"

  mkdir -p "$log_dir" "$out_dir" "$ckpt_dir"

  MPLCONFIGDIR=${MPLCONFIGDIR:-./.mplconfig} \
  "${PYTHON_CMD[@]}" "${COMMON_ARGS[@]}" \
    --gamma "$gamma" \
    --rho "$rho" \
    --geopath-lr "$geopath_lr" \
    --flow-lr "$flow_lr" \
    --seeds "$seed" \
    --wandb-name "$wandb_name" \
    --save-plot "${out_dir}/st_mfm_eval.png" \
    --checkpoint-dir "$ckpt_dir" \
    ${extra_args} \
    >"${log_dir}/run_${wandb_name}.log" 2>&1 &
}

mkdir -p logs outputs checkpoints

run_job 46 0.15 5e-4 1e-4 2e-4 "gamma0.15_rho5e-4" "--metric-velocity land"
run_job 47 0.15 1e-3 5e-4 2e-4 "gamma0.15_rho1e-3" "--metric-velocity land"
# run_job 48 0.20 5e-4 1e-4 2e-4 "gamma0.20_rho5e-4_piecewise" "--metric-velocity land --piecewise-training"
# run_job 49 0.20 1e-3 5e-4 2e-4 "gamma0.20_rho1e-3_piecewise" "--metric-velocity land --piecewise-training"
run_job 50 0.20 5e-4 1e-4 1e-4 "gamma0.20_rho5e-4_all1e-4" "--metric-velocity land" 
run_job 50 0.20 5e-4 1e-4 1e-4 "gamma0.20_rho5e-4_all1e-4_piecewise" "--metric-velocity land" "--piecewise-training"  

# run_job 50 0.30 5e-4 1e-4 2e-4 "gamma0.30_rho5e-4" "--metric-velocity land"
# run_job 51 0.30 1e-3 5e-4 2e-4 "gamma0.30_rho1e-3" "--metric-velocity land"

wait
echo "All ST MFM jobs finished."
