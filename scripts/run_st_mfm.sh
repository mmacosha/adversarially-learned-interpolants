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
  --geopath-epochs 20 --geopath-steps 1
  --flow-epochs 20   --flow-steps 1
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

run_job 42 0.20 5e-4 1e-4 2e-4 "gamma0.20_rho5e-4" "--metric-velocity land"
# run_job 42 0.20 5e-4 1e-4 2e-4 "gamma0.20_rho5e-4_piecewise" "--metric-velocity land --piecewise-training"
# run_job 48 0.20 5e-4 1e-4 2e-4 "gamma0.20_rho5e-4" "--metric-velocity land"
# run_job 48 0.20 5e-4 1e-4 2e-4 "gamma0.20_rho5e-4_piecewise" "--metric-velocity land --piecewise-training"



wait
echo "All ST MFM jobs finished."


# Single run: 

# python rotating_MNIST/train_mfm.py --dataset st --st-data-dir data/ST/nicola_data --geopath-epochs 30 --geopath-steps 2500 --flow-epochs 30 --flow-steps 2500 --gamma 0.2 --rho 5e-4 --geopath-lr 1e-4 --flow-lr 2e-4 --seeds 48 --metric-velocity land --wandb-name mfm_st_s48_gamma0.2 --save-plot outputs/mfm_st_s48_gamma0.2_rho5e-4/st_mfm_eval.png --checkpoint-dir checkpoints/mfm_st_s48_gamma0.2_rho5e-4 --device mps

# or with piecewise 


# python rotating_MNIST/train_mfm.py --dataset st --st-data-dir data/ST/nicola_data --geopath-epochs 30 --geopath-steps 2500 --flow-epochs 30 --flow-steps 2500 --gamma 0.2 --rho 5e-4 --geopath-lr 1e-4 --flow-lr 2e-4 --seeds 48 --metric-velocity land --wandb-name mfm_st_s48_gamma0.2_rho5e-4_piecewise --save-plot outputs/mfm_st_s48_gamma0.2_rho5e-4_piecewise/st_mfm_eval.png --checkpoint-dir checkpoints/mfm_st_s48_gamma0.2_rho5e-4_piecewise --device mps --piecewise-training