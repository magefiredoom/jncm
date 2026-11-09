#!/usr/bin/env bash
#SBATCH --job-name=jncm_tok
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32000M
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
# Wall clock above is a template default; override it on the command line
# (the 500-step pilot uses --time=00:20:00, the full runs use the measured
# limit). GPU type: FIR rejects a generic --gres=gpu:1 and requires an
# explicit model, so this asks for one H100.
# Modules: FIR supplies python 3.12 through the module system.
# Venv: .venv/bin/activate at the repo root (docs/STATE.md, Environment).
#
# Train one dev tokenizer variant to the full dev budget (SPEC 4.4/4.5):
#   sbatch --account="$SBATCH_ACCOUNT" --partition="$SBATCH_PARTITION" \
#          --export=ALL,QUANT=vq scripts/slurm/tok.sbatch
#   sbatch --account="$SBATCH_ACCOUNT" --partition="$SBATCH_PARTITION" \
#          --export=ALL,QUANT=noisy scripts/slurm/tok.sbatch
# Optional overrides (defaults are unchanged: tok_<quant>_dev, config steps):
#   --export=ALL,QUANT=vq,RUN_ID=pilot_vq,STEPS=500
#   CONFIG (optional) selects the config; default configs/tinystories_dev.yaml,
#   the lite runs use CONFIG=configs/tinystories_lite.yaml.
#
# Account and partition are deliberately absent from this file (SPEC 7): Slurm
# does not expand shell variables in #SBATCH lines, so they come from the sbatch
# command line above (or from your site's submit wrapper, which may read
# $SLURM_ACCOUNT). Submit from the repo root.
#
# The compute node is offline: the HF_*_OFFLINE variables below forbid downloads
# and any cache miss fails loudly instead of reaching the network.
#
# Wall-clock limits are usually shorter than the dev budget. Resubmitting the
# same command resumes from results/<run_id>/ckpt/latest.pt, which the trainer
# rewrites every train.save_every steps, so at most that many steps are redone.

set -euo pipefail

: "${QUANT:?set QUANT=vq or QUANT=noisy, e.g. sbatch --export=ALL,QUANT=vq scripts/slurm/tok.sbatch}"
case "${QUANT}" in
  vq | noisy) ;;
  *)
    echo "ERROR: QUANT must be 'vq' or 'noisy', got '${QUANT}'" >&2
    exit 2
    ;;
esac

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# T4.5: a finished run is a no-op. Chained chunks beyond the real budget land
# here and exit 0 before a venv, a module or a dataset is touched.
run_id="${RUN_ID:-tok_${QUANT}_dev}"
done_file="results/${run_id}/DONE"
if [[ -f "${done_file}" ]]; then
  echo "DONE exists at ${done_file}; ${run_id} is finished, nothing to do"
  exit 0
fi

config="${CONFIG:-configs/tinystories_dev.yaml}"
if [[ -n "${CONFIG:-}" && ! -f "${config}" ]]; then
  echo "ERROR: CONFIG ${config} not found" >&2
  exit 1
fi

if [[ ! -f jncm/train_tokenizer.py ]]; then
  echo "ERROR: jncm/train_tokenizer.py not found in ${PWD}; submit from the repo root" >&2
  exit 1
fi

# Cluster environment from docs/STATE.md "Environment".
module load python/3.12
if [[ ! -f .venv/bin/activate ]]; then
  echo "ERROR: .venv/bin/activate not found; create the cluster venv (docs/STATE.md)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

ckpt="results/${run_id}/ckpt/latest.pt"

cmd=(
  python -m jncm.train_tokenizer
  --config "${config}"
  --set "tokenizer.quantizer=${QUANT}"
  --run_id "${run_id}"
)
if [[ -n "${STEPS:-}" ]]; then
  cmd+=(--steps "${STEPS}")
fi

if [[ -f "${ckpt}" ]]; then
  echo "Resuming ${run_id} from ${ckpt}"
  cmd+=(--resume "${ckpt}")
else
  echo "No checkpoint at ${ckpt}; starting ${run_id} from scratch"
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
