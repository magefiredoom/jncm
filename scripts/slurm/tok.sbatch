#!/usr/bin/env bash
#SBATCH --job-name=jncm_tok
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=HH:MM:SS
# TODO(placeholder): wall clock -- replace HH:MM:SS above before submitting.
# TODO(placeholder): GPU type -- --gres=gpu:1 asks for one generic GPU; add a
#   type (for example gpu:A100:1) only if your FIR partition requires it.
# TODO(placeholder): modules -- add 'module load ...' lines if your cluster
#   needs them. None are guessed here.
# TODO(placeholder): venv -- this script sources .venv/bin/activate at the repo
#   root (docs/STATE.md, Environment). Change the path if yours differs.
#
# Train one dev tokenizer variant to the full dev budget (SPEC 4.4/4.5):
#   sbatch --account="$SBATCH_ACCOUNT" --partition="$SBATCH_PARTITION" \
#          --export=ALL,QUANT=vq scripts/slurm/tok.sbatch
#   sbatch --account="$SBATCH_ACCOUNT" --partition="$SBATCH_PARTITION" \
#          --export=ALL,QUANT=noisy scripts/slurm/tok.sbatch
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
if [[ ! -f jncm/train_tokenizer.py ]]; then
  echo "ERROR: jncm/train_tokenizer.py not found in ${PWD}; submit from the repo root" >&2
  exit 1
fi

# Cluster environment from docs/STATE.md "Environment".
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

run_id="tok_${QUANT}_dev"
ckpt="results/${run_id}/ckpt/latest.pt"

cmd=(
  python -m jncm.train_tokenizer
  --config configs/tinystories_dev.yaml
  --set "tokenizer.quantizer=${QUANT}"
  --run_id "${run_id}"
)

if [[ -f "${ckpt}" ]]; then
  echo "Resuming ${run_id} from ${ckpt}"
  cmd+=(--resume "${ckpt}")
else
  echo "No checkpoint at ${ckpt}; starting ${run_id} from scratch"
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
