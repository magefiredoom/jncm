#!/usr/bin/env bash
#SBATCH --job-name=jncm_jepa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=HH:MM:SS
# TODO(placeholder): wall clock -- replace HH:MM:SS above before submitting.
# TODO(placeholder): GPU type -- --gres=gpu:1 asks for one generic GPU; add a
#   type (for example gpu:h100:1) only if your FIR partition requires it.
# TODO(placeholder): modules -- add 'module load ...' lines if your cluster
#   needs them. None are guessed here.
# TODO(placeholder): venv -- this script sources .venv/bin/activate at the repo
#   root (docs/STATE.md, Environment). Change the path if yours differs.
#
# Train one JEPA joint arm (SPEC section 8, TASKS T5.4): the online noisy
# tokenizer (encoder E and decoder D) and the generator G are updated together
# against EMA targets, starting from a Phase-1 noisy tokenizer checkpoint:
#   sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
#          --export=ALL,RUN_ID=jepa_base_dev,STEPS=10000 \
#          TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt scripts/slurm/jepa.sbatch
#   SETS="jepa.vicreg=false" \
#   sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
#          --export=ALL,RUN_ID=jepa_no_vicreg_dev,STEPS=10000 \
#          TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt scripts/slurm/jepa.sbatch
#
# Environment:
#   RUN_ID    (required) names results/<RUN_ID>/ and its resume checkpoint.
#   TOKENIZER (required) noisy tokenizer checkpoint, results/<tok_run>/ckpt/latest.pt.
#   STEPS     (optional) training budget; unset uses configs/jepa_dev.yaml
#             (generator.train.max_steps = 10000).
#   SETS      (optional) space-separated list of ablation overrides, each one
#             written as key=value and passed as its own --set key=value. A
#             leading "--set=" is tolerated. Example:
#               SETS="jepa.vicreg=false jepa.decoder_through_E=false"
#             Export SETS before sbatch (as above): a space-separated value in
#             sbatch's --export list is fragile, --export=ALL copies the
#             environment variable as it is.
#
# The config is configs/jepa_dev.yaml: head flow, target_chain online, EMA on,
# VICReg on, decoder through E. The ablation flags are the SETS above; the
# config file is not copied per run.
#
# Account and partition are deliberately absent from this file (SPEC 7): Slurm
# does not expand shell variables in #SBATCH lines, so they come from the sbatch
# command line (or from your site's submit wrapper, which may read
# $SLURM_ACCOUNT). Submit from the repo root.
#
# The compute node is offline: the HF_*_OFFLINE variables below forbid downloads
# and any cache miss fails loudly instead of reaching the network.
#
# Wall-clock limits are usually shorter than the dev budget. Resubmitting the
# same command with the same SETS resumes from results/<RUN_ID>/ckpt/latest.pt,
# which the trainer rewrites every generator.train.save_every steps (5000 in
# configs/jepa_dev.yaml), so at most that many steps are redone. Milestone
# checkpoints (10/25/50/100% of the budget) are written behind
# generator.train.save_milestones.

set -euo pipefail

: "${RUN_ID:?set RUN_ID, e.g. sbatch --export=ALL,RUN_ID=jepa_base_dev,STEPS=10000 TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt scripts/slurm/jepa.sbatch}"
: "${TOKENIZER:?set TOKENIZER to a noisy tokenizer checkpoint, e.g. TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt (run scripts/slurm/tok.sbatch with QUANT=noisy first)}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f jncm/train_jepa.py ]]; then
  echo "ERROR: jncm/train_jepa.py not found in ${PWD}; submit from the repo root" >&2
  exit 1
fi
if [[ ! -f "${TOKENIZER}" ]]; then
  echo "ERROR: tokenizer checkpoint ${TOKENIZER} not found; train it first with QUANT=noisy" >&2
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

cmd=(
  python -m jncm.train_jepa
  --config configs/jepa_dev.yaml
  --tokenizer "${TOKENIZER}"
  --run_id "${RUN_ID}"
)

if [[ -n "${STEPS:-}" ]]; then
  cmd+=(--steps "${STEPS}")
fi

if [[ -n "${SETS:-}" ]]; then
  read -r -a sets_array <<< "${SETS}"
  for override in "${sets_array[@]}"; do
    override="${override#--set=}"
    if [[ "${override}" != *=* ]]; then
      echo "ERROR: SETS entry '${override}' is not key=value (e.g. jepa.vicreg=false)" >&2
      exit 2
    fi
    cmd+=(--set "${override}")
  done
fi

ckpt="results/${RUN_ID}/ckpt/latest.pt"

if [[ -f "${ckpt}" ]]; then
  echo "Resuming ${RUN_ID} from ${ckpt}"
  cmd+=(--resume "${ckpt}")
else
  echo "No checkpoint at ${ckpt}; starting ${RUN_ID} from scratch"
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
