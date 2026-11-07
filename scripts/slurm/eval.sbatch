#!/usr/bin/env bash
#SBATCH --job-name=jncm_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --time=HH:MM:SS
# TODO(placeholder): wall clock -- replace HH:MM:SS above before submitting.
# TODO(placeholder): array range -- 0-4 matches N_SEEDS=5 (one task per seed).
#   Widen or narrow it together with N_SEEDS.
# TODO(placeholder): GPU type -- --gres=gpu:1 asks for one generic GPU; add a
#   type (for example gpu:A100:1) only if your FIR partition requires it.
# TODO(placeholder): modules -- add 'module load ...' lines if your cluster
#   needs them. None are guessed here.
# TODO(placeholder): venv -- this script sources .venv/bin/activate at the repo
#   root (docs/STATE.md, Environment). Change the path if yours differs.
#
# Evaluate the milestone checkpoints of one generator run (SPEC 6.1, T4.2):
#   sbatch --account="$SBATCH_ACCOUNT" --partition="$SBATCH_PARTITION" \
#          --export=ALL,RUN_ID=gen_vq_ce_dev \
#          TOKENIZER=results/tok_vq_dev/ckpt/latest.pt scripts/slurm/eval.sbatch
#   then, once the array has finished:
#   python scripts/run_eval.py --run_id gen_vq_ce_dev --merge
#
# One array task is one seed: it writes results/<RUN_ID>/eval/step_<n>.seed_<i>.json
# for every milestone checkpoint (10/25/50/100% of training). --merge folds
# those shards into results/<RUN_ID>/eval/step_<n>.json, which
# scripts/plot_curves.py reads. Running scripts/run_eval.py once without --seeds
# does all seeds in one process and writes the aggregate directly.
#
# TOKENIZER is the frozen tokenizer checkpoint the generator was trained with;
# RUN_ID names results/<RUN_ID>/ (its ckpt/milestone_*.pt files are evaluated).
# N_SEEDS (default 5) and N_SAMPLES (default 256) match SPEC 6.1; GPT2_MODEL_PATH
# defaults to models/gpt2-large.
#
# Account and partition are deliberately absent from this file (SPEC 7): Slurm
# does not expand shell variables in #SBATCH lines, so they come from the sbatch
# command line above (or from your site's submit wrapper, which may read
# $SLURM_ACCOUNT). Submit from the repo root.
#
# The compute node is offline: the HF_*_OFFLINE variables below forbid downloads
# and any cache miss (GPT-2 weights, tokenizer) fails loudly instead of reaching
# the network.

set -euo pipefail

: "${RUN_ID:?set RUN_ID, e.g. sbatch --export=ALL,RUN_ID=gen_vq_ce_dev TOKENIZER=results/tok_vq_dev/ckpt/latest.pt scripts/slurm/eval.sbatch}"
: "${TOKENIZER:?set TOKENIZER to the frozen tokenizer checkpoint of the generator, e.g. TOKENIZER=results/tok_vq_dev/ckpt/latest.pt}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f scripts/run_eval.py ]]; then
  echo "ERROR: scripts/run_eval.py not found in ${PWD}; submit from the repo root" >&2
  exit 1
fi
if [[ ! -f "${TOKENIZER}" ]]; then
  echo "ERROR: tokenizer checkpoint ${TOKENIZER} not found" >&2
  exit 1
fi
if [[ ! -d "results/${RUN_ID}/ckpt" ]]; then
  echo "ERROR: results/${RUN_ID}/ckpt not found; train the generator first (gen.sbatch)" >&2
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

n_seeds="${N_SEEDS:-5}"
n_samples="${N_SAMPLES:-256}"
gpt2="${GPT2_MODEL_PATH:-models/gpt2-large}"
seed="${SLURM_ARRAY_TASK_ID:-0}"

cmd=(
  python scripts/run_eval.py
  --config configs/tinystories_dev.yaml
  --run_id "${RUN_ID}"
  --tokenizer "${TOKENIZER}"
  --seeds "${seed}"
  --n_seeds "${n_seeds}"
  --n_samples "${n_samples}"
  --gpt2 "${gpt2}"
)

echo "Running: ${cmd[*]}"
"${cmd[@]}"
