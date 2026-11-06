#!/usr/bin/env bash
#SBATCH --job-name=jncm_prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=HH:MM:SS
# TODO(placeholder): wall clock -- replace HH:MM:SS above before submitting.
# TODO(placeholder): modules -- add 'module load ...' lines if your cluster
#   needs them. None are guessed here.
# TODO(placeholder): venv -- this script sources .venv/bin/activate at the repo
#   root (docs/STATE.md, Environment). Change the path if yours differs.
#
# Prepare TinyStories under data/tinystories/ (SPEC 3, TASKS T0.2):
# train.bin, val.bin, meta.json, tokenizer.json, at L=128 and V=8192 so they
# match configs/tinystories_dev.yaml (train_tokenizer refuses a mismatch).
#   sbatch --account="$SBATCH_ACCOUNT" --partition="$SBATCH_PARTITION" \
#          scripts/slurm/prep.sbatch
#
# Account and partition are deliberately absent from this file (SPEC 7): Slurm
# does not expand shell variables in #SBATCH lines, so they come from the sbatch
# command line above (or from your site's submit wrapper, which may read
# $SLURM_ACCOUNT). Submit from the repo root.
#
# No GPU is requested: tokenization is CPU-only and prep is numpy/tokenizers.
# This step downloads roneneldan/TinyStories from Hugging Face, so it needs a
# node where the Hub is reachable (an internet-enabled partition, or run it on a
# login node instead). It therefore does NOT set the HF_*_OFFLINE variables;
# those belong to the offline compute jobs (tok.sbatch, later gen/eval).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f scripts/prep_tinystories.py ]]; then
  echo "ERROR: scripts/prep_tinystories.py not found in ${PWD}; submit from the repo root" >&2
  exit 1
fi

# Cluster environment from docs/STATE.md "Environment".
if [[ ! -f .venv/bin/activate ]]; then
  echo "ERROR: .venv/bin/activate not found; create the cluster venv (docs/STATE.md)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

export PYTHONUNBUFFERED=1

cmd=(
  python scripts/prep_tinystories.py
  --out data/tinystories
  --seq_len 128
  --vocab_size 8192
)

echo "Running: ${cmd[*]}"
"${cmd[@]}"
