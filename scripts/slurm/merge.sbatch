#!/usr/bin/env bash
#SBATCH --job-name=jncm_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
# Merge the per-seed eval shards of one run (T4.9, SPEC 6.1 v1.7): a CPU job, so
# that no Python runs on a login node. It requests no GPU and no array; chain.sh
# --merge-after submits it with --dependency=afterok:<eval array id>, which waits
# for every task of the array and never starts if one of them fails (Slurm then
# keeps the job PENDING with reason DependencyNeverSatisfied, or cancels it if the
# cluster sets kill_invalid_depend; not yet checked on FIR). Rerun the lost seed
# alone with --array=<i> (SPEC 10.3), then submit this merge again by hand.
#
#   sbatch --account="$SBATCH_ACCOUNT" --dependency=afterok:<eval job id> \
#          --export=ALL,RUN_ID=gen_vq_ce_dev scripts/slurm/merge.sbatch
#
# It runs `python scripts/run_eval.py --run_id "$RUN_ID" --merge`, which folds
# results/<RUN_ID>/eval/step_<n>.seed_<i>.json into step_<n>.json without
# rescoring (no GPT-2, no checkpoint). Resources: 1 CPU and 4 GB cover
# `import torch` (about 230 MB on FIR) and the JSON shards; 15 minutes cover the
# cold torch import from the cluster filesystem (about 93 s measured on FIR).
#
# Account and partition are deliberately absent from this file (SPEC 7): they
# come from the sbatch command line or the environment. Submit from the repo root.

set -euo pipefail

if [[ -z "${RUN_ID:-}" ]]; then
  echo "ERROR: RUN_ID is unset or empty; submit with --export=ALL,RUN_ID=<run> (T4.9)" >&2
  exit 2
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ ! -f scripts/run_eval.py ]]; then
  echo "ERROR: scripts/run_eval.py not found in ${PWD}; submit from the repo root" >&2
  exit 1
fi
if [[ ! -d "results/${RUN_ID}/eval" ]]; then
  echo "ERROR: results/${RUN_ID}/eval not found; run the eval array first (eval.sbatch)" >&2
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

export PYTHONUNBUFFERED=1

cmd=(python scripts/run_eval.py --run_id "${RUN_ID}" --merge)
echo "Running: ${cmd[*]}"
"${cmd[@]}"
