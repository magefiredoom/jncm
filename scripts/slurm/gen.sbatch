#!/usr/bin/env bash
#SBATCH --job-name=jncm_gen
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
# Train one generator arm against a frozen tokenizer checkpoint (SPEC 5.5-5.7):
#   sbatch --account="$SBATCH_ACCOUNT" --partition="$SBATCH_PARTITION" \
#          --export=ALL,HEAD=ce,RUN_ID=gen_vq_ce_dev \
#          TOKENIZER=results/tok_vq_dev/ckpt/latest.pt scripts/slurm/gen.sbatch
#   sbatch --account="$SBATCH_ACCOUNT" --partition="$SBATCH_PARTITION" \
#          --export=ALL,HEAD=regress,RUN_ID=gen_noisy_regress_dev \
#          TOKENIZER=results/tok_noisy_dev/ckpt/latest.pt scripts/slurm/gen.sbatch
#   (HEAD=flow is the same with RUN_ID=gen_noisy_flow_dev.)
#
# Account and partition are deliberately absent from this file (SPEC 7): Slurm
# does not expand shell variables in #SBATCH lines, so they come from the sbatch
# command line above (or from your site's submit wrapper, which may read
# $SLURM_ACCOUNT). Submit from the repo root.
#
# HEAD chooses the head and, by the SPEC 5.3 pairing, the tokenizer variant:
# ce -> vq, regress/flow -> noisy. TOKENIZER is the frozen tokenizer checkpoint
# (from tok.sbatch), RUN_ID names results/<RUN_ID>/ and its resume checkpoint.
# CONFIG is optional (default configs/tinystories_dev.yaml); the lite runs use
# CONFIG=configs/tinystories_lite.yaml.
# STEPS is optional (unset = the config budget, 10,000 in the dev config); the
# pilots pass STEPS=300. GEN_PPL=1 appends --gen_ppl, which decodes the samples
# and scores them through the local GPT-2-large (offline; config
# eval.gpt2_model_path, models/gpt2-large); the pilots always turn it on.
#
# The compute node is offline: the HF_*_OFFLINE variables below forbid downloads
# and any cache miss fails loudly instead of reaching the network.
#
# Wall-clock limits are usually shorter than the dev budget. Resubmitting the
# same command resumes from results/<RUN_ID>/ckpt/latest.pt, which the trainer
# rewrites every generator.train.save_every steps (5000 in the dev config), so
# at most that many steps are redone.

set -euo pipefail

: "${HEAD:?set HEAD=ce, regress or flow, e.g. sbatch --export=ALL,HEAD=ce,RUN_ID=gen_vq_ce_dev TOKENIZER=results/tok_vq_dev/ckpt/latest.pt scripts/slurm/gen.sbatch}"
: "${TOKENIZER:?set TOKENIZER to a tokenizer checkpoint, e.g. TOKENIZER=results/tok_vq_dev/ckpt/latest.pt (run scripts/slurm/tok.sbatch first)}"
: "${RUN_ID:?set RUN_ID, e.g. RUN_ID=gen_vq_ce_dev}"

case "${HEAD}" in
  ce) quant=vq ;;
  regress | flow) quant=noisy ;;
  *)
    echo "ERROR: HEAD must be 'ce', 'regress' or 'flow', got '${HEAD}'" >&2
    exit 2
    ;;
esac

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# T4.5: a finished run is a no-op (chained surplus chunks exit here, cheaply).
done_file="results/${RUN_ID}/DONE"
if [[ -f "${done_file}" ]]; then
  echo "DONE exists at ${done_file}; ${RUN_ID} is finished, nothing to do"
  exit 0
fi

config="${CONFIG:-configs/tinystories_dev.yaml}"
if [[ -n "${CONFIG:-}" && ! -f "${config}" ]]; then
  echo "ERROR: CONFIG ${config} not found" >&2
  exit 1
fi

if [[ ! -f jncm/train_generator.py ]]; then
  echo "ERROR: jncm/train_generator.py not found in ${PWD}; submit from the repo root" >&2
  exit 1
fi
if [[ ! -f "${TOKENIZER}" ]]; then
  echo "ERROR: tokenizer checkpoint ${TOKENIZER} not found; train it first with QUANT=${quant}" >&2
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

ckpt="results/${RUN_ID}/ckpt/latest.pt"

cmd=(
  python -m jncm.train_generator
  --config "${config}"
  --tokenizer "${TOKENIZER}"
  --set "generator.head=${HEAD}"
  --set "tokenizer.quantizer=${quant}"
  --run_id "${RUN_ID}"
)

if [[ -n "${STEPS:-}" ]]; then
  cmd+=(--steps "${STEPS}")
fi
if [[ "${GEN_PPL:-0}" == "1" ]]; then
  cmd+=(--gen_ppl)
fi

if [[ -f "${ckpt}" ]]; then
  echo "Resuming ${RUN_ID} from ${ckpt}"
  cmd+=(--resume "${ckpt}")
else
  echo "No checkpoint at ${ckpt}; starting ${RUN_ID} from scratch"
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}"
