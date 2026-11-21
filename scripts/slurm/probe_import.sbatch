#!/usr/bin/env bash
#SBATCH --job-name=jncm_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000M
#SBATCH --time=00:40:00
# CPU-only diagnostic, no GPU and no training (2026-11-20). Findings that motivate it:
#   - canary 4742789 spent about nine minutes in
#     `from transformers import AutoModelForCausalLM, AutoTokenizer` on an H100 node;
#   - probe 4746172 showed the process walking transformers/models/*/ one directory at a
#     time (open fd .../models/lasr/tokenization_lasr.py after 11 minutes, 65 MB RSS,
#     state IN, wchan cl_sync_io_wait), and `module load` plus the venv took 4m51s
#     before its first echo.
# This job times each step in its own bounded process, so a slow one cannot hide the
# rest: module load, venv activation, `import torch`, the import variants that avoid
# the Auto classes, the tokenizers library alone, and the GPT-2 weights load.
# Submit from the repo root:
#   sbatch --account="$SBATCH_ACCOUNT" scripts/slurm/probe_import.sbatch
# Everything is printed to slurm-<jobid>.out (rc=124 means the variant hit its limit).
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ ! -f .venv/bin/activate ]]; then
  echo "ERROR: .venv/bin/activate not found; submit from the repo root" >&2
  exit 1
fi
echo "host: $(hostname)  start: $(date -Is)"
t0=$(date +%s)
module load python/3.12
echo "module load python/3.12: $(( $(date +%s) - t0 ))s"
t0=$(date +%s)
# shellcheck disable=SC1091
source .venv/bin/activate
echo "source .venv/bin/activate: $(( $(date +%s) - t0 ))s"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
GPT2_DIR="${GPT2_LARGE_PATH:-${HOME}/jncm/models/gpt2-large}"
export GPT2_DIR
echo "gpt2: ${GPT2_DIR}"

run_variant() {
  local name="$1" limit="$2" code="$3" t0 rc
  t0=$(date +%s)
  set +e
  timeout "${limit}" python -c "${code}"
  rc=$?
  set -e
  echo "  [${name}] $(( $(date +%s) - t0 ))s rc=${rc} (limit ${limit}s)"
}

echo "== interpreter start"
run_variant "python -c pass" 120 'pass'
echo "== import torch"
run_variant "import torch" 300 'import torch'
echo "== tokenizers library alone (GPT-2 tokenizer.json)"
run_variant "tokenizers.Tokenizer.from_file" 300 \
  'import os; from tokenizers import Tokenizer; t = Tokenizer.from_file(os.environ["GPT2_DIR"] + "/tokenizer.json"); print(len(t.encode("hello world").ids), "ids")'
echo "== transformers, model-specific classes (no Auto classes)"
run_variant "import transformers" 300 'import transformers; print(transformers.__version__)'
run_variant "from transformers import GPT2LMHeadModel, GPT2TokenizerFast" 600 \
  'import sys; from transformers import GPT2LMHeadModel, GPT2TokenizerFast; m = [n for n in sys.modules if n.startswith("transformers.models.")]; print(len(m), "transformers.models.* modules loaded")'
echo "== GPT-2 weights, CPU, model-specific classes"
run_variant "GPT2LMHeadModel.from_pretrained" 900 \
  'import os; from transformers import GPT2LMHeadModel; m = GPT2LMHeadModel.from_pretrained(os.environ["GPT2_DIR"], local_files_only=True); print(sum(p.numel() for p in m.parameters()), "parameters")'
echo "probe done: $(date -Is)"
