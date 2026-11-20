#!/usr/bin/env bash
#SBATCH --job-name=jncm_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --time=00:30:00
# CPU-only diagnostic, no GPU and no training (2026-11-20). Canary 4742789 spent about
# nine minutes inside `from transformers import AutoModelForCausalLM, AutoTokenizer`
# on an H100 node, with the process in Lustre I/O wait while the venv on /home
# (~/jncm/.venv) was read. This job measures, on a cold compute node:
#   1. `import torch` and the transformers import, cold, with -X importtime;
#   2. the same imports again (warm client cache);
#   3. the transformers import from a copy of the package on node-local disk
#      ($SLURM_TMPDIR), including the copy time, which is the obvious mitigation.
# Submit from the repo root:
#   sbatch --account="$SBATCH_ACCOUNT" scripts/slurm/probe_import.sbatch
# Everything is printed to slurm-<jobid>.out.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-${PWD}}"
if [[ ! -f .venv/bin/activate ]]; then
  echo "ERROR: .venv/bin/activate not found; submit from the repo root" >&2
  exit 1
fi
module load python/3.12
# shellcheck disable=SC1091
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "host: $(hostname)  date: $(date -Is)"
site="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "site-packages: ${site}"
python -c 'import sys; print("python", sys.version.split()[0])'

timed_import() {
  python - <<'PY'
import sys
import time

start = time.perf_counter()
import torch  # noqa: F401

mid = time.perf_counter()
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

end = time.perf_counter()
import transformers

models = [name for name in sys.modules if name.startswith("transformers.models.")]
families = {name.split(".")[2] for name in models}
print(
    f"  import torch {mid - start:.1f}s; transformers {transformers.__version__} "
    f"import {end - mid:.1f}s; {len(models)} transformers.models.* modules "
    f"({len(families)} families) loaded",
    flush=True,
)
PY
}

echo "== 1. cold, python -X importtime (transformers part), then timed"
( time python -X importtime -c \
    'from transformers import AutoModelForCausalLM, AutoTokenizer' 2> "${SLURM_TMPDIR:-/tmp}/importtime.txt" ) 2>&1 | grep real || true
echo "  slowest cumulative imports (microseconds):"
sort -t'|' -k2 -n -r "${SLURM_TMPDIR:-/tmp}/importtime.txt" | head -8 | cut -c1-110 || true
echo "== 2. warm, a new interpreter"
timed_import

echo "== 3. transformers copied to node-local disk"
local_pkgs="${SLURM_TMPDIR:-/tmp}/pkgs"
mkdir -p "${local_pkgs}"
( time cp -r "${site}/transformers" "${local_pkgs}/" ) 2>&1 | grep real || true
echo "  copied files: $(find "${local_pkgs}/transformers" -type f | wc -l)"
PYTHONPATH="${local_pkgs}" timed_import
echo "probe done: $(date -Is)"
