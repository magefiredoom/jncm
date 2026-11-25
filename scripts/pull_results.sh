#!/usr/bin/env bash
# Copy run results from FIR's scratch into the local results/ (a human step: one MFA login).
#
# From the repo root, in WSL (Ubuntu) or any shell with rsync and ssh:
#   source site.env                        # sets FIR_USER (see site.env.example)
#   bash scripts/pull_results.sh                # every run on FIR
#   bash scripts/pull_results.sh --skip-running # leave out the runs whose training chain is not finished
#   bash scripts/pull_results.sh RUN_ID ...     # only these runs
#   bash scripts/pull_results.sh --print ...    # print the rsync command, run nothing
#
# Copies only: nothing on FIR is deleted (jobs still read the tokenizers and milestones), nothing local
# is deleted (rsync without --delete), and half-written checkpoints (*.tmp) are skipped. Resumable:
# run it again after an interruption. -rt keeps times but not owners or permissions, which an exFAT
# drive mounted in WSL cannot store.
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

PRINT=""
SKIP_RUNNING=""
RUNS=()
for arg in "$@"; do
  case "$arg" in
    --print) PRINT=1 ;;
    --skip-running) SKIP_RUNNING=1 ;;
    -*) die "unknown option '$arg'" ;;
    *) RUNS+=("$arg") ;;
  esac
done

[[ -n "${FIR_USER:-}" ]] || die "FIR_USER is unset; run 'source site.env' first (copy site.env.example)"
[[ -d results ]] || die "run from the repo root (no results/ here)"

host="${FIR_SSH:-${FIR_USER}@fir.alliancecan.ca}"
remote_root="${FIR_RESULTS:-/scratch/${FIR_USER}/jncm/results}"

# The main-study chains that are still training when this was written (STATE "FIR jobs");
# --skip-running leaves them out until their DONE exists.
RUNNING=(gen_vq_ce_dev gen_noisy_flow_dev gen_noisy_regress_dev jepa_base_dev)

cmd=(rsync -rt --partial --info=progress2 --modify-window=2 --exclude='*.tmp')
if [[ -n "$SKIP_RUNNING" ]]; then
  for run in "${RUNNING[@]}"; do
    cmd+=("--exclude=/${run}/")
  done
fi

if [[ ${#RUNS[@]} -eq 0 ]]; then
  cmd+=("${host}:${remote_root}/" "results/")
else
  for run in "${RUNS[@]}"; do
    [[ "$run" =~ ^[A-Za-z0-9_.-]+$ ]] || die "bad run id '$run'"
    cmd+=("${host}:${remote_root}/${run}")
  done
  cmd+=("results/")
fi

if [[ -n "$PRINT" ]]; then
  printf '%q ' "${cmd[@]}"
  echo
  exit 0
fi
exec "${cmd[@]}"
