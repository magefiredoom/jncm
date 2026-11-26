#!/usr/bin/env bash
# Chain several copies of one Slurm template (T4.5).
#
#   scripts/slurm/chain.sh <script> <n_chunks> [--export=ALL,K=V,...]
#       [--wall=HH:MM:SS] [--gres=gpu:h100:1] [--array=A-B]
#       [--dependency=<slurm spec>] [--merge-after] [--submit]
#
# Dry-run by default: it prints the sbatch commands and submits nothing. With
# --submit, chunk 1 is submitted, chunk i waits for chunk i-1 with
# --dependency=afterany, and the job ids are printed:
#
#   Submitted batch job 1001 (chunk 1/3 of scripts/slurm/tok.sbatch)
#   ...
#   CHAIN_JOB_IDS=1001 1002 1003
#   CHAIN_LAST_ID=1003
#
# Every trainer resumes from results/<RUN_ID>/ckpt/latest.pt and every template
# exits 0 immediately once results/<RUN_ID>/DONE exists, so the surplus chunks
# of an over-estimated chain are cheap no-ops instead of a full wall clock.
#
# --wall defaults to 03:00:00: our fair-share is over-used, so the launch kit
# prefers several short chunks over one long job. --gres defaults to
# gpu:h100:1 because FIR rejects a generic --gres=gpu:1. --array and
# --dependency are for eval arrays and for a chain that must wait for another.
#
# --merge-after (T4.9, SPEC 6.1): after the last chunk, also submit
# scripts/slurm/merge.sbatch with --dependency=afterok:<last chunk id> (for an
# eval array: every task must succeed) and the same --export, which must name
# RUN_ID. The merge is a CPU job: it gets no --gres, --time or --array (its
# template sets 1 CPU, 4G, 00:15:00). Its id is printed as MERGE_JOB_ID;
# CHAIN_LAST_ID stays the last chunk's id.
#
# Storage guard (T4.10, SPEC 7 v1.7): --submit refuses when `realpath results`
# (links followed) lies under JNCM_FORBID_RESULTS_UNDER (default /home; on FIR
# results is a symlink to /scratch). A dry run prints the resolved root and only
# warns. On Git Bash both paths are normalised with cygpath -m (absent on Linux).
#
# SBATCH_ACCOUNT must be set; the account is never guessed (SPEC 7).

set -euo pipefail

resolve_path() {
  # The absolute, link-free form of $1; cygpath -m only where it exists (Git Bash).
  local resolved
  resolved="$(realpath -m -- "$1")"
  if command -v cygpath >/dev/null 2>&1; then
    resolved="$(cygpath -m -- "$resolved")"
  fi
  printf '%s' "$resolved"
}

results_root_forbidden() {
  # results_root_forbidden <root> <prefix>: true if root is the prefix or below it.
  [[ "$1" == "$2" || "$1" == "$2"/* ]]
}

usage() {
  echo 'usage: scripts/slurm/chain.sh <script> <n_chunks> [--export=ALL,K=V,...]' >&2
  echo '       [--wall=HH:MM:SS] [--gres=gpu:h100:1] [--array=A-B]' >&2
  echo '       [--dependency=<slurm spec>] [--merge-after] [--submit]' >&2
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

script=""
chunks=""
export_spec=""
wall="03:00:00"
gres="gpu:h100:1"
array=""
dependency=""
merge_after=0
merge_script="scripts/slurm/merge.sbatch"
submit=0
positional=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --export=*) export_spec="${1#--export=}" ;;
    --wall=*) wall="${1#--wall=}" ;;
    --gres=*) gres="${1#--gres=}" ;;
    --array=*) array="${1#--array=}" ;;
    --dependency=*) dependency="${1#--dependency=}" ;;
    --merge-after) merge_after=1 ;;
    --submit) submit=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    --*) die "unknown option $1" ;;
    *) positional+=("$1") ;;
  esac
  shift
done

if [[ ${#positional[@]} -ne 2 ]]; then
  usage
  die "expected <script> and <n_chunks>, got ${#positional[@]} argument(s)"
fi
script="${positional[0]}"
chunks="${positional[1]}"

if [[ ! -f "$script" ]]; then
  die "Slurm template not found: $script (run from the repo root)"
fi
if [[ ! "$chunks" =~ ^[0-9]+$ ]]; then
  die "n_chunks must be a positive integer, got '$chunks'"
fi
if ((chunks < 1)); then
  die "n_chunks must be >= 1, got $chunks"
fi
if [[ ! "$wall" =~ ^[0-9]+:[0-9]{2}:[0-9]{2}$ ]]; then
  die "--wall must be HH:MM:SS, got '$wall'"
fi
if [[ -z "${SBATCH_ACCOUNT:-}" ]]; then
  die "SBATCH_ACCOUNT is unset; run 'source site.env' first (copy site.env.example; never guessed, SPEC 7)"
fi
if ((merge_after)); then
  if [[ ! ",${export_spec}," =~ ,RUN_ID=[^,]+, ]]; then
    die "--merge-after needs RUN_ID in --export (the merge job merges results/<RUN_ID>/eval)"
  fi
  if [[ ! -f "$merge_script" ]]; then
    die "merge template not found: $merge_script (run from the repo root)"
  fi
fi

# T4.10: the storage guard, before anything is submitted.
forbid_raw="${JNCM_FORBID_RESULTS_UNDER:-/home}"
forbid_root="$(resolve_path "$forbid_raw")"
results_root="$(resolve_path results)"
if results_root_forbidden "$results_root" "$forbid_root"; then
  if ((submit)); then
    die "results resolves to ${results_root}, under ${forbid_raw} (JNCM_FORBID_RESULTS_UNDER); SPEC 7: run outputs live on scratch, with results a symlink there (RUNBOOK)"
  fi
  echo "WARNING: results resolves to ${results_root}, under ${forbid_raw} (JNCM_FORBID_RESULTS_UNDER); --submit would be refused (SPEC 7)" >&2
fi

base_args=(sbatch "--account=${SBATCH_ACCOUNT}" "--gres=${gres}" "--time=${wall}")
if [[ -n "$array" ]]; then
  base_args+=("--array=${array}")
fi
if [[ -n "$export_spec" ]]; then
  base_args+=("--export=${export_spec}")
fi

if ((!submit)); then
  echo "DRY-RUN: ${chunks} chunk(s) of ${script} (--wall ${wall}); nothing submitted (pass --submit); results root ${results_root}, refused under ${forbid_raw}"
fi

prev=""
job_ids=()
for ((i = 1; i <= chunks; i++)); do
  if ((i > 1)); then
    if ((submit)); then
      if [[ -n "$dependency" ]]; then
        deps="${dependency},afterany:${prev}"
      else
        deps="afterany:${prev}"
      fi
    else
      if [[ -n "$dependency" ]]; then
        deps="${dependency},afterany:<JOB_$((i - 1))>"
      else
        deps="afterany:<JOB_$((i - 1))>"
      fi
    fi
  else
    deps="$dependency"
  fi

  cmd=("${base_args[@]}")
  if [[ -n "$deps" ]]; then
    cmd+=("--dependency=${deps}")
  fi
  cmd+=("$script")

  if ((submit)); then
    out=$("${cmd[@]}")
    job_id="${out##* }"
    if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
      die "could not read a job id from sbatch output: $out"
    fi
    job_ids+=("$job_id")
    prev="$job_id"
    echo "Submitted batch job ${job_id} (chunk ${i}/${chunks} of ${script})"
  else
    printf '%s\n' "${cmd[*]}"
  fi
done

if ((merge_after)); then
  # T4.9: the CPU merge waits for the whole last chunk (all array tasks) to succeed.
  merge_cmd=(sbatch "--account=${SBATCH_ACCOUNT}")
  if ((submit)); then
    merge_cmd+=("--dependency=afterok:${prev}")
  else
    merge_cmd+=("--dependency=afterok:<JOB_${chunks}>")
  fi
  if [[ -n "$export_spec" ]]; then
    merge_cmd+=("--export=${export_spec}")
  fi
  merge_cmd+=("$merge_script")
  if ((submit)); then
    out=$("${merge_cmd[@]}")
    merge_id="${out##* }"
    if [[ ! "$merge_id" =~ ^[0-9]+$ ]]; then
      die "could not read a job id from sbatch output: $out"
    fi
    echo "Submitted batch job ${merge_id} (merge after ${prev})"
  else
    printf '%s\n' "${merge_cmd[*]}"
  fi
fi

if ((submit)); then
  echo "CHAIN_JOB_IDS=${job_ids[*]}"
  echo "CHAIN_LAST_ID=${job_ids[$((${#job_ids[@]} - 1))]}"
  if ((merge_after)); then
    echo "MERGE_JOB_ID=${merge_id}"
  fi
fi
