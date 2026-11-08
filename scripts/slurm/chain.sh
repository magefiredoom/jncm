#!/usr/bin/env bash
# Chain several copies of one Slurm template (T4.5).
#
#   scripts/slurm/chain.sh <script> <n_chunks> [--export=ALL,K=V,...]
#       [--wall HH:MM:SS] [--gres=gpu:h100:1] [--array=A-B]
#       [--dependency=<slurm spec>] [--submit]
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
# SBATCH_ACCOUNT must be set; the account is never guessed (SPEC 7).

set -euo pipefail

usage() {
  echo 'usage: scripts/slurm/chain.sh <script> <n_chunks> [--export=ALL,K=V,...]' >&2
  echo '       [--wall HH:MM:SS] [--gres=gpu:h100:1] [--array=A-B]' >&2
  echo '       [--dependency=<slurm spec>] [--submit]' >&2
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
submit=0
positional=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --export=*) export_spec="${1#--export=}" ;;
    --wall=*) wall="${1#--wall=}" ;;
    --gres=*) gres="${1#--gres=}" ;;
    --array=*) array="${1#--array=}" ;;
    --dependency=*) dependency="${1#--dependency=}" ;;
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
  die "SBATCH_ACCOUNT is unset; export SBATCH_ACCOUNT=<allocation> first (never guessed, SPEC 7)"
fi

base_args=(sbatch "--account=${SBATCH_ACCOUNT}" "--gres=${gres}" "--time=${wall}")
if [[ -n "$array" ]]; then
  base_args+=("--array=${array}")
fi
if [[ -n "$export_spec" ]]; then
  base_args+=("--export=${export_spec}")
fi

if ((!submit)); then
  echo "DRY-RUN: ${chunks} chunk(s) of ${script} (--wall ${wall}); nothing submitted (pass --submit)"
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

if ((submit)); then
  echo "CHAIN_JOB_IDS=${job_ids[*]}"
  echo "CHAIN_LAST_ID=${job_ids[$((${#job_ids[@]} - 1))]}"
fi
