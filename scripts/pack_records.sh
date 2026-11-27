#!/usr/bin/env bash
# Pack the small records of runs into results/_records/<UTC date>.tgz plus its .md5 (T4.11, SPEC 7).
#
# On FIR, from the repo root (bash, tar and md5sum only; fine on a login node):
#   bash scripts/pack_records.sh gen_vq_ce_dev gen_noisy_flow_dev ...
# Then locally (a human step, one MFA login) and from the repo root:
#   bash scripts/pull_results.sh _records
#   python scripts/unpack_records.py results/_records/<UTC date>.tgz
#
# Each run contributes config.yaml, git_sha.txt and metrics.jsonl (required: every run writes them),
# and DONE, eval/ and samples/ when present (an absent one is reported, never skipped silently).
# Never ckpt/, never a half-written *.tmp. One pack per UTC day: an existing archive is never
# overwritten. Owners are stored as 0, so the archive carries no username.
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ $# -ge 1 ]] || die "usage: bash scripts/pack_records.sh RUN_ID [RUN_ID ...] (from the repo root)"
[[ -d results ]] || die "run from the repo root (no results/ here)"

REQUIRED=(config.yaml git_sha.txt metrics.jsonl)
shopt -s nullglob globstar

members=()
report=()
seen=" "
for run in "$@"; do
  # Same characters as pull_results.sh, but never a leading '_' or '.' (results/_records is not a run).
  [[ "$run" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "bad run id '$run'"
  [[ "$seen" != *" $run "* ]] || die "run id '$run' given twice"
  seen+="$run "
  [[ -d "results/$run" ]] || die "no run directory results/$run"
  have=()
  for f in "${REQUIRED[@]}"; do
    [[ -f "results/$run/$f" ]] || die "results/$run/$f is missing (every run writes it; AGENTS.md rule 9)"
    members+=("$run/$f")
    have+=("$f")
  done
  absent=()
  if [[ -f "results/$run/DONE" ]]; then
    members+=("$run/DONE")
    have+=(DONE)
  else
    absent+=("DONE (training not finished)")
  fi
  for d in eval samples; do
    if [[ ! -d "results/$run/$d" ]]; then
      absent+=("$d/")
      continue
    fi
    n=0
    for p in "results/$run/$d"/**; do
      [[ -L "$p" ]] && die "$p is a symlink; records are regular files"
      [[ -f "$p" && "$p" != *.tmp ]] || continue
      members+=("${p#results/}")
      n=$((n + 1))
    done
    have+=("$d/ ($n files)")
  done
  line="  $run: ${have[*]}"
  if [[ ${#absent[@]} -gt 0 ]]; then
    line+="; absent: ${absent[*]}"
  fi
  report+=("$line")
done

out_dir="results/_records"
name="$(date -u +%F).tgz"
mkdir -p "$out_dir"
[[ ! -e "$out_dir/$name" ]] || die "$out_dir/$name exists (one pack per UTC day, never overwritten); copy it home first, then move it aside or pack tomorrow"

tmp="$out_dir/$name.tmp"
tar -czf "$tmp" --owner=0 --group=0 --numeric-owner -C results "${members[@]}"
mv "$tmp" "$out_dir/$name"
(cd "$out_dir" && md5sum "$name" > "$name.md5")

echo "packed $# run(s) into $out_dir/$name: ${#members[@]} files, $(wc -c < "$out_dir/$name") bytes (never ckpt/ or *.tmp)"
printf '%s\n' "${report[@]}"
echo "md5: $(cat "$out_dir/$name.md5")"
echo "Next (local, MFA): bash scripts/pull_results.sh _records; then python scripts/unpack_records.py $out_dir/$name"
