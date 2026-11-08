#!/usr/bin/env bash
# Launch one gate phase (T4.5): G0, G1 or G2, as a pilot or as the full run.
#
#   scripts/launch_phase.sh <g0|g1|g2> <pilot|full> [--submit]
#       [--wall HH:MM:SS] [--sec_per_step X] [--ablations]
#
# Dry-run by default (prints every sbatch command, submits nothing).
#
#   g0: tokenizer vq (tok_vq_10k, skipped when results/<run>/DONE exists)
#       -> generator ce (gen_vq_ce_dev) -> eval.
#   g1: tokenizer noisy (tok_noisy_10k, skipped when DONE exists)
#       -> generator flow (gen_noisy_flow_dev) and regress
#          (gen_noisy_regress_dev) in parallel -> eval for both.
#   g2: JEPA base (jepa_base_dev) per RUNBOOK 3A, started from the noisy
#       tokenizer. The six ablations are listed; they are submitted only with
#       --ablations (seven arms in total, base included).
#
# pilot: run ids pilot_*, 500 tokenizer / 300 generator / 200 JEPA steps,
#   --time 00:20:00, GEN_PPL=1 on the generators (decoded samples plus Gen PPL
#   through the offline GPT-2), eval with N_SAMPLES=32, N_SEEDS=1 and a single
#   array task. It never touches a full run id.
#
# full: n_chunks = ceil(steps * sec_per_step / (0.9 * wall)); sec_per_step is
#   the latest matching pilot results/<pilot>/metrics.jsonl value, else
#   --sec_per_step; with neither the phase stops with an error. The default
#   wall is 03:00:00: our fair-share is over-used, so the kit prefers short
#   chunks that chain and resume (scripts/slurm/chain.sh).
#
# After the eval arrays finish, merge each run:
#   python scripts/run_eval.py --run_id <run> --merge
# then judge the gate with python scripts/gate_report.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CHAIN="scripts/slurm/chain.sh"
PHASE=""
MODE=""
SUBMIT=""
WALL="03:00:00"
WALL_GIVEN=0
CLI_SEC=""
ABLATIONS=0
POSITIONAL=()

PILOT_TOK_STEPS=500
PILOT_GEN_STEPS=300
PILOT_JEPA_STEPS=200
PILOT_WALL="00:20:00"
PILOT_N_SAMPLES=32
PILOT_N_SEEDS=1
FULL_TOK_STEPS=10000
FULL_GEN_STEPS=10000
FULL_JEPA_STEPS=10000

JEPA_RUNS=(
  jepa_base_dev
  jepa_no_vicreg_dev
  jepa_no_dec_through_E_dev
  jepa_neither_dev
  jepa_ema_clean_dev
  jepa_regress_dev
  jepa_no_ema_dev
)
JEPA_SETS=(
  ""
  "jepa.vicreg=false"
  "jepa.decoder_through_E=false"
  "jepa.vicreg=false jepa.decoder_through_E=false"
  "jepa.target_chain=ema_clean"
  "generator.head=regress"
  "jepa.ema=false"
)

usage() {
  echo "usage: scripts/launch_phase.sh <g0|g1|g2> <pilot|full> [--submit]" >&2
  echo "       [--wall HH:MM:SS] [--sec_per_step X] [--ablations]" >&2
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --submit) SUBMIT=1 ;;
    --wall=*)
      WALL="${1#--wall=}"
      WALL_GIVEN=1
      ;;
    --sec_per_step=*) CLI_SEC="${1#--sec_per_step=}" ;;
    --ablations) ABLATIONS=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    --*) die "unknown option $1" ;;
    *) POSITIONAL+=("$1") ;;
  esac
  shift
done

if [[ ${#POSITIONAL[@]} -ne 2 ]]; then
  usage
  die "expected <g0|g1|g2> <pilot|full>, got ${#POSITIONAL[@]} argument(s)"
fi
PHASE="${POSITIONAL[0]}"
MODE="${POSITIONAL[1]}"
case "$PHASE" in
  g0 | g1 | g2) ;;
  *) die "phase must be g0, g1 or g2, got '$PHASE'" ;;
esac
case "$MODE" in
  pilot | full) ;;
  *) die "mode must be pilot or full, got '$MODE'" ;;
esac
if [[ -n "$CLI_SEC" && ! "$CLI_SEC" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
  die "--sec_per_step must be a positive number, got '$CLI_SEC'"
fi
if [[ -z "${SBATCH_ACCOUNT:-}" ]]; then
  die "SBATCH_ACCOUNT is unset; export SBATCH_ACCOUNT=<allocation> first (never guessed, SPEC 7)"
fi

if [[ "$MODE" == "pilot" && "$WALL_GIVEN" -eq 0 ]]; then
  WALL="$PILOT_WALL"
fi
if [[ ! "$WALL" =~ ^[0-9]+:[0-9]{2}:[0-9]{2}$ ]]; then
  die "--wall must be HH:MM:SS, got '$WALL'"
fi
WALL_SECONDS=$(awk -F: '{ print $1 * 3600 + $2 * 60 + $3 }' <<< "$WALL")

LAST_ID=""
LAUNCHED=0

pilot_id() {
  # Full run id -> its pilot counterpart (tok ids end in _10k, rest in _dev).
  case "$1" in
    tok_vq_10k) echo "pilot_tok_vq" ;;
    tok_noisy_10k) echo "pilot_tok_noisy" ;;
    *) echo "pilot_${1%_dev}" ;;
  esac
}

pilot_sec_per_step() {
  # Latest sec_per_step of a pilot run, empty when it has not been measured.
  local path="results/$1/metrics.jsonl" value=""
  if [[ -f "$path" ]]; then
    value=$(grep -o '"sec_per_step"[[:space:]]*:[[:space:]]*[0-9.eE+-]*' "$path" \
      | tail -n 1 | sed -E 's/.*:[[:space:]]*//' || true)
  fi
  printf '%s' "$value"
}

CHUNKS=1
compute_chunks() {
  # compute_chunks <run_id> <steps>: sets CHUNKS for the current mode.
  local run="$1" steps="$2" pilot="" sec="" source=""
  CHUNKS=1
  if [[ "$MODE" == "pilot" ]]; then
    return 0
  fi
  pilot="$(pilot_id "$run")"
  sec="$(pilot_sec_per_step "$pilot")"
  source="pilot ${pilot}"
  if [[ -z "$sec" ]]; then
    if [[ -z "$CLI_SEC" ]]; then
      echo "ERROR: no sec_per_step in results/${pilot}/metrics.jsonl for ${run}" >&2
      echo "       and no --sec_per_step; run the pilot first" >&2
      exit 1
    fi
    sec="$CLI_SEC"
    source="--sec_per_step"
  fi
  CHUNKS=$(awk -v steps="$steps" -v sps="$sec" -v wall="$WALL_SECONDS" \
    'BEGIN { raw = steps * sps / (0.9 * wall); n = int(raw); if (n < raw) n += 1; if (n < 1) n = 1; printf "%d", n }')
  echo "  ${run}: ${steps} steps x ${sec} s/step, 0.9 x ${WALL} -> ${CHUNKS} chunk(s) (${source})" >&2
}

start_chain() {
  # start_chain <script> <chunks> <export> [<dependency id>] [extra sbatch args...]
  local script="$1" chunks="$2" export_spec="$3" dep="${4:-}" out=""
  local args=(--export="$export_spec" --wall="$WALL" "${@:5}")
  if [[ -n "$dep" ]]; then
    args+=(--dependency="afterany:${dep}")
  fi
  if [[ -n "$SUBMIT" ]]; then
    args+=(--submit)
  fi
  out="$("$CHAIN" "$script" "$chunks" "${args[@]}")"
  printf '%s\n' "$out"
  LAST_ID="$(printf '%s\n' "$out" | sed -n 's/^CHAIN_LAST_ID=//p' | tail -n 1)"
  LAUNCHED=1
}

dep_after_last() {
  # The last chain's id, or a placeholder in a dry run.
  if [[ -n "$LAST_ID" ]]; then
    printf '%s' "$LAST_ID"
  else
    printf '<LAST>'
  fi
}

launch_tok() {
  # launch_tok <quant> <run_id> <steps>
  local quant="$1" run="$2" steps="$3"
  LAUNCHED=0
  if [[ -f "results/${run}/DONE" ]]; then
    echo "SKIP ${run}: results/${run}/DONE exists"
    LAST_ID=""
    return 0
  fi
  compute_chunks "$run" "$steps"
  start_chain scripts/slurm/tok.sbatch "$CHUNKS" \
    "ALL,QUANT=${quant},STEPS=${steps},RUN_ID=${run}" ""
}

launch_gen() {
  # launch_gen <head> <run_id> <steps> <dependency>
  local head="$1" run="$2" steps="$3" dep="$4" export_spec=""
  LAUNCHED=0
  export_spec="ALL,HEAD=${head},STEPS=${steps},RUN_ID=${run}"
  export_spec+=",TOKENIZER=results/${TOK_RUN}/ckpt/latest.pt"
  if [[ "$MODE" == "pilot" ]]; then
    export_spec+=",GEN_PPL=1"
  fi
  compute_chunks "$run" "$steps"
  start_chain scripts/slurm/gen.sbatch "$CHUNKS" "$export_spec" "$dep"
}

launch_eval() {
  # launch_eval <run_id> <tokenizer or ""> <dependency>
  local run="$1" tokenizer="$2" dep="$3" export_spec="" extra=()
  LAUNCHED=0
  export_spec="ALL,RUN_ID=${run}"
  if [[ -n "$tokenizer" ]]; then
    export_spec+=",TOKENIZER=${tokenizer}"
  fi
  if [[ "$MODE" == "pilot" ]]; then
    export_spec+=",N_SAMPLES=${PILOT_N_SAMPLES},N_SEEDS=${PILOT_N_SEEDS}"
    extra=(--array=0-0)
  fi
  start_chain scripts/slurm/eval.sbatch 1 "$export_spec" "$dep" "${extra[@]}"
}

require_checkpoint() {
  # require_checkpoint <path> <hint>: fail on submit, warn in a dry run.
  if [[ -f "$1" ]]; then
    return 0
  fi
  if [[ -n "$SUBMIT" ]]; then
    die "missing $1; $2"
  fi
  echo "WARNING: $1 not found (dry-run only); the submitted job would fail" >&2
}

run_g0() {
  local tok_dep="" gen_dep=""
  TOK_RUN="tok_vq_10k"
  GEN_RUN="gen_vq_ce_dev"
  TOK_STEPS_N="$FULL_TOK_STEPS"
  GEN_STEPS_N="$FULL_GEN_STEPS"
  if [[ "$MODE" == "pilot" ]]; then
    TOK_RUN="pilot_tok_vq"
    GEN_RUN="pilot_gen_vq_ce"
    TOK_STEPS_N="$PILOT_TOK_STEPS"
    GEN_STEPS_N="$PILOT_GEN_STEPS"
  fi
  echo "== G0 ${MODE}: ${TOK_RUN} (vq tokenizer) -> ${GEN_RUN} (ce generator) -> eval"
  launch_tok vq "$TOK_RUN" "$TOK_STEPS_N"
  if ((LAUNCHED)); then
    tok_dep="$(dep_after_last)"
  fi
  launch_gen ce "$GEN_RUN" "$GEN_STEPS_N" "$tok_dep"
  if ((LAUNCHED)); then
    gen_dep="$(dep_after_last)"
  fi
  launch_eval "$GEN_RUN" "results/${TOK_RUN}/ckpt/latest.pt" "$gen_dep"
}

run_g1() {
  local tok_dep="" flow_dep="" regress_dep=""
  TOK_RUN="tok_noisy_10k"
  GEN_FLOW="gen_noisy_flow_dev"
  GEN_REGRESS="gen_noisy_regress_dev"
  TOK_STEPS_N="$FULL_TOK_STEPS"
  GEN_STEPS_N="$FULL_GEN_STEPS"
  if [[ "$MODE" == "pilot" ]]; then
    TOK_RUN="pilot_tok_noisy"
    GEN_FLOW="pilot_gen_noisy_flow"
    GEN_REGRESS="pilot_gen_noisy_regress"
    TOK_STEPS_N="$PILOT_TOK_STEPS"
    GEN_STEPS_N="$PILOT_GEN_STEPS"
  fi
  echo "== G1 ${MODE}: ${TOK_RUN} (noisy tokenizer) -> ${GEN_FLOW} + ${GEN_REGRESS} -> eval"
  launch_tok noisy "$TOK_RUN" "$TOK_STEPS_N"
  if ((LAUNCHED)); then
    tok_dep="$(dep_after_last)"
  fi
  launch_gen flow "$GEN_FLOW" "$GEN_STEPS_N" "$tok_dep"
  if ((LAUNCHED)); then
    flow_dep="$(dep_after_last)"
  fi
  launch_gen regress "$GEN_REGRESS" "$GEN_STEPS_N" "$tok_dep"
  if ((LAUNCHED)); then
    regress_dep="$(dep_after_last)"
  fi
  launch_eval "$GEN_FLOW" "results/${TOK_RUN}/ckpt/latest.pt" "$flow_dep"
  launch_eval "$GEN_REGRESS" "results/${TOK_RUN}/ckpt/latest.pt" "$regress_dep"
}

run_g2() {
  local i run full_run sets steps dep=""
  if [[ "$MODE" == "pilot" ]]; then
    TOK_RUN="pilot_tok_noisy"
    steps="$PILOT_JEPA_STEPS"
  else
    TOK_RUN="tok_noisy_10k"
    steps="$FULL_JEPA_STEPS"
  fi
  require_checkpoint "results/${TOK_RUN}/ckpt/latest.pt" \
    "run 'scripts/launch_phase.sh g1 ${MODE} --submit' first (it trains ${TOK_RUN})"
  base_label="${JEPA_RUNS[0]}"
  if [[ "$MODE" == "pilot" ]]; then
    base_label="pilot_${JEPA_RUNS[0]%_dev}"
  fi
  echo "== G2 ${MODE}: ${base_label} and its ablations, tokenizer ${TOK_RUN}"
  for ((i = 0; i < ${#JEPA_RUNS[@]}; i++)); do
    full_run="${JEPA_RUNS[$i]}"
    sets="${JEPA_SETS[$i]}"
    if [[ "$MODE" == "pilot" ]]; then
      run="pilot_${full_run%_dev}"
    else
      run="$full_run"
    fi
    if ((i > 0 && !ABLATIONS)); then
      echo "LIST ${run} (SETS=\"${sets}\"); pass --ablations to submit"
      continue
    fi
    export SETS="$sets"
    if [[ -n "$sets" ]]; then
      echo "  SETS=\"${sets}\""
    fi
    compute_chunks "$full_run" "$steps"
    start_chain scripts/slurm/jepa.sbatch "$CHUNKS" \
      "ALL,RUN_ID=${run},STEPS=${steps},TOKENIZER=results/${TOK_RUN}/ckpt/latest.pt" ""
    unset SETS
    dep=""
    if ((LAUNCHED)); then
      dep="$(dep_after_last)"
    fi
    launch_eval "$run" "" "$dep"
  done
}

echo "launch_phase ${PHASE} ${MODE}: wall ${WALL}, account ${SBATCH_ACCOUNT}, $([[ -n "$SUBMIT" ]] && echo submit || echo "dry-run (pass --submit)")"
case "$PHASE" in
  g0) run_g0 ;;
  g1) run_g1 ;;
  g2) run_g2 ;;
esac
echo "Next: when the jobs finish, merge each run (python scripts/run_eval.py --run_id <run> --merge) and run python scripts/gate_report.py"
