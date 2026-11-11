#!/usr/bin/env bash
# Launch one gate phase (T4.5): G0, G1 or G2, as a pilot or as the full run.
#
#   scripts/launch_phase.sh <g0|g1|g2> <pilot|full> [--submit]
#       [--wall HH:MM:SS] [--sec_per_step X] [--ablations]
#       [--config PATH]
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
# --config selects the YAML passed to every template through CONFIG (default:
#   each template's dev config). The lite configs are
#   configs/tinystories_lite.yaml (g0/g1) and configs/jepa_lite.yaml (g2).
#   A non-dev config also suffixes every generator/JEPA run id this kit builds
#   with the config stem (configs/*_lite.yaml -> _lite): gen_vq_ce_dev becomes
#   gen_vq_ce_lite and pilot_gen_vq_ce becomes pilot_gen_vq_ce_lite, so dev and
#   lite never share a run id, a checkpoint directory or a DONE file. Tokenizer
#   run ids are NEVER suffixed (T5.6): the tokenizer sections of the dev and
#   lite configs are identical, so tok_vq_10k / tok_noisy_10k are shared.
#
# pilot: run ids pilot_*, 500 tokenizer / 300 generator / 200 JEPA steps,
#   --time 00:20:00, GEN_PPL=1 on the generators (decoded samples plus Gen PPL
#   through the offline GPT-2), eval with N_SAMPLES=32, N_SEEDS=1 and a single
#   array task. It never touches a full run id. T5.6: a pilot reuses the real
#   tokenizer (results/tok_vq_10k, results/tok_noisy_10k) whenever its DONE
#   file exists, and submits the pilot tokenizer only when it is missing.
#
# Every run ends with a plan line: the number of jobs, the total requested
#   GPU-hours (chunks x --time x 1 GPU), the micro-batch size of the config in
#   play and the submission order.
#
# full: n_chunks = ceil(steps * sec_per_step / (0.9 * wall)); sec_per_step is
#   the latest matching pilot results/<pilot>/metrics.jsonl value, else
#   --sec_per_step; with neither the phase stops with an error. The default
#   wall is 03:00:00: our fair-share is over-used, so the kit prefers short
#   chunks that chain and resume (scripts/slurm/chain.sh). Tokenizer submission
#   is skipped when results/tok_*_10k/DONE exists, for either config.
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
CONFIG=""
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

# T5.6: the tokenizers are shared by the dev and lite configs, so their run ids
# are fixed and never carry the config suffix.
TOK_VQ_RUN="tok_vq_10k"
TOK_NOISY_RUN="tok_noisy_10k"
PILOT_TOK_VQ_RUN="pilot_tok_vq"
PILOT_TOK_NOISY_RUN="pilot_tok_noisy"

# T5.6: the plan summary (jobs, GPU-hours, micro-batch, submission order).
PLAN_CONFIG="configs/tinystories_dev.yaml"
PLAN_ORDER=()

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
  echo "       [--config PATH]" >&2
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
    --config=*) CONFIG="${1#--config=}" ;;
    --config)
      [[ $# -ge 2 ]] || die "--config needs a path"
      CONFIG="$2"
      shift
      ;;
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

# T5.5c: a non-dev CONFIG gives every run id the config stem as a suffix.
SUFFIX=""
if [[ -n "$CONFIG" ]]; then
  STEM="$(basename "$CONFIG")"
  STEM="${STEM%.yaml}"
  STEM="${STEM%.yml}"
  if [[ "$STEM" != *_dev ]]; then
    STEM="${STEM#tinystories_}"
    STEM="${STEM#jepa_}"
    SUFFIX="$STEM"
  fi
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

rid() {
  # rid <dev run id>: the config suffix replaces a trailing _dev, else appends.
  if [[ -z "$SUFFIX" ]]; then
    printf '%s' "$1"
  elif [[ "$1" == *_dev ]]; then
    printf '%s_%s' "${1%_dev}" "$SUFFIX"
  else
    printf '%s_%s' "$1" "$SUFFIX"
  fi
}

base_id() {
  # base_id <run id>: the dev-shaped id, without the config suffix.
  if [[ -n "$SUFFIX" && "$1" == *"_${SUFFIX}" ]]; then
    printf '%s' "${1%"_${SUFFIX}"}"
  else
    printf '%s' "$1"
  fi
}

pilot_id() {
  # Dev run id -> its pilot counterpart (tok ids end in _10k, rest in _dev).
  # Only generator/JEPA ids carry the config suffix; tokenizer ids do not.
  case "$1" in
    "$TOK_VQ_RUN") printf '%s' "$PILOT_TOK_VQ_RUN" ;;
    "$TOK_NOISY_RUN") printf '%s' "$PILOT_TOK_NOISY_RUN" ;;
    *) rid "pilot_${1%_dev}" ;;
  esac
}

select_tokenizer() {
  # select_tokenizer <real_id> <pilot_id>: TOK_RUN is the finished real
  # tokenizer when its DONE file exists, else the pilot id (T5.6).
  if [[ -f "results/$1/DONE" ]]; then
    TOK_RUN="$1"
  else
    TOK_RUN="$2"
  fi
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
  pilot="$(pilot_id "$(base_id "$run")")"
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
  local script="$1" chunks="$2" export_spec="$3" dep="${4:-}" out="" run_label=""
  local args=(--export="$export_spec" --wall="$WALL" "${@:5}")
  # T5.6: the plan summary counts jobs and GPU-hours in submission order.
  run_label="$(printf '%s' "$export_spec" | sed -n 's/.*RUN_ID=\([^,]*\).*/\1/p')"
  PLAN_ORDER+=("${script}|${run_label}|${chunks}")
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

micro_batch_line() {
  # micro_batch_line <config>: one line with the generator.train micro-batch.
  local cfg="$1" batch="" accum="" line=""
  if [[ ! -f "$cfg" ]]; then
    printf 'micro-batch unknown (config %s not found)' "$cfg"
    return 0
  fi
  line=$(awk '
    /^[A-Za-z_][A-Za-z0-9_]*:/ { sec = $1; sub(/:$/, "", sec); blk = "" }
    sec == "generator" && /^  [A-Za-z_][A-Za-z0-9_]*:/ { blk = $1; sub(/:$/, "", blk) }
    sec == "generator" && blk == "train" && $1 == "batch_size:" { b = $2 }
    sec == "generator" && blk == "train" && $1 == "grad_accum:" { g = $2 }
    END { if (b != "" && g != "") print b, g }
  ' "$cfg")
  if [[ -z "$line" ]]; then
    printf 'micro-batch unknown (no generator.train batch_size/grad_accum in %s)' "$cfg"
    return 0
  fi
  read -r batch accum <<< "$line"
  if ((accum < 1 || batch % accum != 0)); then
    printf 'micro-batch INVALID (batch %s / grad_accum %s is not an integer in %s)' \
      "$batch" "$accum" "$cfg"
    return 0
  fi
  printf 'micro-batch %d (batch %s / grad_accum %s, %s)' \
    "$((batch / accum))" "$batch" "$accum" "$cfg"
}

print_plan() {
  # T5.6: jobs, requested GPU-hours, micro-batch size and submission order.
  local jobs=0 entry script run_label chunks hours="" i=0
  for entry in "${PLAN_ORDER[@]:-}"; do
    [[ -z "$entry" ]] && continue
    IFS='|' read -r script run_label chunks <<< "$entry"
    jobs=$((jobs + chunks))
  done
  hours=$(awk -v j="$jobs" -v w="$WALL_SECONDS" 'BEGIN { printf "%.2f", j * w / 3600 }')
  echo "== plan: ${jobs} job(s), ${hours} requested GPU-hour(s) (1 GPU x ${WALL} each)"
  echo "   $(micro_batch_line "$PLAN_CONFIG")"
  if ((jobs > 0)); then
    echo "   submission order:"
    for entry in "${PLAN_ORDER[@]:-}"; do
      [[ -z "$entry" ]] && continue
      IFS='|' read -r script run_label chunks <<< "$entry"
      i=$((i + 1))
      printf '     %d. %s %s (%s chunk(s))\n' "$i" "$script" "$run_label" "$chunks"
    done
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
  export_spec="ALL,QUANT=${quant},STEPS=${steps},RUN_ID=${run}"
  if [[ -n "$CONFIG" ]]; then
    export_spec+=",CONFIG=${CONFIG}"
  fi
  start_chain scripts/slurm/tok.sbatch "$CHUNKS" "$export_spec" ""
}

launch_gen() {
  # launch_gen <head> <run_id> <steps> <dependency>
  local head="$1" run="$2" steps="$3" dep="$4" export_spec=""
  LAUNCHED=0
  export_spec="ALL,HEAD=${head},STEPS=${steps},RUN_ID=${run}"
  export_spec+=",TOKENIZER=results/${TOK_RUN}/ckpt/latest.pt"
  if [[ -n "$CONFIG" ]]; then
    export_spec+=",CONFIG=${CONFIG}"
  fi
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
  if [[ -n "$CONFIG" ]]; then
    export_spec+=",CONFIG=${CONFIG}"
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
  LAUNCHED=0
  GEN_RUN="$(rid gen_vq_ce_dev)"
  TOK_STEPS_N="$FULL_TOK_STEPS"
  GEN_STEPS_N="$FULL_GEN_STEPS"
  if [[ "$MODE" == "pilot" ]]; then
    GEN_RUN="$(rid pilot_gen_vq_ce)"
    TOK_STEPS_N="$PILOT_TOK_STEPS"
    GEN_STEPS_N="$PILOT_GEN_STEPS"
    select_tokenizer "$TOK_VQ_RUN" "$PILOT_TOK_VQ_RUN"
  else
    TOK_RUN="$TOK_VQ_RUN"
  fi
  echo "== G0 ${MODE}: ${TOK_RUN} (vq tokenizer) -> ${GEN_RUN} (ce generator) -> eval"
  if [[ "$MODE" == "full" || "$TOK_RUN" != "$TOK_VQ_RUN" ]]; then
    launch_tok vq "$TOK_RUN" "$TOK_STEPS_N"
  else
    echo "REUSE ${TOK_RUN}: results/${TOK_RUN}/DONE exists, no pilot tokenizer"
  fi
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
  LAUNCHED=0
  GEN_FLOW="$(rid gen_noisy_flow_dev)"
  GEN_REGRESS="$(rid gen_noisy_regress_dev)"
  TOK_STEPS_N="$FULL_TOK_STEPS"
  GEN_STEPS_N="$FULL_GEN_STEPS"
  if [[ "$MODE" == "pilot" ]]; then
    GEN_FLOW="$(rid pilot_gen_noisy_flow)"
    GEN_REGRESS="$(rid pilot_gen_noisy_regress)"
    TOK_STEPS_N="$PILOT_TOK_STEPS"
    GEN_STEPS_N="$PILOT_GEN_STEPS"
    select_tokenizer "$TOK_NOISY_RUN" "$PILOT_TOK_NOISY_RUN"
  else
    TOK_RUN="$TOK_NOISY_RUN"
  fi
  echo "== G1 ${MODE}: ${TOK_RUN} (noisy tokenizer) -> ${GEN_FLOW} + ${GEN_REGRESS} -> eval"
  if [[ "$MODE" == "full" || "$TOK_RUN" != "$TOK_NOISY_RUN" ]]; then
    launch_tok noisy "$TOK_RUN" "$TOK_STEPS_N"
  else
    echo "REUSE ${TOK_RUN}: results/${TOK_RUN}/DONE exists, no pilot tokenizer"
  fi
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
    select_tokenizer "$TOK_NOISY_RUN" "$PILOT_TOK_NOISY_RUN"
    if [[ "$TOK_RUN" == "$TOK_NOISY_RUN" ]]; then
      echo "REUSE ${TOK_RUN}: results/${TOK_RUN}/DONE exists, no pilot tokenizer"
    fi
    steps="$PILOT_JEPA_STEPS"
  else
    TOK_RUN="$TOK_NOISY_RUN"
    steps="$FULL_JEPA_STEPS"
  fi
  require_checkpoint "results/${TOK_RUN}/ckpt/latest.pt" \
    "run 'scripts/launch_phase.sh g1 ${MODE} --submit' first (it trains ${TOK_RUN})"
  base_label="${JEPA_RUNS[0]}"
  if [[ "$MODE" == "pilot" ]]; then
    base_label="pilot_${JEPA_RUNS[0]%_dev}"
  fi
  base_label="$(rid "$base_label")"
  echo "== G2 ${MODE}: ${base_label} and its ablations, tokenizer ${TOK_RUN}"
  for ((i = 0; i < ${#JEPA_RUNS[@]}; i++)); do
    full_run="${JEPA_RUNS[$i]}"
    sets="${JEPA_SETS[$i]}"
    if [[ "$MODE" == "pilot" ]]; then
      run="$(rid "pilot_${full_run%_dev}")"
    else
      run="$(rid "$full_run")"
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
    export_spec="ALL,RUN_ID=${run},STEPS=${steps},TOKENIZER=results/${TOK_RUN}/ckpt/latest.pt"
    if [[ -n "$CONFIG" ]]; then
      export_spec+=",CONFIG=${CONFIG}"
    fi
    start_chain scripts/slurm/jepa.sbatch "$CHUNKS" "$export_spec" ""
    unset SETS
    dep=""
    if ((LAUNCHED)); then
      dep="$(dep_after_last)"
    fi
    launch_eval "$run" "" "$dep"
  done
}

if [[ "$PHASE" == "g2" ]]; then
  PLAN_CONFIG="configs/jepa_dev.yaml"
fi
PLAN_CONFIG="${CONFIG:-$PLAN_CONFIG}"

echo "launch_phase ${PHASE} ${MODE}: wall ${WALL}, config ${CONFIG:-template default}, run-id suffix ${SUFFIX:-none}, account ${SBATCH_ACCOUNT}, $([[ -n "$SUBMIT" ]] && echo submit || echo "dry-run (pass --submit)")"
case "$PHASE" in
  g0) run_g0 ;;
  g1) run_g1 ;;
  g2) run_g2 ;;
esac
print_plan
echo "Next: when the jobs finish, merge each run (python scripts/run_eval.py --run_id <run> --merge) and run python scripts/gate_report.py"
