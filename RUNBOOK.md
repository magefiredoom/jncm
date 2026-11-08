# RUNBOOK — jncm on FIR

Operator commands for Gate G0a (tokenizers) and the Phase 0 / Phase 1 / Phase 2
runs. Every wall clock is written `<WALL_CLOCK>` (`HH:MM:SS`); nothing else is
blank.

Fixed facts (from docs/STATE.md; do not re-derive):

- Host `aefad@fir.alliancecan.ca`, login1/login2, repo `~/jncm`, venv `.venv`
  (built with `module load python/3.12`).
- Slurm account `$SBATCH_ACCOUNT` (`def-cheng23`; GPU jobs record
  `def-cheng23_gpu`). No `--partition`: FIR places H100 jobs on `gpubase_b`.
- GPU request `--gres=gpu:h100:1` (FIR rejects a plain `--gres=gpu:1`).
- Compute nodes are offline; `tok/gen/eval` set the `HF_*_OFFLINE` variables.
- Run every block from `~/jncm` unless the block starts with `cd` somewhere.

At the top of each FIR shell:

```bash
cd ~/jncm
export SBATCH_ACCOUNT=def-cheng23     # only if your shell does not set it already
```

## 0. Pilots first — go/no-go before any full run

Our fair-share is over-used, so a 10,000-step run submitted whole waits too
long. The launch kit submits short chunks that chain and resume: every trainer
writes `results/<run_id>/DONE` when its final step completes and records
`sec_per_step` in every metrics record; `scripts/slurm/chain.sh` submits
`ceil(steps * sec_per_step / (0.9 * wall))` copies with
`--dependency=afterany` (each resumes from `ckpt/latest.pt`, surplus chunks
exit 0 on `DONE`); `scripts/launch_phase.sh` builds a whole phase from that.

Run the pilot of a phase first and clear its checklist before the full run.

```bash
cd ~/jncm
scripts/launch_phase.sh g0 pilot            # dry run: prints the exact plan
scripts/launch_phase.sh g0 pilot --submit   # pilot_tok_vq (500 steps) -> pilot_gen_vq_ce (300) -> eval
```

The pilot is 500 tokenizer / 300 generator / 200 JEPA steps behind
`--time 00:20:00`, uses run ids `pilot_*`, sets `GEN_PPL=1` on the generators
and evaluates with `N_SAMPLES=32`, `N_SEEDS=1`. It never writes a full run id,
so it cannot poison `tok_vq_10k`, `gen_vq_ce_dev`, .... A pilot that cannot
finish inside the 20-minute wall is itself a stop signal: fix the speed before
the full run.

| Phase | Pilot run ids | Eval |
| --- | --- | --- |
| g0 | `pilot_tok_vq`, `pilot_gen_vq_ce` | `pilot_gen_vq_ce` |
| g1 | `pilot_tok_noisy`, `pilot_gen_noisy_flow`, `pilot_gen_noisy_regress` | both generators |
| g2 | `pilot_jepa_base` (plus six `pilot_jepa_*` ablations with `--ablations`) | every submitted arm |

### 0.1 Go/no-go checklist (per pilot)

Every line must hold for every pilot of the phase before its full run:

| Check | Where to look | Pass |
| --- | --- | --- |
| No crash | `sacct -j <ids> --format=JobID,State,Elapsed,ExitCode` | all `COMPLETED`, `ExitCode 0:0` |
| Real data | last `metrics.jsonl` record | `data_source` is `real` |
| Finite losses | last `metrics.jsonl` record | `train_loss` finite (g2 also `loss_dec`, `loss_pred`, `loss_vic`); no NaN/inf |
| Gen PPL hook ran offline | `results/pilot_gen_*/metrics.jsonl` | `gen_ppl` present; the job runs under `TRANSFORMERS_OFFLINE=1` against `models/gpt2-large` (the directory `GPT2_LARGE_PATH` points at), so a missing model fails the job instead of downloading |
| Samples decode to text | `head -n 3 results/pilot_gen_*/samples/step_*.txt` | readable English, not token ids |
| `pairwise_agree` present | last record of the flow AND regress pilots | both carry `pairwise_agree` (and `unique_frac`); `pairwise_agree > 0.5` is a collapse flag |
| `sec_per_step` recorded | every `metrics.jsonl` record | a positive `sec_per_step`; it is what sizes the full chain |

Paste back (well under 30 lines per run):

```bash
cd ~/jncm
python - <<'PY'
import glob, json, math, os

for run in sorted(glob.glob("results/pilot_*")):
    run_id = os.path.basename(run)
    path = f"{run}/metrics.jsonl"
    if not os.path.exists(path):
        print(f"== {run_id}: no metrics.jsonl (not started, or failed early)")
        continue
    rows = [json.loads(line) for line in open(path) if line.strip()]
    last = rows[-1]
    finite = all(math.isfinite(float(v)) for k, v in last.items()
                 if isinstance(v, (int, float)) and k != "step")
    samples = sorted(glob.glob(f"{run}/samples/step_*.txt"))
    print(f"== {run_id}: step {last['step']} data_source {last.get('data_source')} "
          f"loss {last.get('train_loss'):.4f} finite {finite} "
          f"sec_per_step {last.get('sec_per_step')}")
    print(f"   gen_ppl {last.get('gen_ppl', 'n/a')} "
          f"unique_frac {last.get('unique_frac', 'n/a')} "
          f"pairwise_agree {last.get('pairwise_agree', 'n/a')} samples {len(samples)}")
    if samples:
        print("   " + open(samples[-1]).readline().strip()[:90])
PY
```

`gen_ppl` is `n/a` for tokenizer and JEPA pilots by design (a JEPA run's Gen
PPL belongs to its eval array, not the trainer). For g2 also require
`collapse_alarm: false` on the last record.

### 0.2 Full runs (same launcher)

`n_chunks = ceil(steps * sec_per_step / (0.9 * wall))`, with `sec_per_step`
from the latest matching pilot record and `--wall` defaulting to `03:00:00`.
A tokenizer run that already has `DONE` is skipped. Without a pilot record and
without `--sec_per_step` the launcher stops with an error.

```bash
cd ~/jncm
scripts/launch_phase.sh g0 full                       # dry run: the exact chain
scripts/launch_phase.sh g0 full --submit              # tok_vq_10k -> gen_vq_ce_dev -> eval
scripts/launch_phase.sh g1 full --submit              # tok_noisy_10k -> flow + regress -> eval both
scripts/launch_phase.sh g2 full --submit              # jepa_base_dev (+ eval)
scripts/launch_phase.sh g2 full --submit --ablations  # base + six ablations, each with eval
python scripts/budget.py                              # projected H100-hours; flags any run > 12 h
```

`--wall=HH:MM:SS` overrides the 3 h default; `--sec_per_step=X` supplies the
speed when a pilot record is missing. Nothing is submitted without `--submit`;
with it every chain prints its job ids and `CHAIN_LAST_ID`. Resubmitting the
same command is always safe: chains resume from `ckpt/latest.pt` and finished
chunks no-op on `DONE`. Then merge and judge the gate as in §2.2, §3, §3A.3
and §3B.

## 1. Syncing FIR

### 1.1 Code: private git remote (preferred) — this is how code updates work

The FIR checkout must be a real git repository, so `git pull` updates the code
and `results/<run_id>/git_sha.txt` records a real SHA (STATE known issue 11).

Local, once:

```bash
git remote add fir <PRIVATE_REMOTE_URL>
git push fir HEAD:main
git rev-parse HEAD
```

FIR, once (the current `~/jncm` was extracted from a tar and is not a repo; it
is kept as `jncm_tar_old`, nothing is deleted):

```bash
cd ~
mv jncm jncm_tar_old
git clone <PRIVATE_REMOTE_URL> jncm
mkdir -p jncm/data jncm/models
rsync -a jncm_tar_old/data/ jncm/data/
rsync -a jncm_tar_old/models/ jncm/models/
rsync -a jncm_tar_old/results/ jncm/results/
```

Confirm both trees are at the same commit — the two SHAs must be identical:

```bash
# local (repo root)
git rev-parse HEAD
# FIR
cd ~/jncm && git rev-parse HEAD && git status --short
```

`git status --short` must print nothing: a dirty tree appends `-dirty` to
`git_sha.txt`. Code update loop after that:

```bash
# local
git push fir HEAD:main
# FIR
cd ~/jncm && git pull --ff-only
```

If `~/jncm` is already a checkout, skip the clone and only run the
`git pull --ff-only` step.

### 1.2 Data and models (human step, MFA prompt)

`rsync` (resumable; run it inside `tmux` because each connection asks for MFA):

```bash
# local, repo root
rsync -avhP --partial data/tinystories/ aefad@fir.alliancecan.ca:~/jncm/data/tinystories/
rsync -avhP --partial models/gpt2-large/ aefad@fir.alliancecan.ca:~/jncm/models/gpt2-large/
```

`scp` equivalent:

```bash
scp data/tinystories/train.bin data/tinystories/val.bin \
    data/tinystories/meta.json data/tinystories/tokenizer.json \
    aefad@fir.alliancecan.ca:~/jncm/data/tinystories/
scp -r models/gpt2-large aefad@fir.alliancecan.ca:~/jncm/models/
```

Verify the copy (expect `933740544`, `9384448`, and the meta below; the GPT-2
directory must contain `config.json`, `pytorch_model.bin`, `tokenizer.json`,
`vocab.json`, `merges.txt`):

```bash
ssh aefad@fir.alliancecan.ca 'cd ~/jncm && wc -c data/tinystories/train.bin data/tinystories/val.bin && cat data/tinystories/meta.json && ls models/gpt2-large'
```

Expected meta: `train_windows 3647424`, `val_windows 36658`, `L 128`, `V 8192`.

## 2. Phase 0 — tokenizers (G0a), then vq + ce (G0)

### 2.1 G0a recap: the two 10,000-step tokenizer runs

Already submitted: `tok_vq_10k` = job 2880034, `tok_noisy_10k` = job 2880035,
`STEPS=10000`, `--time=04:00:00`, PENDING at submit. Both are the budget-limited
pair the gate is judged on (DECISIONS 5); SPEC 4.4 and the thresholds are
unchanged.

| Job ID | Run ID | STEPS | Time limit | State |
| --- | --- | --- | --- | --- |
| 2880034 | `tok_vq_10k` | 10,000 | 04:00:00 | PENDING at submit |
| 2880035 | `tok_noisy_10k` | 10,000 | 04:00:00 | PENDING at submit |

Status:

```bash
squeue -u "$USER" -n jncm_tok -o "%.8i %.10j %.8T %.10M %.12l"
sacct -j 2880034,2880035 --format=JobID%12,JobName%12,State,Elapsed,Timelimit,ExitCode
```

Resubmit the identical command only if a job was preempted or timed out; it
resumes from `results/<run_id>/ckpt/latest.pt` when that file exists:

```bash
cd ~/jncm
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,QUANT=vq,STEPS=10000,RUN_ID=tok_vq_10k scripts/slurm/tok.sbatch
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,QUANT=noisy,STEPS=10000,RUN_ID=tok_noisy_10k scripts/slurm/tok.sbatch
```

Check `ls results/<run_id>/ckpt/latest.pt` before assuming a resume works: the
trainer writes a checkpoint every `train.save_every` steps (5,000 in the dev
config), so a job killed before step 5,000 restarts from scratch. If `sacct`
shows `TIMEOUT` below step 5,000, resubmit with a larger `--time=<WALL_CLOCK>`
instead.

G0a verdict: vq `acc_full >= 0.999`, noisy `>= 0.995`, on a final record with
`data_source = real` (SPEC 4.4). `scripts/g0a_report.py` is keyed to the
`tok_vq_dev` / `tok_noisy_dev` run ids, so use the record dump in §4.2 for the
10k pair.

### 2.2 G0: vq + ce

Requires `results/tok_vq_10k/ckpt/latest.pt` to exist. `STEPS` is unset, so the
generator uses the config budget (10,000 steps, SPEC 5.7).

```bash
cd ~/jncm
ls -l results/tok_vq_10k/ckpt/latest.pt
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,HEAD=ce,RUN_ID=gen_vq_ce_dev,TOKENIZER=results/tok_vq_10k/ckpt/latest.pt \
  scripts/slurm/gen.sbatch
```

Monitor:

```bash
squeue -u "$USER" -n jncm_gen -o "%.8i %.12j %.8T %.10M %.12l"
tail -n 2 results/gen_vq_ce_dev/metrics.jsonl
```

Evaluate the milestone checkpoints (10/25/50/100%; they appear as
`results/gen_vq_ce_dev/ckpt/milestone_*.pt`):

```bash
cd ~/jncm
ls results/gen_vq_ce_dev/ckpt/milestone_*.pt
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,RUN_ID=gen_vq_ce_dev,TOKENIZER=results/tok_vq_10k/ckpt/latest.pt \
  scripts/slurm/eval.sbatch
```

The array is `0-4` (5 seeds, 256 samples each). When it has finished:

```bash
cd ~/jncm
python scripts/run_eval.py --run_id gen_vq_ce_dev --merge
python scripts/plot_curves.py
```

`--merge` folds `eval/step_<n>.seed_<i>.json` into `eval/step_<n>.json`;
`plot_curves.py` writes `curves.png` at the repo root (`--out <path>` to change
it) from every `results/**/eval/step_*.json` aggregate.

G0 gate: Gen PPL `<= 20` and H `>= 3.8` (NCM's budget-matched runs landed
between 15.4 and 19.6).

## 3. Phase 1 — noisy + flow and noisy + regress

Both use the noisy tokenizer and can run in parallel. Same config budget and
same tokenizer step count as G0, so milestone `total_positions` line up.

```bash
cd ~/jncm
ls -l results/tok_noisy_10k/ckpt/latest.pt
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,HEAD=flow,RUN_ID=gen_noisy_flow_dev,TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt \
  scripts/slurm/gen.sbatch
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,HEAD=regress,RUN_ID=gen_noisy_regress_dev,TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt \
  scripts/slurm/gen.sbatch
```

Evaluate and plot each run (two arrays, then two merges):

```bash
cd ~/jncm
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,RUN_ID=gen_noisy_flow_dev,TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt \
  scripts/slurm/eval.sbatch
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,RUN_ID=gen_noisy_regress_dev,TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt \
  scripts/slurm/eval.sbatch
python scripts/run_eval.py --run_id gen_noisy_flow_dev --merge
python scripts/run_eval.py --run_id gen_noisy_regress_dev --merge
python scripts/plot_curves.py
```

### 3.1 Reporting table (fill from the eval JSON; SPEC 6.1)

T1.2 data row is the fixed reference: **Gen PPL 10.80 +/- 0.26, H 4.03 +/-
0.01** (256 samples x 5 seeds, `results/t1_2_protocol_gate.json`).

| Run | total_positions | Gen PPL (absolute) | Gen PPL / 10.80 | H (GPT-2) | H (dataset) | Gen PPL vs G0 (ratio) |
| --- | --- | --- | --- | --- | --- | --- |
| T1.2 data row (reference) | n/a | 10.80 +/- 0.26 | 1.000 | 4.03 +/- 0.01 | 4.03 +/- 0.01 | n/a |
| `gen_vq_ce_dev` (Phase 0) | | | | | | 1.000 (baseline) |
| `gen_noisy_flow_dev` | | | | | | |
| `gen_noisy_regress_dev` | | | | | | |

How to judge (G1):

- **Gen PPL.** Report the absolute value, the ratio to the 10.80 data row
  (`ppl / 10.80`), and the ratio to the G0 run at the same milestone fraction
  (10/25/50/100%; `total_positions` are equal because both tokenizers and both
  generators use the same budgets). G1 passes if the Phase 1 ratio to G0 is
  `<= 1.15` at every compared milestone.
- **Entropy.** Judge both H variants against **4.03**: `|H - 4.03| <= 0.15` is
  the G1 tolerance; G0 only needs `H >= 3.8`.
- Also paste `unique_frac` and `pairwise_agree` for the record: a run that
  matches entropy while collapsing onto one sample is not a pass.

## 3A. Phase 2 — JEPA joint mode (G2)

The joint trainer updates the noisy tokenizer (E and D) and the generator G in
one loop against EMA targets, starting from `results/tok_noisy_10k/ckpt/latest.pt`.
Config `configs/jepa_dev.yaml`: head `flow`, `target_chain=online`, EMA on,
VICReg on, `decoder_through_E` true, batch 1024, 10,000 steps. Sizes match the
Phase-1 dev configs, so the milestone grids are comparable on the position axes.

`RUN_ID` and `STEPS` travel in `--export`. `SETS` is exported in front of
`sbatch` (not inside `--export`) because it carries spaces: one `key=value` per
ablation override, each turned into `--set key=value` by the script. `STEPS`
unset uses the config budget; `TOKENIZER` is the noisy tokenizer checkpoint.
Every run resumes from `results/<RUN_ID>/ckpt/latest.pt` when that file exists,
and writes `ckpt/milestone_{010,025,050,100}.pt` behind
`generator.train.save_milestones` (on in the config).

### 3A.1 Base run

```bash
cd ~/jncm
ls -l results/tok_noisy_10k/ckpt/latest.pt
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,RUN_ID=jepa_base_dev,STEPS=10000,TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt \
  scripts/slurm/jepa.sbatch
```

### 3A.2 Ablations (same command, only `RUN_ID` and `SETS` change)

| Run ID | `SETS` | Question |
| --- | --- | --- |
| `jepa_base_dev` | (empty) | base JEPA |
| `jepa_no_vicreg_dev` | `jepa.vicreg=false` | is VICReg needed? |
| `jepa_no_dec_through_E_dev` | `jepa.decoder_through_E=false` | does the decoder gradient into E help? |
| `jepa_neither_dev` | `jepa.vicreg=false jepa.decoder_through_E=false` | 2x2 corner |
| `jepa_ema_clean_dev` | `jepa.target_chain=ema_clean` | teacher-forced targets vs E-bar's own chain (SPEC 8 step 2) |
| `jepa_regress_dev` | `generator.head=regress` | deterministic head instead of flow |
| `jepa_no_ema_dev` | `jepa.ema=false` | is the EMA target needed at all? |

```bash
cd ~/jncm
SETS="jepa.vicreg=false" \
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,RUN_ID=jepa_no_vicreg_dev,STEPS=10000,TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt \
  scripts/slurm/jepa.sbatch
SETS="jepa.decoder_through_E=false" \
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,RUN_ID=jepa_no_dec_through_E_dev,STEPS=10000,TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt \
  scripts/slurm/jepa.sbatch
SETS="jepa.vicreg=false jepa.decoder_through_E=false" \
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,RUN_ID=jepa_neither_dev,STEPS=10000,TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt \
  scripts/slurm/jepa.sbatch
SETS="jepa.target_chain=ema_clean" \
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,RUN_ID=jepa_ema_clean_dev,STEPS=10000,TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt \
  scripts/slurm/jepa.sbatch
SETS="generator.head=regress" \
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,RUN_ID=jepa_regress_dev,STEPS=10000,TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt \
  scripts/slurm/jepa.sbatch
SETS="jepa.ema=false" \
sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
  --export=ALL,RUN_ID=jepa_no_ema_dev,STEPS=10000,TOKENIZER=results/tok_noisy_10k/ckpt/latest.pt \
  scripts/slurm/jepa.sbatch
```

Monitor:

```bash
squeue -u "$USER" -n jncm_jepa -o "%.8i %.12j %.8T %.10M %.12l"
tail -n 2 results/jepa_base_dev/metrics.jsonl
```

### 3A.3 Evaluate and plot

`TOKENIZER` is optional: a joint JEPA checkpoint carries its own tokenizer, so
leave it unset for Phase 2 and `run_eval.py` reads the tokenizer from the
checkpoint. (Set `TOKENIZER=<path>` only for a staged Phase-0/Phase-1 run.) One
array task per seed (`--array=0-4`), 4 milestone checkpoints each:

```bash
cd ~/jncm
for r in jepa_base_dev jepa_no_vicreg_dev jepa_no_dec_through_E_dev \
         jepa_neither_dev jepa_ema_clean_dev jepa_regress_dev jepa_no_ema_dev; do
  sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=<WALL_CLOCK> \
    --export=ALL,RUN_ID="$r" \
    scripts/slurm/eval.sbatch
done
```

When every array has finished (`sacct` shows `COMPLETED` for each task):

```bash
cd ~/jncm
for r in jepa_base_dev jepa_no_vicreg_dev jepa_no_dec_through_E_dev \
         jepa_neither_dev jepa_ema_clean_dev jepa_regress_dev jepa_no_ema_dev; do
  python scripts/run_eval.py --run_id "$r" --merge
done
python scripts/plot_curves.py --x total_positions --out results/curves.png
python scripts/plot_curves.py --x total_positions_conservative \
  --out results/curves_conservative.png
```

The two plots hold the same data on two x axes: SPEC 6.2 counts a joint step
once (`total_positions = joint_positions`) and conservatively twice
(`total_positions_conservative = 2 x joint_positions`). A staged run has no
conservative count, so on that plot it keeps its own `total_positions`.

### 3A.4 G2 verdict under both accountings

G2 (TASKS): the JEPA variant matches or beats the best Phase-1 variant at equal
`total_positions`. Read it off both plots and keep both verdicts; a value within
the Phase-1 `ppl_std` counts as a match.

- **`curves.png` (x = `total_positions`).** Compare at equal x. The milestone
grids differ: Phase 1 adds the 655.36M-token tokenizer budget and lands at
0.79B / 0.98B / 1.31B / 1.97B, JEPA at 0.13B / 0.33B / 0.66B / 1.31B. Compare
each JEPA milestone with the Phase-1 curve at the same x (interpolate between
the two surrounding Phase-1 milestones when needed); the headline point is
JEPA 100% (1.31B) against the Phase-1 50% milestone (1.31B). Pass if JEPA Gen
PPL is `<=` the Phase-1 value at every compared x.
- **`curves_conservative.png` (x = `total_positions_conservative`).** The same
read at double x. Only the JEPA 10/25/50% points (0.26B / 0.65B / 1.31B) have
Phase-1 counterparts below 1.97B; the JEPA 100% point (2.62B) is past every
Phase-1 run, so report it and exclude it from pass/fail. Pass under the same
`<=` rule.
- The best Phase-1 variant is whichever of `gen_noisy_flow_dev` /
`gen_noisy_regress_dev` has the lower Gen PPL at the compared position; name it.
- A JEPA run that trips the collapse alarm or has `pairwise_agree` near 1 fails
regardless of Gen PPL.

### 3A.5 Paste back (JEPA diagnostics and eval)

```bash
cd ~/jncm
python - <<'PY'
import glob, json, os

for run in ("jepa_base_dev",):
    print(f"== {run}")
    rows = [json.loads(line) for line in open(f"results/{run}/metrics.jsonl") if line.strip()]
    last = rows[-1]
    print(f"step {last['step']} loss {last['train_loss']:.4f} dec {last['loss_dec']:.4f} "
          f"pred {last['loss_pred']:.4f} vic {last['loss_vic']:.4f} "
          f"joint_positions {last['joint_positions']} "
          f"conservative {last['total_positions_conservative']}")
    for k in range(len(last["std_min"])):
        print(f"  L{k + 1:02d} std {last['std_min'][k]:.3f}/{last['std_mean'][k]:.3f} "
              f"target_std {last['target_std_min'][k]:.3f}/{last['target_std_mean'][k]:.3f} "
              f"eff_rank {last['eff_rank'][k]:.2f} "
              f"target_eff_rank {last['target_eff_rank'][k]:.2f} "
              f"R2 {last['r2'][k]:.3f}")
    print(f"  collapse_alarm {last['collapse_alarm']} steps {last['collapse_steps']}")
    for path in sorted(glob.glob(f"results/{run}/eval/step_*.json")):
        if ".seed_" in os.path.basename(path):
            continue
        rec = json.load(open(path))
        print(f"  step {rec['step']} total_positions {rec['total_positions']} "
              f"conservative {rec['total_positions_conservative']} "
              f"ppl {rec['ppl_mean']:.2f} ppl/10.80 {rec['ppl_mean'] / 10.80:.3f} "
              f"h_gpt2 {rec['h_gpt2_mean']:.2f} h_dataset {rec['h_dataset_mean']:.2f} "
              f"unique_frac {rec['unique_frac_mean']:.3f} "
              f"pairwise_agree {rec['pairwise_agree_mean']:.3f}")
PY
```

Paste per-level `std` (c_k and t_k), both effective ranks, `R2`, the collapse
alarm with its consecutive-step counts, Gen PPL ratio to the 10.80 data row
(`ppl_mean / 10.80`), and `pairwise_agree` for every arm.

## 3B. Gate report: G0, G1 and G2 in one command

`scripts/gate_report.py` reads only the merged eval aggregates
(`results/<run_id>/eval/step_<n>.json`, so run every `--merge` first: §2.2, §3,
§3A.3). It prints one table -- Gen PPL mean +/- std, PPL/10.80, H(gpt2) and
H-4.03, H(dataset), `unique_frac`, `pairwise_agree`, `total_positions` and
`total_positions_conservative`, one row per milestone -- then the verdict lines.
No GPU and no torch: it runs on a login node.

```bash
cd ~/jncm
python scripts/gate_report.py
echo "EXIT=$?"
```

Defaults: G0 `gen_vq_ce_dev`, flow `gen_noisy_flow_dev`, regress
`gen_noisy_regress_dev`, JEPA `jepa_base_dev`. To judge every Phase-2 arm
against the best Phase-1 run:

```bash
cd ~/jncm
python scripts/gate_report.py --jepa jepa_base_dev jepa_no_vicreg_dev \
  jepa_no_dec_through_E_dev jepa_neither_dev jepa_ema_clean_dev \
  jepa_regress_dev jepa_no_ema_dev
echo "EXIT=$?"
```

How it judges (thresholds are the TASKS gates, reference row is T1.2):

- **G0:** Gen PPL `<= 20` and H(gpt2) `>= 3.8` at every milestone.
- **G1 flow:** Gen PPL ratio to G0 `<= 1.15` at equal `total_positions`, and
  `|H(gpt2) - 4.03| <= 0.15`, at every milestone. Requires at least one
  comparable milestone, else the verdict is `not comparable`.
- **G1 regress:** reported only (no verdict); a milestone with
  `pairwise_agree > 0.5` is marked `FLAG` (collapse onto one sample).
- **G2:** for every JEPA milestone, Gen PPL `<=` the best Phase-1
  (flow/regress) Gen PPL at equal positions plus that run's interpolated
  `ppl_std`; judged once under `total_positions` and once under
  `total_positions_conservative`, and each verdict line names its accounting.
- Equal positions are compared by linear interpolation of Gen PPL (and its
  std) between the two surrounding milestones of the reference curve. A
  position outside the reference curve's observed range prints
  `not comparable`; nothing is extrapolated.
- Exit code `1` if a gate FAILs or a run has no merged eval records (`MISSING`),
  else `0`. `not comparable` is not a failure -- it means the runs do not
  overlap in positions yet.

## 4. Paste back (metrics tails for review)

Run on a login node in `~/jncm`; these read JSON/JSONL only (no torch, no GPU).
Each block prints well under 30 lines.

### 4.1 Scheduler state

```bash
cd ~/jncm
squeue -u "$USER" -n "jncm_tok,jncm_gen,jncm_eval,jncm_jepa" -o "%.8i %.12j %.8T %.10M %.12l"
sacct -j 2880034,2880035 --format=JobID%12,JobName%12,State,Elapsed,Timelimit,ExitCode
```

### 4.2 Tokenizer tails (G0a verdict)

```bash
cd ~/jncm
for r in tok_vq_10k tok_noisy_10k; do
  echo "== $r"
  python - "$r" <<'PY'
import json, sys
run = sys.argv[1]
rows = [json.loads(line) for line in open(f"results/{run}/metrics.jsonl") if line.strip()]
last = rows[-1]
cum = ", ".join(f"{v:.4f}" for v in last.get("acc_cum", []))
thr = 0.999 if "vq" in run else 0.995
acc = float(last.get("acc_full", 0.0))
print(f"step {last.get('step')} acc_full {acc:.4f} acc_cum [{cum}] "
      f"data_source {last.get('data_source')} tok_positions {last.get('tok_positions')}")
print(("PASS" if last.get("data_source") == "real" and acc >= thr else "FAIL")
      + f" vs {thr:.3f}")
PY
done
```

### 4.3 Generator training tail

```bash
cd ~/jncm
for r in gen_vq_ce_dev gen_noisy_flow_dev gen_noisy_regress_dev; do
  echo "== $r"
  tail -n 1 "results/$r/metrics.jsonl"
done
```

### 4.4 Eval aggregates (absolute PPL, ratio to 10.80, entropy, diversity)

```bash
cd ~/jncm
python - <<'PY'
import glob, json, os

REF_PPL, REF_H = 10.80, 4.03


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


for run in ("gen_vq_ce_dev", "gen_noisy_flow_dev", "gen_noisy_regress_dev"):
    print(f"== {run}")
    for path in sorted(glob.glob(f"results/{run}/eval/step_*.json")):
        if ".seed_" in os.path.basename(path):
            continue
        rec = json.load(open(path))
        ppl = rec.get("ppl")
        ratio = "n/a" if ppl is None else f"{ppl / REF_PPL:.3f}"
        h_gpt2, h_data = rec.get("h_gpt2"), rec.get("h_dataset")
        dev = "n/a" if h_gpt2 is None else f"{h_gpt2 - REF_H:+.3f}"
        print(f"step {rec.get('step')} total_positions {rec.get('total_positions')} "
              f"ppl {fmt(ppl)} std {fmt(rec.get('ppl_std'))} ppl/10.80 {ratio} "
              f"h_gpt2 {fmt(h_gpt2)} h_dataset {fmt(h_data)} h-4.03 {dev} "
              f"unique_frac {fmt(rec.get('unique_frac'))} "
              f"pairwise_agree {fmt(rec.get('pairwise_agree'))}")
PY
```

`ppl_std` is absent when the aggregate has no records; `n/a` then means the
milestone has no merged aggregate yet (run `--merge` first).

### 4.5 Commit identity for the run being reviewed

```bash
cd ~/jncm && git rev-parse HEAD && git status --short
cat results/<run_id>/git_sha.txt
```

The two SHAs must match; a `-dirty` suffix on `git_sha.txt` means the run used
uncommitted code.
