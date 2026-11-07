# RUNBOOK — jncm on FIR

Operator commands for Gate G0a (tokenizers) and the Phase 0 / Phase 1 runs.
Every wall clock is written `<WALL_CLOCK>` (`HH:MM:SS`); nothing else is blank.

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

## 4. Paste back (metrics tails for review)

Run on a login node in `~/jncm`; these read JSON/JSONL only (no torch, no GPU).
Each block prints well under 30 lines.

### 4.1 Scheduler state

```bash
cd ~/jncm
squeue -u "$USER" -n "jncm_tok,jncm_gen,jncm_eval" -o "%.8i %.12j %.8T %.10M %.12l"
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
