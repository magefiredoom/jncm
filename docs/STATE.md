# STATE — jncm

Last updated: 2026-11-07 (T0.2 and T1.2 verified on real data; Gen PPL 10.80, H 4.03)

Rule: newest information wins. Rewrite sections; do not append history.

## Current position
- Milestone / task in progress: M0–M2 complete; **T2.4, T3.1, T3.2, T3.3, T3.4, T3.5, T4.1, T0.2, T1.2 complete**; next task **T4.2** (eval and plots), with G0a/G0 still human-run on FIR.
- Last completed task: T1.2 protocol gate on real TinyStories validation data (`scripts/eval_data_row.py`, 256 samples): Gen PPL 10.80 +/- 0.26 and H 4.03 +/- 0.01, both inside tolerance. Before that, T3.5 (`jncm/train_generator.py`); Q8 resolved.
- T3.5 acceptance passes as written in TASKS.md: `ce` and `flow` memorise 8 sequences; `regress` memorises one unconditional sequence and, with the true level-1 code supplied, reproduces all 8 sequences when sampling levels 2..K.
- T4.1 is ticked for prep/tok/gen only. TASKS assigns `scripts/slurm/eval.sbatch` to T4.2 (it evaluates checkpoints written by `run_eval.py`).
- T0.2 real data prep and the T1.2 protocol gate are now verified locally on real TinyStories data; `models/gpt2-large` is present in `models/gpt2-large`.
- Gate G0a is next (cluster, human): copy `data/tinystories/` and `models/gpt2-large/` to FIR first.

## VERIFIED (command + last lines of output)

### T0.2 real data prep — VERIFIED 2026-11-07
- `data/tinystories/meta.json`: `train_windows 3647424`, `val_windows 36658`, `L 128`, `V 8192`; `train.bin` 933,740,544 bytes = 3,647,424*128*2 and `val.bin` 9,384,448 bytes = 36,658*128*2.
- Token-id scan over the first and last 1,000,000 tokens of both bins: max id 8191 (train) and 8191 (val, tail max 8190), all < V.
- Val windows 0 and 36657 decode with `tokenizer.json` to readable English prose.
- `python scripts/prep_tinystories.py --max_stories 2000 --out data/tiny_test`
  `Train: 3210 windows (410880 tokens)` / `Val:   637 windows (81536 tokens)`; `data/tiny_test/meta.json` has L 128, V 8192 and both bin sizes match exactly. (The shell exit 1 was PowerShell treating the HF auth warning on stderr as an error; the script printed success and the files are complete.)
- `pytest -q tests/test_data_prep.py tests/test_eval_data_row.py tests/test_genppl.py`
  `13 passed in 2.03s`

### T1.2 protocol gate — VERIFIED 2026-11-07
- `GPT2_LARGE_PATH=models/gpt2-large python scripts/eval_data_row.py --data data/tinystories/val.bin --tokenizer data/tinystories/tokenizer.json --gpt2 models/gpt2-large --n_samples 256 --out results/t1_2_protocol_gate.json`
  `Gen PPL:           10.80 +/- 0.26 (Target: ~9.67, tol 15% [8.22, 11.12])`
  `H (GPT-2 tokens):  4.03 +/- 0.01 (Target: ~3.96, tol 0.15 [3.81, 4.11])`
  `H (Dataset tokens):4.03 +/- 0.01`
  `Verdict: PASS`
- `results/t1_2_protocol_gate.json`: `n_samples 256`, `n_seeds 5`; PPL runs 10.46–11.23, H std <= 0.008. Metric, tokenizer and thresholds unchanged.

### G0a cluster run — in progress (2026-11-07)
- Local tree: `git status` clean after committing `AGENTS.md` (FIR operator rules); one empty untracked `docs/test.py` remains.
- Fast suite: `.\.venv\Scripts\python.exe -m pytest -q -m "not slow"`
  `122 passed, 4 deselected in 19.36s`  (EXIT=0)
- Real data already present: `data/tinystories/{train,val}.bin` + `tokenizer.json` + `meta.json`; meta `train_windows 3647424`, `val_windows 36658`, `L 128`, `V 8192`. `scripts/prep_tinystories.py` was not re-run.
- `scripts/slurm/tok.sbatch` placeholders filled: `--time=12:00:00` (template default, overridden per submit), `--cpus-per-task=6`, `--mem=32000M`, `module load python/3.12`, venv `.venv` unchanged; optional `STEPS` / `RUN_ID` overrides (`RUN_ID` defaults to `tok_<quant>_dev`, `STEPS` unset = config budget). `bash -n` (Git bash) exit 0.
- FIR target (from user): host `login2.int.fir.alliancecan.ca`, user `aefad`, repo `~/jncm`, python module `python/3.12`, no partition flag. FIR rejects the literal account `def-cheng23_`; the valid account is `def-cheng23` (Slurm error lists it).
- FIR requires an explicit GPU model: `gpu:h100` exists (`sinfo -o "%G"` shows `gpu:h100:4(S:2-3)`), so `tok.sbatch` now requests `--gres=gpu:h100:1`; a generic `--gres=gpu:1` is rejected at submit time.
- Venv on FIR: `module load python/3.12` (3.12.4), `virtualenv --no-download .venv`, `pip install --no-index torch numpy pyyaml` -> torch 2.14.1+computecanada, numpy 2.5.3, pyyaml 6.0.3.
- Smoke: `python -m jncm.train_tokenizer --config configs/smoke.yaml --steps 5 --set L=128 --set V=8192 --run_id smoke_g0a` -> 5 steps, real data, exit 0. (The first attempt failed loudly as designed: real meta L=128 vs smoke config L=32.)
- Pilot submission: `sbatch --account=def-cheng23 --gres=gpu:h100:1 --time=00:20:00 --export=ALL,QUANT=vq,STEPS=500,RUN_ID=pilot_vq scripts/slurm/tok.sbatch` -> job 2796353.
- FIR channel proven through the zellij pane (`hostname -f` = `login2.int.fir.alliancecan.ca`). `ssh -o BatchMode=yes` from this host is denied (publickey,keyboard-interactive), so every new ssh/rsync needs human MFA.
- G0a numbers: not run yet (no pilot, no full runs).

### T4.1 Slurm templates — prep, tok, gen (eval moved to T4.2)
- One command, `set -o pipefail`, from the repo root: `bash -n` on each script, then the full suite.
  `bash -n scripts/slurm/prep.sbatch exit=0`
  `bash -n scripts/slurm/tok.sbatch exit=0`
  `bash -n scripts/slurm/gen.sbatch exit=0`
  `.venv/Scripts/python.exe -m pytest -x -q`
  `123 passed, 1 xfailed in 144.20s (0:02:24)`  (EXIT=0)
- Requirement audit from the same command's `grep -n` output:
  - `TODO(placeholder)` lines: `prep.sbatch` 6 (wall clock), 7 (modules), 9 (venv); `tok.sbatch` 7 (wall clock), 8 (GPU type), 10 (modules), 12 (venv); `gen.sbatch` same four at 7/8/10/12.
  - `^#SBATCH` lines are only `--job-name`, `--nodes`, `--ntasks`, `--gres=gpu:1`, `--time=HH:MM:SS`. No account, no partition, no GPU type anywhere in the three files (`--gres=gpu:1` is generic; the only `gpu:A100:1` text is the example inside a TODO comment).
  - `HF_HUB_OFFLINE=1` + `HF_DATASETS_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`: 3/3 in `tok.sbatch` and `gen.sbatch`, 0 in `prep.sbatch` (deliberate, Q9).
  - `set -euo pipefail` at `prep.sbatch:29`, `tok.sbatch:33`, `gen.sbatch:41`; each script then checks `.venv/bin/activate`, sources it, and exits 1 with a hint if it is missing.
- `python -m jncm.train_tokenizer --help` lists `--resume RESUME` and `--run_id RUN_ID`; loading `configs/tinystories_dev.yaml` with `--set tokenizer.quantizer=noisy` gives `quantizer 'noisy' L 128 V 8192 max_steps 40000 save_every 5000 device cuda`.
- Option plumbing: `tok.sbatch` requires `QUANT` (`: "${QUANT:?...}"`, case-checked to `vq|noisy`) and runs `--set tokenizer.quantizer=${QUANT} --run_id tok_${QUANT}_dev`; `gen.sbatch` requires `HEAD`, `TOKENIZER` and `RUN_ID`, maps `ce -> vq` and `regress|flow -> noisy` (SPEC 5.3), passes `--tokenizer ${TOKENIZER}`, `--set generator.head=${HEAD}`, `--set tokenizer.quantizer=${quant}`, `--run_id ${RUN_ID}`, and fails with exit 1 if the tokenizer checkpoint is absent. Both append `--resume results/<run_id>/ckpt/latest.pt` only when that file exists.
- Not submitted: this host has no Slurm, so the jobs are syntax-checked and inspected only (UNVERIFIED as cluster jobs; see "Human must fill").
- T4.1 TODO markers appear only in `scripts/slurm/*.sbatch` (no `TODO`/`pass`/`...` stub in `jncm/`), and this section lists every one of them.

### T3.5 generator training — frozen tokenizer, all levels per pass, accounting, resume, diversity
- `.venv/Scripts/python.exe -m pytest -q -m "not slow" tests/test_train_generator.py`
  `.............                                                            [100%]`
  `13 passed, 4 deselected in 8.66s`
  Covers (a) `load_tokenizer` returns an eval-mode, `requires_grad`-off tokenizer plus the checkpoint's `tok_positions`; wrong `L`, wrong variant and a missing file raise; (b) the weighted loss equals a manual SPEC 5.1/5.4/5.5 recomputation for `corrupt.mode` `none` and `swap`, and with `swap` the chain codes really change while the targets stay clean; (c) exactly one packed backbone pass per loss, shape `[B*K, L, H]`; (d) records carry `step`, `train_loss`, `per_level_loss`, `lr`, `data_source`, `tok_positions` (offset from the tokenizer checkpoint), `gen_positions = step*B*L`, `total_positions = sum`; (e) resume continues from the checkpoint's step and positions; (f) the eval hook writes `samples/step_<n>.txt` (<= 32 lines), decodes every sample and logs `gen_ppl` (fake scorer gives exactly exp(1) = 2.718), `unique_frac` and `pairwise_agree`; (g) head/variant pairing is enforced (`ce`+vq, `regress`/`flow`+noisy) and `corrupt.mode=noise` on vq raises; (h) the SPEC 6.1 endpoints: identical samples give `unique_frac = 1/n` and `pairwise_agree = 1`, fully different samples with no shared tokens give 1 and 0.
- Slow acceptance (`pytest -m slow`, CPU): tokenizer overfit 700 steps on the 8 fixed sequences (round-trip accuracy 1.000, vq and noisy), generator batch 8, `corrupt.mode=none`.
  - `ce` 600 steps, lr 1e-3: 8 sequences at >= 0.90 (previous run 1.000).
  - `flow` 1200 steps, lr 2e-3: 8 sequences at >= 0.80 (previous run 0.961, distinct matches 4).
  - `regress` one-sequence arm, 1000 steps: >= 0.90.
  - `regress` teacher-forced arm, 2500 steps: the true level-1 code is supplied and levels 2..K are sampled; all 8 sequences >= 0.90 per sequence.
- The old `regress` xfail arm is removed (Q8 resolved, DECISIONS 4). `pytest -q -m slow tests/test_train_generator.py -k teacher_forced` -> `1 passed, 16 deselected in 485.66s (0:08:05)`.
- Full suite: `.venv/Scripts/python.exe -m pytest -x -q`
  `........................................................................ [ 57%]`
  `......................................................                   [100%]`
  `126 passed in 188.08s (0:03:08)`  (EXIT=0)

### T3.4 sampler — exactly K backbone calls, shapes, seeded determinism
- `pytest -q tests/test_generator_sample.py`
  `..............                                                           [100%]`
  `14 passed in 3.00s`
  Covers (a) a forward hook on the backbone sees exactly K calls, with inputs `[B, l_k, H]` for the SPEC schedule, at B = 1 and B = 3; (b) the count is K for K = 1 as well as K = 4; (c) tokens `[B, L]`, dtype long, ids in `[0, V)` for `ce`/`regress`/`flow`; (d) same `torch.Generator` seed gives `torch.equal` samples and a different seed gives different samples, for all three heads; (e) the sampler reproduces the SPEC 5.6 pseudocode step for step (manual `embed_level` -> `backbone` -> `head.sample_level` -> `contrib` loop with the same seed, `flow` head, `torch.equal`); (f) mismatched head K and `batch_size=0` raise; (g) the sampler leaves module train/eval modes untouched.
- Test-teeth probes (temporary source mutants, then restored; `grep -c MUTANT` = 0 afterwards):
  `MUTANT 1 (drop zhat accumulation)`: `4 failed, 10 passed in 4.18s` (all three seeded tests plus the manual-loop test)
  `MUTANT 2 (extra backbone call at level 1)`: `4 failed, 10 passed in 2.42s` (both call-count tests)
- Full suite from repo root: `.venv/Scripts/python.exe -m pytest -x -q`
  `........................................................................ [ 66%]`
  `.....................................                                    [100%]`
  `109 passed in 18.94s`  (EXIT=0)

### T3.3 corruption — swap rate, clean targets, identity `none`
- `pytest -q tests/test_generator_corruption.py`
  `.........                                                                [100%]`
  `9 passed in 2.00s`
  Covers (a) empirical swap rate per level within 0.03 of `p_j` (levels 1/11/22/32, B=1024); (b) every swapped code is another (batch, position) of its own level, and at rate 1.0 nothing stays put; (c) targets untouched through the real frozen tokenizer (and `enc.codes` unmodified; the corrupted conditioning does change the backbone output, so the check is not vacuous); (d) `mode="none"` returns the identical tensors; (e) seeded determinism plus out-of-place behaviour; (f) `noise` empirical std/mean per level; (g) `noise` rejected unless `quantizer="noisy"` and rejected for integer codes; (h) `linear_schedule` endpoints (0.5/0.1 for noise), wrong rate count, rate out of [0,1] and negative sigma raise.
- Probe used for the numbers above (B=1024, seeds 3/11, schedule [1, 11, 22, 32]):
  `swap target [0.8, 0.6, 0.4, 0.2]`
  `swap observed [0.8164, 0.6009, 0.3957, 0.1959]`
  `noise target [0.5, 0.3667, 0.2333, 0.1]`
  `noise observed std [0.5001, 0.3662, 0.2338, 0.1001]`
- Full suite from repo root: `.venv/Scripts/python.exe -m pytest -x -q`
  `........................................................................ [ 75%]`
  `.......................                                                  [100%]`
  `95 passed in 16.90s`  (EXIT=0)

### T3.2 heads and losses — ce/regress overfit, flow bimodal toy, determinism
- `pytest -q tests/test_generator_heads.py`
  `.............                                                            [100%]`
  `13 passed in 11.82s`
  Covers (a) `ce` overfits a fixed batch through the real frozen-tokenizer → backbone → head path: all 66 code indices at all 4 levels predicted exactly, total loss < 1e-2; (b) `regress` overfits the same path to total MSE 4.62e-4 (per level <= 3.1e-4), cosine >= 0.99; (c) `flow` learns the 2-D two-Gaussian toy (modes +-(1,1), std 0.1): 97.3% of 512 samples within 0.5 of a mode and a 0.453 / 0.547 split (thresholds 35% each, 80% near a mode); (d) seeded determinism for all three samplers (same generator → `torch.equal`, different seed → different samples) and for the flow training loss; (e) `ce` loss equals `F.cross_entropy` of the head logits, `regress` cosine option matches `1 - cos`; (f) `regress` renorm gives per-position unit variance; (g) SPEC 5.5 weights (`alpha=0` position-weighted, `alpha=1` equal per level, `sum_k w_k l_k = 1`) and the factory's argument validation.
- Overfit-batch sizing probe (batch size 1 for a reason, see ASSUMPTIONs): `H=32 L=1 steps=1200 lr=0.005: loss=4.62e-04 per=[1e-06, 0.000107, 0.000308, 4.7e-05] t=4.8s`; `H=64 L=2 steps=600: loss=1.72e-02`. Toy probe at 600 steps: `loss 0.8926 frac+ 0.453 near<0.5 0.973 xmean [-0.069, -0.07] xstd [0.951, 0.957]` (the two-mode target has an irreducible ambiguity floor near 0.86, so the toy test checks bimodality and a 40% loss drop, not a small absolute loss).
- Full suite from repo root: `pytest -x -q`
  `........................................................................ [ 83%]`
  `..............                                                           [100%]`
  `86 passed in 16.56s`  (EXIT=0)

### T3.1 backbone — no-leak, padding invariance, shapes
- `pytest -q tests/test_generator_backbone.py`
  `.........                                                                [100%]`
  `9 passed in 2.04s`
  Covers (a) for k = 1..K, level-k outputs are bit-identical (`torch.equal`) when the chain codes at levels >= k are replaced, for both `vq` and `noisy`; (b) level-k output does change when only level k-1 codes change, so the no-leak test is not vacuous; (c) garbage at padded positions (`randn*100`) leaves every valid output bit-identical through `forward_levels`; (d) shapes `[B, l_k, H]` for all K levels plus the single-level `embed_level`/`forward` path, which matches the packed path to 1e-5; (e) `embed_levels` raises when fewer than K-1 prefix codes are given.
- Test-teeth probe (monkeypatched mutants, then restored):
  `no-leak (level k sees its own code): mutant killed: vq: level 1 output changed when only codes of levels >= 1 changed`
  `padding (mask dropped): mutant killed: padding leaked into level 1`
- Full suite from repo root: `pytest -x -q`
  `........................................................................ [ 98%]`
  `.                                                                        [100%]`
  `73 passed in 8.43s`

### T2.4 regression tests and full suite
- Baseline before fixes: `pytest -q tests/test_review_fixes.py`
  `6 failed, 4 passed in 1.85s`
  Failures: F2[vq] (`index_add_` Float vs BFloat16), F3 missing `WindowPacker`, F4 explicit-synthetic kwarg, F4 meta mismatch, F6 unknown key, F10 schedule `[1, 2, 4]`.
  Already passing at baseline: F1 (QKV was fixed last session), F2[noisy], F4 no-silent-fallback, F7 (straight-through VQ is the intended design).
- Final: `pytest -q tests/test_review_fixes.py`
  `............                                                             [100%]`
  `12 passed in 6.29s`
- Final full suite from repo root (venv on PATH): `pytest -q`
  `................................................................         [100%]`
  `64 passed in 8.67s`
- F1 right-reason probe:
  `params 49984 expected 49984 has_qkv_out True`
  (exact 12*d^2+13*d parameter count; learned `qkv` and `out_proj` exist.)
- F4 right-reason probe:
  `missing: FileNotFoundError Prepared data not found at missing (need train.bin and val.b`  
  `explicit synthetic: [128, 32]`  
  `meta: ValueError meta.json L=128 does not match config L=32`

### T2.3 / T2.4 smoke run (CPU)
- `.\.venv\Scripts\python.exe -m jncm.train_tokenizer --config configs/smoke.yaml --steps 50 --allow_synthetic --run_id tok_vq_t24`
  `Starting training run tok_vq_t24 (vq) for 50 steps on cpu...`  
  `Step     1 | Loss: 4.3445 | acc_full: 0.0176 | tok_pos: 512`  
  `Step    50 | Loss: 4.0646 | acc_full: 0.0703 | tok_pos: 25600`  
  `Training completed. Results written to results\tok_vq_t24`
- `results/tok_vq_t24/git_sha.txt` = `373f9141c7b5a2b067c5598ca832c6082618ddd6` (clean tree).
- Last `metrics.jsonl` record: `data_source: "synthetic"`, `acc_cum: [0.015625, 0.017578125, 0.02734375, 0.0703125]`, `acc_full: 0.0703125`, `tok_positions: 25600`, `gen_positions: 0`, `total_positions: 25600`, `nominal_bits: 6.0`, codebook usage/perplexity present.
- `python -m pip install -e .` finished with `Successfully installed jncm-0.1.0`.
- `python scripts/eval_data_row.py --help` starts with:
  `usage: eval_data_row.py [-h] [--data DATA] [--tokenizer TOKENIZER] ...`
  `Evaluate Gen PPL and H on real validation data`

## DECISIONS
1. **VQ codes stay straight-through vectors.** `Encoded.codes` are float vectors with the straight-through estimator; `Encoded.targets` are the integer indices. The generator conditions from `targets`, never `codes` (SPEC 4.1; F7 test `reconstruct(enc.targets)` passes).
2. **Keep the learned QKV/output projections and delete old smoke checkpoints.** All pre-existing `.pt` files under `results/` were deleted; only checkpoints from the current architecture remain (`results/tok_vq_t24`). Old Gemini/pre-QKV checkpoints are not loadable.
3. **Level schedule uses the exact half-up integer formula** `l_k = 1 + (2*(k-1)*(L-1) + (K-1)) // (2*(K-1))` (SPEC 2 / v1.2, F10). No Python `round`.
4. **Q8 resolved: keep SPEC 5.1 and the deterministic `regress` head.** Level 1 has one position and zero conditioning, so a deterministic head can only emit one code. The old 8-sequence unconditional acceptance was an invalid test, not a model bug. T3.5 now accepts (a) one unconditional fixed sequence at >= 0.90 and (b) the true level-1 code supplied, sampling levels 2..K, all 8 sequences at >= 0.90. SPEC 5.1 and the regress head are unchanged.

## Files touched in T0.2 / T1.2 (verification only)
- Docs: `docs/STATE.md`, `docs/TASKS.md`. No code changes; `data/tiny_test` was generated by the existing prep script, and `results/t1_2_protocol_gate.json` is the gate record (both git-ignored).

## Files touched in T3.5
- Code: `jncm/train_generator.py` (new: `load_tokenizer`, `build_generator`, `build_optimizer`, `compute_generator_loss`, `diversity_metrics`, `sample_and_score`, `run_diagnostics`, `train`, `build_decode_fn`, CLI), `jncm/config.py` (`GeneratorTrainConfig` with the SPEC 5.7 defaults, exposed as `generator.train`).
- Configs: `configs/smoke.yaml`, `configs/tinystories_dev.yaml` (`generator.train` block; smoke uses batch 16 / 200 steps, dev uses the SPEC 5.7 values).
- Tests: `tests/test_train_generator.py` (new: 13 fast tests + 4 slow acceptance arms; SPEC 6.1 diversity endpoints).
- Docs: `docs/STATE.md`, `docs/TASKS.md`, `docs/QUESTIONS.md` (Q8 resolved) (SPEC.md not edited; the researcher replaced it with v1.3).

## Files touched in T3.4
- Code: `jncm/generator/sample.py` (new: `sample_tokens`, the SPEC 5.6 coarse-to-fine loop), `jncm/generator/__init__.py` (exports `sample_tokens`, `LevelHead`).
- Tests: `tests/test_generator_sample.py` (new, 14 tests).
- Docs: `docs/STATE.md`, `docs/TASKS.md`, `docs/QUESTIONS.md` (Q7) (SPEC.md not edited).

## Files touched in T3.3
- Code: `jncm/generator/corruption.py` (new: `linear_schedule`, `swap_codes`, `add_code_noise`, `corrupt_codes`, `MODES`), `jncm/generator/__init__.py` (exports).
- Tests: `tests/test_generator_corruption.py` (new, 9 tests).
- Docs: `docs/STATE.md`, `docs/TASKS.md` (SPEC.md not edited; no docs/QUESTIONS.md entry needed — the three ASSUMPTIONs below are non-blocking).

## Files touched in T3.2
- Code: `jncm/generator/heads.py` (new: `CEHead`, `RegressHead`, `FlowHead`/`FlowMLP`, `SinusoidalTimeEmbedding`, `build_head`, `level_weights`, `weighted_total`, `sample_codes`), `jncm/generator/__init__.py` (exports), `jncm/config.py` (`generator.regress_cosine`).
- Configs: `configs/smoke.yaml`, `configs/tinystories_dev.yaml` (`regress_cosine: false`).
- Tests: `tests/test_generator_heads.py` (new, 13 tests).
- Docs: `docs/STATE.md`, `docs/TASKS.md`, `docs/QUESTIONS.md` (Q6) (SPEC.md not edited).

## Files touched in T3.1
- Code: `jncm/generator/backbone.py` (was untracked scratch from an earlier session; reviewed, no behavioural change needed), `jncm/generator/__init__.py`.
- Tests: `tests/test_generator_backbone.py` (new, 9 tests).
- Docs: `docs/STATE.md`, `docs/TASKS.md` (SPEC.md not edited).

## Files touched in T4.1
- `scripts/slurm/prep.sbatch`, `scripts/slurm/tok.sbatch`, `scripts/slurm/gen.sbatch` (new), `docs/STATE.md`, `docs/TASKS.md`, `docs/QUESTIONS.md` (Q9). No `eval.sbatch` — T4.2 owns it.

## Files touched in T2.4
- Code: `jncm/config.py`, `jncm/levels.py`, `jncm/tokenizer/vq.py`, `jncm/train_tokenizer.py`, `scripts/prep_tinystories.py`.
- Packaging: `pyproject.toml` (new), `.gitignore` (zip archives).
- Tests: `tests/test_review_fixes.py` (F5/F8 added), `tests/test_train_tokenizer.py` (F8 `results_dir`, F9 dirty/optimizer tests), `tests/test_data_prep.py` (F9 byte alphabet).
- Docs: `docs/STATE.md`, `docs/TASKS.md`, `docs/QUESTIONS.md` (SPEC.md not edited).

## UNVERIFIED / not run
- T0.2 and T1.2 are verified on this host (see VERIFIED); they are not yet repeated on FIR.
- T3.5: verified only at CPU smoke scale and on the 600–1200 step memorization runs above. Real generator runs (H=768, 12 layers, batch 1024, 10k steps, about 1.3B positions, SPEC 5.7) are untested; Gen PPL in T3.5 used a fake scorer (the real GPT-2 path is verified separately by T1.2). The `regress` memorization arm is structurally blocked (Q8).
- T3.1–T3.4 were verified on eval-mode modules; T3.5 is the first end-to-end (tokenizer -> generator -> decode) check, but only on 8 fixed sequences.
- G0a/G0/G1/G2: human cluster gates, not run.
- If any F2 test passed before its fix: `F2[noisy]` passed. The vq crash was real (`index_add_` dtype); the fp32 quantizer change was kept for both variants.

## Known issues / deviations
1. `GPT2Scorer.score_sample` scores one text at a time; batching is deferred to T4.2 as the review suggests.
2. Resume restores model/optimizer but not the data iterator or RNG; reproducibility across resumes assumes the same batch size and seed. Acceptable for dev, noted here.
3. Weight decay now excludes embeddings, LayerNorm parameters and biases (F9); this changes optimizer parameter groups, so pre-fix checkpoints cannot be resumed (model-only loading still works).
4. `train_tokenizer` logs `train_loss` as the last batch loss, not an interval mean. SPEC does not define this field.
5. `git_sha.txt` now ends in `-dirty` when the working tree has uncommitted changes.
6. T0.2 and T1.2 were ticked 2026-11-07 after the real-data runs above; the earlier "not run" entries are superseded.
7. `tok.sbatch` resumes from `results/<run_id>/ckpt/latest.pt`, which the trainer rewrites every `train.save_every` steps (5000 in the dev config), so a preempted job loses at most that many steps. Resume already does not restore the data iterator or RNG (item 2).
8. The working tree carries two uncommitted edits from an earlier session (`tests/test_review_fixes.py`, `tests/test_train_tokenizer.py` pass an explicit temporary `data_dir` to `train`). They are unrelated to T3.5 and were left uncommitted; the suite passes with them in place.
9. T3.5 acceptance is blocked for `regress` (Q8): level 1 is conditioned on `zhat_0 = 0` only, so the deterministic regression head predicts the batch mean level-1 code and cannot emit 8 distinct sequences. The arm is `xfail(strict=True)`; T3.5 stays un-ticked pending a researcher decision.
10. The slow acceptance takes about 2 minutes on CPU (mostly the flow arm), so `pytest -x -q` now takes about 2.5 minutes. `pytest -m "not slow"` is the fast loop.

## ASSUMPTIONs (spec ambiguities resolved; details in docs/QUESTIONS.md)
- Synthetic data is allowed only with `allow_synthetic=True` / `--allow_synthetic`; real runs fail loudly if `data/tinystories/{train,val}.bin` are missing.
- VQ `codes` remain straight-through vectors so decoder gradients reach the encoder; `targets` remain integer indices.
- Learned Q/K/V and output projections are part of the SPEC's pre-LN transformer block.
- SPEC v1.2 resolves the schedule tie-breaking ambiguity: exact half-up integer arithmetic.
- SPEC 5.1 says "64 log-spaced frequencies" without a range: SinPos uses 64 geometric frequencies from 1 to 1e-4 cycles per normalized position, applies 2*pi*p_i*freq, concatenates sin and cos (128 features), and projects linearly to H.
- SPEC 5.2's "dropout 0.1" is applied as residual dropout (after the attention output projection and after the MLP), as in the tokenizer blocks; attention-internal dropout is 0.
- SPEC 5.3 "3 residual blocks (Linear 512 -> 512, SiLU)" is implemented as `x + SiLU(Linear(x))`; the time embedding is 64 log-spaced frequencies from 1 to 1e-4 cycles over `2*pi*t` for t in [0, 1], sin and cos concatenated to 128 dims (same convention as the backbone positions).
- SPEC 5.3's regress "option `cosine`" is the config flag `generator.regress_cosine` (default false = MSE). `FlowHead` widths follow SPEC (hidden 512, 3 blocks, 128 time dims) and are constructor arguments only so tests can shrink them.
- Head `loss` returns per-level means plus their sum; SPEC 5.5 weighting is `weighted_total(losses, schedule, alpha)` = `sum_k w_k * l_k * mean_k`, exact for the SPEC formula with per-position losses.
- SPEC 5.4 corruption: `corrupt.mode="noise"` requires `quantizer="noisy"` in `corrupt_codes` (missing/other variant raises, so the vq baseline cannot silently get channel noise).
- SPEC 5.4 swap: a level with fewer than two (batch, position) slots (e.g. B=1, l_1=1) has no "other" position to draw from and is left unchanged. `linear_schedule(K=1)` returns `[first]`, since the single level is the first level.
- SPEC 5.6 does not say whether the sampler switches modules to `eval()`. `sample_tokens` leaves train/eval modes untouched (the caller decides) and runs under `torch.no_grad()`; all draws take one explicit `torch.Generator`, drawn per level in order, so a seed fixes the whole sample.
- Corruption is out-of-place and takes one explicit `torch.Generator` for all levels (draw order: per level, mask then replacement index), so a run is reproducible.
- Overfit tests use batch size 1: level 1 is built from `zhat_0 = 0`, so every row of a level-1 batch sees the identical input and a per-position head can emit only one level-1 prediction (same for the regression floor). A perfect fit is only meaningful with one sequence; this is a property of SPEC 5.1/5.3, not a test shortcut.

- T3.5 acceptance metric: `sample_tokens` draws independently per row, so with 8 fixed sequences each of the 8 samples is compared with its **best-matching** training sequence (token accuracy), and the test also requires >= 3 distinct matches so that a collapse onto one sequence cannot pass. This is the reading that makes the TASKS wording meaningful for a stochastic sampler; the 90/90/80 thresholds are unchanged (Q8 covers the regress blocker).
- SPEC 5.7's generator defaults live in `generator.train` (lr 3e-4 -> 3e-5, wd 0.1, warmup 500, batch 1024, 10k steps), separate from the tokenizer's SPEC 4.4 `train` block. `smoke.yaml` overrides them to batch 16 / 200 steps so CPU smoke runs stay short; tests override them further.
- `train_generator` reads `tok_positions` from the tokenizer checkpoint as the curve offset for SPEC 6.2; `gen_positions = step * batch * L` and `total_positions` is their sum. A resume keeps the checkpoint's `tok_positions` and recomputes `gen_positions` from the step (same batch size assumed, as in the tokenizer trainer).
- `train_generator` requires the head/variant pairing of SPEC 5.3 (`ce` with vq, `regress`/`flow` with noisy) and fails loudly otherwise; the config does not guess.
- Q9: T4.1 asks every template for the three `HF_*_OFFLINE` variables, but `prep.sbatch` exists to download TinyStories. `tok.sbatch`/`gen.sbatch` set all three; `prep.sbatch` sets none and says it needs Hub access. Smallest reversible choice; the submit forms and placeholder list above are the contract.

## Human must fill / do
- T0.2 and T1.2 are complete locally (2026-11-07); no action. `models/gpt2-large` and `data/tinystories` are present on this host.
- Before G0a/G0: copy `data/tinystories/` and `models/gpt2-large/` to FIR (rsync templates in the session report; T4.3 RUNBOOK will own them).
- Run Gate G0a on a cluster GPU: dev tokenizers (`vq` and `noisy`); record exact-token accuracy.
- **T4.1 placeholders — every one of them (nothing is guessed in the scripts):**
  - Account: pass `--account="$SBATCH_ACCOUNT"` on the sbatch command line (or let your submit wrapper read `$SLURM_ACCOUNT`); no account appears in any of the three files.
  - Partition: pass `--partition="$SBATCH_PARTITION"` on the sbatch command line; no partition appears in any file.
  - Wall clock: `#SBATCH --time=HH:MM:SS` is invalid on purpose in `prep.sbatch:5`, `tok.sbatch:6`, `gen.sbatch:6` (`TODO(placeholder)` at lines 6, 7, 7). Replace each with a real limit.
  - GPU type: `tok.sbatch` (`#SBATCH --gres=gpu:1`, line 5, TODO at line 8) and `gen.sbatch` (same, line 5, TODO at line 8) ask for one generic GPU. Add a type only if FIR requires one. `prep.sbatch` requests no GPU (CPU-only numpy/tokenizers) — that is not a missing placeholder.
  - Modules: no `module load` line exists (TODO at `prep.sbatch:7`, `tok.sbatch:10`, `gen.sbatch:10`). Add your site's.
  - Venv: all three source `.venv/bin/activate` at the repo root and exit 1 if it is absent (TODO at `prep.sbatch:9`, `tok.sbatch:12`, `gen.sbatch:12`); create it per "Environment" below or edit the path.
  - `prep.sbatch` must reach the Hugging Face Hub (internet-enabled partition or a login node): it deliberately does not set `HF_*_OFFLINE` (Q9). `tok.sbatch` and `gen.sbatch` set all three offline variables.
  - Submit forms (all from the repo root): `sbatch --account="$SBATCH_ACCOUNT" --partition="$SBATCH_PARTITION" scripts/slurm/prep.sbatch`; `... --export=ALL,QUANT=vq scripts/slurm/tok.sbatch` (then `QUANT=noisy`); `... --export=ALL,HEAD=ce,RUN_ID=gen_vq_ce_dev TOKENIZER=results/tok_vq_dev/ckpt/latest.pt scripts/slurm/gen.sbatch` (and `HEAD=regress|flow` with the `noisy` tokenizer).
  - Not yet provided: `scripts/slurm/eval.sbatch` (T4.2). These three scripts have no `--array` seed sweep; a run's seed comes from `seed:` in the config and eval-time seed sweeps arrive with T4.2.

## Environment
- Python / torch: Python 3.13.7, torch 2.15.0+cpu (CUDA unavailable on this host); numpy 2.5.4, pytest 9.2.0.
- Recreate on Windows (PowerShell), from repo root:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  .\.venv\Scripts\python.exe -m pip install -e .
  .\.venv\Scripts\pytest.exe -q
  ```
- Recreate on Linux/cluster:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install -e .
  pytest -q
  ```
- Data path: `data/tinystories/`   Models path: `models/`

## Next 3 steps
1. Human: copy `data/tinystories/` and `models/gpt2-large/` to FIR, then run Gate G0a (dev `vq` and `noisy` tokenizers) and G0.
2. T4.2 (`scripts/run_eval.py`, `scripts/plot_curves.py`, and `scripts/slurm/eval.sbatch`, which moved here from T4.1).
3. T4.3 (`RUNBOOK.md`): fill the wall-clock fields using the real limits chosen above.

## Results log (human-run gates; paste numbers from results/ only)
- T1.2 protocol gate (local, 2026-11-07): Gen PPL 10.80 +/- 0.26; H 4.03 +/- 0.01 (GPT-2 tokens) and 4.03 +/- 0.01 (dataset tokens); 256 samples x 5 seeds; verdict PASS; `results/t1_2_protocol_gate.json`.
- G0a: not run
- G0: not run
- G1: not run
- G2: not run
