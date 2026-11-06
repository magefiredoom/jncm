# STATE — jncm

Last updated: 2026-11-06 (T3.4 sampler done; 109 tests pass on CPU)

Rule: newest information wins. Rewrite sections; do not append history.

## Current position
- Milestone / task in progress: M0–M2 code complete; **T2.4, T3.1, T3.2, T3.3, T3.4 complete** (`jncm/generator/{backbone,heads,corruption,sample}.py`); **T4.1 partial** — `scripts/slurm/tok.sbatch` and `prep.sbatch` only, gen/eval wait for T3.5/T4.2.
- Last completed task: T3.4 (`sample_tokens` in `generator/sample.py`, SPEC 5.6; acceptance: exactly K backbone calls, shapes/token range, seeded determinism).
- Next task: T3.5 (`jncm/train_generator.py`, SPEC 5.5–5.7). Not started.
- T4.1 is deliberately **not** ticked: `scripts/slurm/{gen,eval}.sbatch` do not exist yet.
- T0.2 real data prep and T1.2 real-data protocol gate remain human-run; their CPU tests pass.
- Gate G0a is next after T1.2 (cluster, human).

## VERIFIED (command + last lines of output)

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

### T4.1 partial — sbatch syntax, trainer flags, suite
- `bash -n scripts/slurm/tok.sbatch; echo "tok.sbatch bash -n exit=$?"; bash -n scripts/slurm/prep.sbatch; echo "prep.sbatch bash -n exit=$?"`
  `tok.sbatch bash -n exit=0`
  `prep.sbatch bash -n exit=0`
- Full suite: `.venv/Scripts/python.exe -m pytest -q` → `64 passed in 9.69s`
- `python -m jncm.train_tokenizer --help` lists `--resume RESUME` and `--run_id RUN_ID`; loading `configs/tinystories_dev.yaml` with `--set tokenizer.quantizer=noisy` gives `quantizer 'noisy' L 128 V 8192 max_steps 40000 save_every 5000 device cuda`.
- Scripts checked by inspection, **not submitted**: one generic GPU (`--gres=gpu:1`), the three HF offline vars, `--run_id tok_${QUANT}_dev`, `--set tokenizer.quantizer=${QUANT}`, and `--resume results/<run_id>/ckpt/latest.pt` only when that file exists. No account or partition on any `#SBATCH` line.

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

## Files touched in T4.1 (partial)
- `scripts/slurm/tok.sbatch`, `scripts/slurm/prep.sbatch` (new), `docs/STATE.md`.

## Files touched in T2.4
- Code: `jncm/config.py`, `jncm/levels.py`, `jncm/tokenizer/vq.py`, `jncm/train_tokenizer.py`, `scripts/prep_tinystories.py`.
- Packaging: `pyproject.toml` (new), `.gitignore` (zip archives).
- Tests: `tests/test_review_fixes.py` (F5/F8 added), `tests/test_train_tokenizer.py` (F8 `results_dir`, F9 dirty/optimizer tests), `tests/test_data_prep.py` (F9 byte alphabet).
- Docs: `docs/STATE.md`, `docs/TASKS.md`, `docs/QUESTIONS.md` (SPEC.md not edited).

## UNVERIFIED / not run
- T0.2 real data prep (`python scripts/prep_tinystories.py --max_stories 2000 --out data/tiny_test`): needs Hugging Face network access; not run. Packing/streaming tests pass.
- T1.2 protocol gate on real data (`scripts/eval_data_row.py` with GPT-2-large): needs GPT-2-large weights and real TinyStories; not run. Mock test passes; `--help` verified.
- T3.5 generator training: not implemented (no `train_generator.py`). T3.1–T3.4 are verified only by unit tests on eval-mode modules fed by a frozen, untrained tokenizer; end-to-end generator quality (and the T3.5 memorization acceptance) is unverified. Real-run head hyperparameters (SPEC 5.7) are untested at 768 wide / 90M params.
- G0a/G0/G1/G2: human cluster gates, not run.
- If any F2 test passed before its fix: `F2[noisy]` passed. The vq crash was real (`index_add_` dtype); the fp32 quantizer change was kept for both variants.

## Known issues / deviations
1. `GPT2Scorer.score_sample` scores one text at a time; batching is deferred to T4.2 as the review suggests.
2. Resume restores model/optimizer but not the data iterator or RNG; reproducibility across resumes assumes the same batch size and seed. Acceptable for dev, noted here.
3. Weight decay now excludes embeddings, LayerNorm parameters and biases (F9); this changes optimizer parameter groups, so pre-fix checkpoints cannot be resumed (model-only loading still works).
4. `train_tokenizer` logs `train_loss` as the last batch loss, not an interval mean. SPEC does not define this field.
5. `git_sha.txt` now ends in `-dirty` when the working tree has uncommitted changes.
6. T0.2 and T1.2 remain unchecked in TASKS because their real-data acceptance runs have not happened.
7. `tok.sbatch` resumes from `results/<run_id>/ckpt/latest.pt`, which the trainer rewrites every `train.save_every` steps (5000 in the dev config), so a preempted job loses at most that many steps. Resume already does not restore the data iterator or RNG (item 2).
8. The working tree carries two uncommitted edits from an earlier session (`tests/test_review_fixes.py`, `tests/test_train_tokenizer.py` pass an explicit temporary `data_dir` to `train`). They are unrelated to T3.1 and were left uncommitted; the suite passes with them in place.

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

## Human must fill / do
- Run `python scripts/prep_tinystories.py --max_stories 2000 --out data/tiny_test` on a login node (internet) to satisfy T0.2.
- Run `python scripts/fetch_models.py` on a login node to download `gpt2-large` into `models/gpt2-large`.
- Run `python scripts/prep_tinystories.py` to prepare `data/tinystories/`.
- Run the T1.2 protocol gate on real validation data; record Gen PPL and H.
- Run Gate G0a on a cluster GPU: dev tokenizers (`vq` and `noisy`); record exact-token accuracy.
- **T4.1 placeholders (nothing is guessed in the scripts):**
  - Account: pass `--account="$SBATCH_ACCOUNT"` on the sbatch command line (or let your submit wrapper read `$SLURM_ACCOUNT`); no account appears in either file.
  - Partition: pass `--partition="$SBATCH_PARTITION"` on the sbatch command line; no partition appears in either file.
  - Wall clock: `#SBATCH --time=HH:MM:SS` in both scripts is invalid on purpose (`TODO(placeholder)` at `tok.sbatch:7`, `prep.sbatch:6`). Replace it.
  - GPU type: `tok.sbatch` asks for one generic GPU (`--gres=gpu:1`, TODO at line 8). Add a type only if FIR requires one.
  - Modules: no `module load` line exists (TODO at `tok.sbatch:10`, `prep.sbatch:7`). Add your site's.
  - Venv: both scripts source `.venv/bin/activate` at the repo root (TODO at `tok.sbatch:12`, `prep.sbatch:9`); create it per "Environment" below or edit the path.
  - `prep.sbatch` must reach the Hugging Face Hub (internet-enabled partition or a login node): it deliberately does not set `HF_*_OFFLINE`.

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
1. T3.5 (`jncm/train_generator.py`: frozen tokenizer, all levels per pass, sampling via `sample_tokens` + Gen PPL hook, position accounting, resume; call `corrupt_codes(codes, cfg.generator.corrupt, generator=..., quantizer=cfg.tokenizer.quantizer)`). Remember to call `tokenizer.eval()`, `backbone.eval()`, `head.eval()` before sampling.
2. Finish T4.1 (`scripts/slurm/{gen,eval}.sbatch`) once T3.5/T4.2 exist.
3. T4.2 (`scripts/run_eval.py`, `scripts/plot_curves.py`).

## Results log (human-run gates; paste numbers from results/ only)
- G0a: not run
- G0: not run
- G1: not run
- G2: not run
