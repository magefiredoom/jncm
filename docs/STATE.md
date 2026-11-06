# STATE — jncm

Last updated: 2026-11-06 (T4.1 partial: tok.sbatch + prep.sbatch; 64 tests pass on CPU)

Rule: newest information wins. Rewrite sections; do not append history.

## Current position
- Milestone / task in progress: M0–M2 code complete; **T2.4 complete**; **T4.1 partial** — `scripts/slurm/tok.sbatch` and `prep.sbatch` only, gen/eval wait for T3.5/T4.2. M3 (generator) not started.
- Last completed task: T2.4 (review findings F1–F10, tests first, one commit per finding).
- T4.1 is deliberately **not** ticked: `scripts/slurm/{gen,eval}.sbatch` do not exist yet.
- T0.2 real data prep and T1.2 real-data protocol gate remain human-run; their CPU tests pass.
- Gate G0a is next after T1.2; do not start T3.1 until instructed.

## VERIFIED (command + last lines of output)

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
- T3.1–T3.5 generator: not implemented. Do not treat generator behavior as verified.
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

## ASSUMPTIONs (spec ambiguities resolved; details in docs/QUESTIONS.md)
- Synthetic data is allowed only with `allow_synthetic=True` / `--allow_synthetic`; real runs fail loudly if `data/tinystories/{train,val}.bin` are missing.
- VQ `codes` remain straight-through vectors so decoder gradients reach the encoder; `targets` remain integer indices.
- Learned Q/K/V and output projections are part of the SPEC's pre-LN transformer block.
- SPEC v1.2 resolves the schedule tie-breaking ambiguity: exact half-up integer arithmetic.

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
1. T3.1 (`jncm/generator/backbone.py`: level-block transformer with no-leak guarantee). Do not start until instructed.
2. T3.2 (`jncm/generator/heads.py`: ce, regress, and flow matching heads).
3. T3.3 (`jncm/generator/corruption.py`: none/swap/noise per SPEC 5.4).

## Results log (human-run gates; paste numbers from results/ only)
- G0a: not run
- G0: not run
- G1: not run
- G2: not run
