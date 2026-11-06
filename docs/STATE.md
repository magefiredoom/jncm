# STATE — jncm

Last updated: 2026-11-06 (verification + conformance pass; 49 tests pass on CPU)

Rule: newest information wins. Rewrite sections; do not append history.

## Current position
- Milestone / task in progress: M2 complete and VERIFIED (T2.1, T2.2, T2.3). M3 (generator) not started.
- Last completed task: T2.3 (Tokenizer training loop, checkpointing, diagnostics, and tests).
- Conformance pass done for M0–M2. Generator-side checks are N/A because `jncm/generator/` does not exist yet.

## VERIFIED (command + last lines of output)
- `.\.venv\Scripts\pytest.exe -x -q` (from repo root, CPU):
  `.................................................`  
  `49 passed in 6.23s`
- `.\.venv\Scripts\python.exe -m jncm.train_tokenizer --config configs/smoke.yaml --steps 50 --run_id tok_vq_verified`:
  `Step    50 | Loss: 4.0646 | acc_full: 0.0703 | tok_pos: 25600`  
  `Training completed. Results written to results\tok_vq_verified`  
  Last record contains `acc_cum`, `acc_full`, `tok_positions=25600`, `gen_positions=0`, `total_positions=25600`, `nominal_bits=6.0`, `codebook_usage`, `codebook_perplexity`.
- `.\.venv\Scripts\python.exe -m jncm.train_tokenizer --config configs/smoke.yaml --steps 50 --set tokenizer.quantizer=noisy --run_id tok_noisy_verified`:
  `Step    50 | Loss: 4.1415 | acc_full: 0.0352 | tok_pos: 25600`  
  `Training completed. Results written to results\tok_noisy_verified`  
  Last record contains `std_min`, `std_mean`, `eff_rank`, `nominal_bits=11.4319`, and the same position accounting.
- Checkpoint round-trip (fresh model loads `results/tok_vq_verified/ckpt/latest.pt` with `strict=True`):
  `step 49 tok_positions 25600`  
  `missing [] unexpected []`
- `test_tokenizer_checkpoint_roundtrip` (vq and noisy; state_dict → disk → fresh model → identical logits) passes.
- Position accounting: all 38 pre-existing metrics records and all new records satisfy `tok_positions + gen_positions == total_positions`.
- Conformance probes: `reconstruct(enc.codes)` matches `enc.zhat` with max abs diff `0.0` for both variants; noisy `codes - targets` std `0.409` (sigma 0.4), clean-code per-position variance `0.9999995`; VQ `targets` dtype `int64`, `codes` float vectors.

## UNVERIFIED / not run
- T0.2 real data prep (`python scripts/prep_tinystories.py --max_stories 2000 --out data/tiny_test`): needs Hugging Face network access; not run here. `pack_stories` unit tests pass.
- T1.2 protocol gate on real data (`scripts/eval_data_row.py` with GPT-2-large): needs GPT-2-large weights and real TinyStories; not run. Mock-based test passes.
- T3.1–T3.5 generator: not implemented in this draft. Checks (a) no-leak, (b) nested dropout vs generator targets, (c) generator targets/chain codes, and (f) generator checkpoint round-trip are N/A, not verified.
- G0a/G0/G1/G2: human cluster gates, not run.

## Known issues / bugs / deviations (fixed or not)
Fixed in this session:
1. `pytest.ini` was in `configs/` instead of the repo root, so `pytest -x -q` failed with `ModuleNotFoundError: No module named 'jncm'`. Moved to root and added `pythonpath = .`.
2. `TransformerBlock` used the same tensor as Q, K, V and had no output projection. A pre-LN transformer (SPEC 4.1/5.2) needs learned projections; added `qkv = Linear(d, 3d)` and `out_proj = Linear(d, d)`.
3. `nominal_bits` was logged only for `noisy`; SPEC 4.3 requires `bits_per_position(cfg)` for both variants. Now logged in every tokenizer diagnostics record.
4. `requirements.txt` was under `configs/`; moved to repo root per T0.1/AGENTS layout.
5. `setup_data` silently substituted synthetic random data when prepared memmaps were missing. Added `train.allow_synthetic_data` (default false; true only in `smoke.yaml`); real runs now raise `FileNotFoundError`.
6. `meta.json` wrote the requested vocab size instead of the trained tokenizer's actual vocab size; now uses `tokenizer.get_vocab_size()`.
7. Added the missing checkpoint round-trip test (T2.2(f)), explicit `tok+gen=total` assertions, and cutoff-range validation.

Open deviations / decisions:
1. VQ `Encoded.codes` are straight-through float vectors, not indices. SPEC 4.1's API comment says “vq: same as targets”, but indices would break the straight-through gradient from decoder CE into the encoder; `targets` are indices as required and `reconstruct(enc.codes)` is bit-exact.
2. `get_level_schedule` uses Python `round` (banker's rounding). SPEC 2 only says `round`; the specified dev/smoke schedules are unaffected. Example ambiguity: L=64, K=5 gives l_3=32 (half-up would give 33).
3. `train_tokenizer` logs `train_loss` as the last batch loss, not an interval mean. SPEC does not define this field.
4. `train_tokenizer` recomputes `tok_positions` from `(step+1) * batch_size * L`; resuming with a different batch size would mis-account. The tested resume path uses the same batch size.
5. No generator module; M3 is the next milestone. Do not treat any generator behavior as verified.

## ASSUMPTIONs (spec ambiguities I resolved; each also in docs/QUESTIONS.md)
- Learned Q/K/V and output projections are part of the SPEC's “pre-LN transformer block” even though the block diagram does not spell them out.
- VQ chain codes stay as straight-through vectors so decoder gradients reach the encoder; `targets` remain the integer indices.
- Synthetic data is allowed only when `train.allow_synthetic_data=true`; real runs must fail loudly if `data/tinystories/{train,val}.bin` are missing.
- Python's banker's rounding is an acceptable reading of the schedule's `round`.

## Human must fill / do
- Run `python scripts/prep_tinystories.py --max_stories 2000 --out data/tiny_test` on a login node (internet) to satisfy T0.2.
- Run `python scripts/fetch_models.py` on a login node to download `gpt2-large` into `models/gpt2-large`.
- Run `python scripts/prep_tinystories.py` to prepare `data/tinystories/`.
- Run the T1.2 protocol gate on real validation data; record Gen PPL and H.
- Run Gate G0a on cluster GPU: dev tokenizers (`vq` and `noisy`); record exact-token accuracy.
- Fill Slurm account and environment details.
- Review the new QKV projections: checkpoints produced by the Gemini draft before this session are architecture-incompatible.

## Environment
- Python / torch: Python 3.13.7, torch 2.15.0+cpu (CUDA unavailable on this host); numpy 2.5.4, pytest 9.2.0.
- Recreate on Windows (PowerShell), from repo root:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  .\.venv\Scripts\python.exe -m pytest -x -q
  ```
- Recreate on Linux/cluster:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pytest -x -q
  ```
- Data path: `data/tinystories/`   Models path: `models/`

## Next 3 steps
1. T3.1 (`jncm/generator/backbone.py`: level-block transformer with no-leak guarantee).
2. T3.2 (`jncm/generator/heads.py`: ce, regress, and flow matching heads).
3. T3.3 (`jncm/generator/corruption.py`: none/swap/noise per SPEC 5.4).

## Results log (human-run gates; paste numbers from results/ only)
- G0a: not run
- G0: not run
- G1: not run
- G2: not run
