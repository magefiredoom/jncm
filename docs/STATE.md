# STATE — jncm

Last updated: 2026-11-04 (Sandbox check turn: 5 torch-free functions verified in sandbox)
Rule: newest information wins. Rewrite sections; do not append history.

## Current position
- Milestone / task in progress: M2 complete (T2.1, T2.2, T2.3 implemented; Gate G0a is next human cluster gate)
- Last completed task: T2.3 (Tokenizer training loop, checkpointing, diagnostics, and tests)

## VERIFIED (command + last lines of output)
- `pytest -q`: 41 passed on CPU in turn 2 session.

## UNVERIFIED (written but not run; list SANDBOX-CHECKED items here with their output)
- SANDBOX-CHECKED:
  1. `get_level_schedule` (`jncm/levels.py`):
     - L=128, K=12 -> `[1, 13, 24, 36, 47, 59, 70, 82, 93, 105, 116, 128]` (sum: 774)
     - L=32, K=4 -> `[1, 11, 22, 32]` (sum: 66)
     - Output: strictly increasing, first = 1, last = L.
  2. `pack_stories` (`scripts/prep_tinystories.py`):
     - Output: windows shape `(2, 4)`, dtype `uint16`, contiguous windows `[[10, 11, 1, 20], [21, 22, 1, 30]]`, dropped remainder `[1]`, exact match with hand calculation.
  3. `unigram_entropy` (`jncm/eval/genppl.py`):
     - Output: uniform over n distinct tokens matches `ln(n)` to machine epsilon ($n=2, 4, 8, 16, 64$), constant sequence gives `0.0`.
  4. `gen_ppl` (`jncm/eval/genppl.py`):
     - Output: constant token NLL $c \in \{0.5, 1.0, 2.27, 3.5\}$ gives exactly $\exp(c)$ (diff `0.0e+00`).
  5. `bits_per_position` (`jncm/utils.py`):
     - Output: noisy ($d_k=8, \sigma_k=0.4$) gives `11.4319` bits (spec: 11.43); vq ($V_k=4096$) gives `12.0000` bits (spec: 12.0).
- UNVERIFIED:
  - T2.2 fixes in `jncm/tokenizer/vq.py` (added optional `threshold` keyword arg to `reinit_dead_codes`) and `jncm/tokenizer/noisy.py` (`eps=1e-8` for LayerNorm unit variance)
  - T2.3 `jncm/train_tokenizer.py` and `tests/test_train_tokenizer.py`

## Known issues / bugs
- none

## ASSUMPTIONs (spec ambiguities I resolved; each also in docs/QUESTIONS.md)
- none

## Human must fill / do
- Run `pytest -q` on CPU to confirm all 43 tests pass with T2.2 fixes and T2.3 tests.
- Run acceptance command: `python -m jncm.train_tokenizer --config configs/smoke.yaml --steps 50` on CPU.
- Run `python scripts/fetch_models.py` on a login node (internet access) to download `gpt2-large` into `models/gpt2-large`.
- Run `python scripts/prep_tinystories.py` on a login node to prepare `data/tinystories/`.
- Run Gate G0a on cluster GPU: train dev tokenizers (`vq` and `noisy`) and record exact-token accuracy on validation.
- Fill Slurm account and environment details.

## Environment
- Python / torch versions: PyTorch 2.x, Python 3.13.7 (Windows test host detected)
- How to activate the venv: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux)
- Data path: `data/tinystories/`   Models path: `models/`

## Next 3 steps
1. Gate G0a (Human runs dev tokenizers on GPU cluster: validation accuracy >= 99.9% for VQ, >= 99.5% for noisy)
2. T3.1 (`jncm/generator/backbone.py`: level-block transformer with no-leak guarantee)
3. T3.2 (`jncm/generator/heads.py`: ce, regress, and flow matching heads)

## Results log (human-run gates; paste numbers from results/ only)
- G0a: not run
- G0: not run
- G1: not run
- G2: not run
