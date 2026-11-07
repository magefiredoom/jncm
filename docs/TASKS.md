# TASKS — jncm

Legend: `[ ]` todo, `[x]` done. Tick a box only after its acceptance command passed in the current session and the output tail is pasted into docs/STATE.md.
Work on the first unchecked task. One task per session unless told otherwise.
"Gate" items are run by the human on the cluster; agents prepare them but do not run long jobs.

## M0 — Scaffold and data

- [x] **T0.1 Skeleton.** Package `jncm/` with `config.py` (dataclasses, YAML loader, `--set a.b=c` overrides), `utils.py` (seeding, JSONL logger, `bits_per_position`), `configs/smoke.yaml`, `configs/tinystories_dev.yaml`, `requirements.txt`, `pytest.ini`, `.gitignore` (data/, results/, models/, *.pt).
  Accept: `pytest -q` passes, including a config round-trip test (YAML -> dataclass -> YAML) and an override test. — PASSED 2026-11-06 (bare `pytest -q`: 64 passed; root `pytest.ini`, `requirements.txt`, `pyproject.toml`).
- [x] **T0.2 Data prep script.** `scripts/prep_tinystories.py` per SPEC section 3, with `--max_stories` for tests.
  Accept: `python scripts/prep_tinystories.py --max_stories 2000 --out data/tiny_test` writes `train.bin`, `val.bin`, `meta.json`, `tokenizer.json` (needs internet; run on a login node or laptop). A test with a tiny local text fixture checks packing: windows are exactly L tokens, `<eos>` separates stories, nothing overlaps. — PASSED 2026-11-07 (`data/tinystories`: train 3,647,424 / val 36,658 windows, L 128, V 8192, bin sizes 933,740,544 / 9,384,448 bytes match meta; max token id 8191 in both bins, first/last 1e6 tokens; two val windows decode to English; `python scripts/prep_tinystories.py --max_stories 2000 --out data/tiny_test` -> 3,210 train / 637 val windows, sizes match; `pytest -q tests/test_data_prep.py tests/test_eval_data_row.py tests/test_genppl.py` -> 13 passed).
- [x] **T0.3 Data loader.** `jncm/data.py`: memmap dataset and infinite seeded batch iterator returning LongTensor [B, L]; helper that counts positions seen.
  Accept: tests for shapes, determinism under a fixed seed, and train/val disjointness. — PASSED 2026-11-06.

## M1 — Evaluation harness

- [x] **T1.1 Gen PPL and entropy.** `jncm/eval/genppl.py`: `gen_ppl(texts, scorer)`, `unigram_entropy(token_id_lists)`, `GPT2Scorer` (local path from `$GPT2_LARGE_PATH`, offline), plus `scripts/fetch_models.py` (login node) to download gpt2-large.
  Accept: tests use a fake scorer (no download). Check entropy of a uniform distribution over n distinct tokens equals ln n, and a perplexity case with known NLLs. — PASSED 2026-11-06 (fake-scorer tests; real GPT-2 path unverified).
- [x] **T1.2 Protocol gate on real data.** `scripts/eval_data_row.py` decodes N real validation sequences and reports Gen PPL and H.
  Accept (human runs it): about 9.67 PPL and about 3.96 H (SPEC 6.1). Record the numbers in STATE.md. If outside tolerance, write a diagnosis in docs/QUESTIONS.md; do not tune the metric toward the numbers. — PASSED 2026-11-07 (`GPT2_LARGE_PATH=models/gpt2-large python scripts/eval_data_row.py --data data/tinystories/val.bin --tokenizer data/tinystories/tokenizer.json --gpt2 models/gpt2-large --n_samples 256 --out results/t1_2_protocol_gate.json`: Gen PPL 10.80 +/- 0.26 in [8.22, 11.12]; H GPT-2 tokens 4.03 +/- 0.01 and dataset tokens 4.03 +/- 0.01 in [3.81, 4.11]; script verdict PASS).

## M2 — Tokenizer

- [x] **T2.1 Levels.** `jncm/levels.py`: schedule, `down(x, l)`, `up(x, L)`.
  Accept: tests for schedule properties (several L, K including smoke and dev), `down(x, L)` and `up(x, L)` are identities, `down(x, 1)` equals the mean over positions, output shapes. — PASSED 2026-11-06 (exact half-up schedule).
- [x] **T2.2 Tokenizer.** `jncm/tokenizer/` (trunk, `vq`, `noisy`, decoder) per SPEC section 4.
  Accept tests (CPU, smoke config):
  (a) output shapes for both variants;
  (b) chain identity: `zhat[k] - zhat[k-1]` equals the level-k contribution, and `z - zhat[K]` equals the final residual;
  (c) `reconstruct(enc.codes)` reproduces `enc.zhat` exactly;
  (d) with cutoff k the decoder input is `zhat[k]`;
  (e) nested-dropout sampler: empirical cutoff frequencies match p_drop;
  (f) vq: EMA update changes codebooks; dead-code reinit triggers on an engineered dead code;
  (g) noisy: every clean code c_k has unit power per position (LayerNorm), and the empirical std of `codes - targets` matches sigma_k;
  (h) overfit-a-batch: both variants reach at least 99% token accuracy on 16 fixed sequences within 600 CPU steps (adjust lr and steps, never the threshold). — PASSED 2026-11-06 (a–h plus checkpoint round-trip; full suite 64 passed).
- [x] **T2.3 Tokenizer training.** `jncm/train_tokenizer.py` per SPEC 4.4 and 4.5: AdamW with cosine schedule, bf16 on CUDA, checkpoint and resume, diagnostics every `eval_every`, position accounting.
  Accept: `python -m jncm.train_tokenizer --config configs/smoke.yaml --steps 50 --allow_synthetic` finishes on CPU and writes `metrics.jsonl` containing `acc_cum`, `acc_full`, `tok_positions`, and the variant-specific diagnostics. — PASSED 2026-11-06 (`results/tok_vq_t24`; 50 steps, 25600 tok positions, `data_source=synthetic`).
- [x] **T2.4 Review fixes.** Findings F1 to F10 in `docs/REVIEW_M0-M2.md`, tests first (`tests/test_review_fixes.py`), one commit per finding.
  Accept: every test in `tests/test_review_fixes.py` failed before its fix and passes after; bare `pytest -q` passes from the repo root (pytest.ini and requirements.txt at the root, `pip install -e .` done); `python scripts/eval_data_row.py --help` works; the smoke command above runs on CPU. Do T2.4 before Gate G0a. — PASSED 2026-11-06 (12 review tests pass, full suite 64; baseline: F1, F2[noisy], F4 no-silent and F7 already passed; F2[vq], F3, F4 remainder, F6, F10 failed first).
- **Gate G0a (human, GPU), after T2.4 and T0.2** (T1.2 is only needed before G0). TinyStories dev tokenizers: vq reaches >= 99.9% exact-token accuracy on validation; noisy reaches >= 99.5% (record the number; if lower, report per-level `acc_cum` before changing d_k or sigma_k). `acc_cum` should rise gradually.

## M3 — Generator

- [x] **T3.1 Backbone.** Level-block transformer per SPEC 5.1 and 5.2.
  Accept tests: (a) **no leak**: with `eval()` and a fixed seed, two inputs that differ only in codes of levels >= k give identical level-k outputs; (b) padding invariance: changing values at padded positions does not change valid outputs; (c) shapes. — PASSED 2026-11-06 (`tests/test_generator_backbone.py`: 9 tests, both `vq` and `noisy`; leak and dropped-mask mutants each fail the suite; full suite 73 passed).
- [x] **T3.2 Heads and losses.** `ce`, `regress`, `flow` per SPEC 5.3.
  Accept tests: ce and regress overfit a fixed batch; flow learns a 2-D toy (mixture of two Gaussians at +-1, std 0.1) and its samples are bimodal (at least 35% in each mode); seeded determinism. — PASSED 2026-11-06 (`tests/test_generator_heads.py`: 13 tests; ce fits 66/66 indices, regress MSE 4.6e-4, toy 0.45/0.55 split with 97% near a mode; full suite 86 passed).
- [x] **T3.3 Corruption.** `generator/corruption.py` per SPEC 5.4.
  Accept tests: empirical swap rate matches p_j within 0.03; targets are untouched; `none` is the identity. — PASSED 2026-11-06 (`tests/test_generator_corruption.py`: 9 tests; observed 0.816/0.601/0.396/0.196 vs 0.8/0.6/0.4/0.2, noise std 0.500/0.366/0.234/0.100 vs 0.5/0.3667/0.2333/0.1; full suite 95 passed).
- [x] **T3.4 Sampler.** `generator/sample.py` per SPEC 5.6.
  Accept tests: exactly K backbone calls (forward hook), correct shapes, token ids in [0, V), same seed gives same samples. — PASSED 2026-11-06 (`tests/test_generator_sample.py`: 14 tests; hook count K at B=1/3 and K=1/4, ids in range for ce/regress/flow, seeded determinism, sampler equals the manual SPEC 5.6 loop; mutant probes (no accumulation, extra backbone call) each kill 4 tests; full suite 109 passed).
- [x] **T3.5 Generator training.** `jncm/train_generator.py` per SPEC 5.5 to 5.7: frozen tokenizer checkpoint, all levels in one pass, periodic sampling plus Gen PPL and diversity hook (SPEC 6.1), position accounting, resume.
  Accept (`pytest -m slow`, CPU): memorization tests. Overfit the tokenizer on 8 fixed sequences, then the generator. `ce`: sampled outputs match the training sequences at >= 90% token accuracy. `flow`: >= 80%. `regress`: (a) memorizes ONE fixed sequence at >= 90%; (b) with the true level-1 code supplied (teacher-forced), sampling levels 2..K reproduces all 8 sequences at >= 90%. Reproducing 8 different sequences from an unconditional level 1 is not a pass criterion for a deterministic head (SPEC 5.3, Q8). Unit tests for `unique_frac` and `pairwise_agree` (identical samples give 1/n and 1; fully different samples give 1 and 0). — PASSED 2026-11-07 (fast 13 passed; slow arms all pass: `ce` and `flow` 8 sequences, `regress` one sequence at 1000 steps, `regress` teacher-forced 8 sequences at 2500 steps; diversity metrics added with endpoint tests; full suite 126 passed). Q8 resolved: SPEC 5.1 and the regress head are unchanged.

## M4 — Experiment infrastructure

- [x] **T4.1 Slurm templates.** `scripts/slurm/{prep,tok,gen,eval}.sbatch`: take the account and partition from `SBATCH_ACCOUNT` / `SBATCH_PARTITION` or the sbatch command line (never from a guess), set the offline environment variables (`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`) in every script except `prep.sbatch`, which must reach the Hub (Q9), venv activation, `--time` and GPU request as clearly marked placeholders, array jobs for seeds. TODO markers are allowed only here; list every one under "Human must fill" in STATE.md. — PASSED 2026-11-06 for `prep.sbatch`, `tok.sbatch`, `gen.sbatch` (each `bash -n` exit 0; full suite 123 passed + 1 xfailed; no account/partition/GPU type in any file; QUANT and HEAD+TOKENIZER+RUN_ID come from `--export=ALL`; resume only when `results/<run_id>/ckpt/latest.pt` exists; every TODO listed under "Human must fill"). **`eval.sbatch` moved to T4.2** because it evaluates checkpoints written by `run_eval.py`, and `prep.sbatch` deliberately omits `HF_*_OFFLINE` because it downloads data (Q9). Seed `--array` sweeps move to T4.2 with `eval.sbatch`.
- [x] **T4.2 Eval and plots.** `scripts/run_eval.py` (evaluate generator checkpoints at 10/25/50/100% of training, write results JSON), `scripts/plot_curves.py` (Gen PPL against `total_positions`, with an entropy panel; one line per run; reads `results/**/eval/*.json`), plus `scripts/slurm/eval.sbatch` (moved here from T4.1) with seed array jobs. — PASSED 2026-11-07 (batched padding-safe `GPT2Scorer.score_batch`, real gpt2-large mean-NLL diff 4.4e-7 vs `score_sample`; `run_eval.py` reports Gen PPL mean/std, H on both token variants, `unique_frac`, `pairwise_agree` and `total_positions` per milestone checkpoint, with `ckpt/milestone_<pct>.pt` saved behind `generator.train.save_milestones` and tested; `plot_curves.py` reads `results/**/eval/step_<n>.json` and writes a 3-panel PNG, tested on fixtures; `eval.sbatch` has a seed `--array`, no account/partition, offline variables, `bash -n` exit 0 on all four templates; focused 27 passed, full suite 147 passed; end-to-end CLI smoke wrote eval JSON and a PNG on a throwaway run).
- [ ] **T4.3 Runbook.** `RUNBOOK.md` with exact commands for Phase 0 (vq + ce) and Phase 1 (noisy + regress, noisy + flow), with blank fields for wall-clock.
- **Gate G0 (human; needs T1.2 passed).** vq + ce on TinyStories dev budget: Gen PPL <= 20 with H >= 3.8 (NCM's ablation runs at this budget landed between 15.4 and 19.6).
- **Gate G1 (human).** noisy + flow within 15% of the G0 Gen PPL at equal `total_positions`, with entropy within 0.15 of data. Also report noisy + regress.

## M5 — JEPA joint mode (SPEC section 8)

- [ ] **T5.1 EMA.** `jncm/jepa/ema.py`. Tests: update numerics against a hand-computed value; target has no grad; target is in eval mode; momentum schedule endpoints.
- [ ] **T5.2 VICReg.** `jncm/jepa/vicreg.py`. Tests: iid N(0,I) gives var term about 0 and small cov; a constant gives var = gamma; correlated dimensions give cov > 0.
- [ ] **T5.3 Joint trainer.** `jncm/jepa/joint.py`, `jncm/train_jepa.py`, all flags, diagnostics (std, effective rank, R2 per level, collapse alarm).
  Accept: smoke overfit test; each flag combination of the 2x2 runs 50 CPU steps without error.
- [ ] **T5.4 Ablation configs and runbook section.**
- **Gate G2 (human).** JEPA variant matches or beats the best Phase-1 variant at equal `total_positions`. If not, the paper becomes discrete versus continuous generation, with JEPA as a reported negative result.

## M6 — Later (spec to be written after G1/G2)

Unconditional prior on level 1 and LM1B (bert-base-uncased, L = 128, [CLS] separators); Sudoku; XSum / WMT14 prefix conditioning; MBR decoding. Do not start M6 from this file.