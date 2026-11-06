# QUESTIONS — jncm

Record of questions, ambiguities, and assumptions. Spec decisions belong to the researcher; do not change anything marked [DECISION].

## Status
SPEC v1.2 resolves the level-schedule tie ambiguity. F1–F10 review fixes are implemented and tested as of 2026-11-06. No test needed a finding-unrelated change (no test was edited to work around a different kwarg name or API).

## Q1 — VQ `Encoded.codes`: indices or straight-through vectors? (resolved)
SPEC 4.1 (v1.1+) says `codes` are "the codes actually used in the residual chain" and explicitly documents vq codes as straight-through code vectors; `targets` are the integer indices. The generator conditions on `targets`.
**Decision:** keep `codes` as straight-through float vectors so decoder CE gradients reach the encoder; keep `targets` as integer indices. `reconstruct(enc.codes)` and `reconstruct(enc.targets)` both reproduce `enc.zhat`; the F7 test uses `targets`, exactly as the generator will.

## Q2 — Rounding ties in the level schedule (resolved)
SPEC v1.2 defines `l_k = 1 + (2*(k-1)*(L-1) + (K-1)) // (2*(K-1))`, i.e. exact half-up integer rounding, not Python's banker's rounding.
**Decision:** implemented the exact integer formula (F10). Tests cover `(4, 3) == [1, 3, 4]` and `(128, 11)[5] == 65`.

## Q3 — Q/K/V projections in the "pre-LN transformer" (resolved)
SPEC 4.1/5.2 say "pre-LN transformer block" and now explicitly require learned q, k, v and output projections (`12d^2 + 13d` parameters). The draft's parameter-free attention was a bug.
**Decision:** keep the standard learned `qkv = Linear(d, 3d)` and `out_proj = Linear(d, d)` (F1). All pre-QKV checkpoints are deleted.

## Q4 — Synthetic data fallback (resolved)
SPEC 3 forbids a silent fallback. `setup_data` now raises `FileNotFoundError` unless real memmaps exist or synthetic data is explicitly allowed; `meta.json` L/V mismatches raise `ValueError`; every metrics record carries `data_source`.
**Decision:** use `allow_synthetic=True` / `--allow_synthetic` only for smoke tests; `smoke.yaml` keeps `train.allow_synthetic_data: true`.

## Q6 — T3.2 head details SPEC 5.3 leaves open (ASSUMPTION, non-blocking)
These are recorded as ASSUMPTIONs in STATE.md; none changes a SPEC decision.
- Flow residual block: SPEC says "3 residual blocks (Linear 512 -> 512, SiLU)". Implemented as `x + SiLU(Linear(x))` (one linear per block, nonlinear, residual). A two-linear bottleneck/pre-LN form is the alternative if the researcher prefers.
- Time embedding: "sinusoidal, 128 dims" without a range. Implemented as 64 log-spaced frequencies from 1 to 1e-4 cycles over `2*pi*t`, t in [0, 1], sin and cos concatenated — the same convention as the backbone positions (5.1).
- Regress `cosine` option: SPEC names the option but not a config key. Added `generator.regress_cosine: false` (false = MSE).
- Loss granularity: each head's `loss` returns per-level means; SPEC 5.5's `sum_k w_k * sum_i loss_{k,i}` is `weighted_total` = `sum_k w_k * l_k * mean_k`, which is exact for per-position losses (`alpha=0` gives the position-weighted mean, `alpha=1` the equal-per-level mean; both tested).
- Overfit acceptance batches use B = 1: at level 1 the input comes from `zhat_0 = 0`, so all rows of a level-1 batch are identical and a per-position head can predict only one level-1 code (likewise the regression floor is the variance of the level-1 targets). A 100% batch fit requires one sequence; that is a property of SPEC 5.1/5.3, not a weakened test.

## Q7 — Sampler module mode and no_grad (ASSUMPTION, non-blocking)
SPEC 5.6 gives the sampling loop but not the module state it runs in. `sample_tokens` leaves train/eval modes untouched (a sampler that flips modes would surprise a training loop) and wraps the loop in `torch.no_grad()`; callers that care about dropout/init-noise semantics must call `.eval()` first. All randomness (head sampling, flow start, channel noise) consumes one explicit `torch.Generator`, per level in order, so `Generator().manual_seed(s)` reproduces a whole sample. Recorded as an ASSUMPTION in STATE.md; SPEC.md not edited.

## Q8 — T3.5 memorization acceptance is unreachable for `regress` (OPEN, blocks ticking T3.5)
TASKS T3.5: "Overfit the tokenizer on 8 fixed sequences, then the generator; sampled outputs match the training sequences at >= 90% token accuracy for `ce` and `regress`, >= 80% for `flow`."

Measured on CPU smoke scale (8 fixed sequences, L=32, K=4, H=64/2 layers, `corrupt.mode=none`, 12 fast + 3 slow tests in `tests/test_train_generator.py`): `ce` 1.000, `flow` 0.961, `regress` 0.363 (plateau at 200/400/600/800/1000 steps: 0.24/0.34/0.33/0.36/0.36).

Why `regress` cannot reach it, with the numbers from this session:
- SPEC 5.1 makes the level-1 input depend on `zhat_0 = 0` only, so every sequence in a batch has the same level-1 hidden state (1 position); the only sequence-identifying information the generator can ever carry is the level-1 code it *samples*.
- The tokenizer's level-1 codes for the 8 sequences are 8 distinct values (vq ids 31, 12, 38, 10, 50, 6, 27, 43; noisy clean codes pairwise 3.72 apart), so the sequences really do differ at level 1.
- `RegressHead` is deterministic plus iid channel noise, so its MSE optimum is the batch mean level-1 code: the trained prediction sits 0.01 from the mean and is identical for all 8 rows. One sampled sequence only, or a blend; never 8 distinct sequences. Teacher-forced decoding with the true chain codes is 1.000, so this is the head/conditioning, not the tokenizer or the decoder.
- `ce` escapes by categorical sampling (a discrete id) and `flow` by sampling a multimodal level-1 distribution. That is precisely the discrete-vs-continuous distinction the project studies.
- A swap-corrupted run is worse for memorization (`ce` 1.000 -> 0.277), because swapping at level 1 replaces the only identifying code.

Current state: the test keeps the TASKS thresholds unchanged; the `regress` arm is `xfail(strict=True)` with this reason, so it runs, reports, and fails loudly if the architecture changes. TASKS T3.5 is **not** ticked.

Options for the researcher (pick one; none is implemented):
1. Run the `regress` memorization arm with **one** sequence (the level-1 ambiguity disappears) and keep 8 sequences for `ce`/`flow`.
2. Change the protocol so the level-1 code is teacher-forced for the continuous heads, i.e. measure memorization of levels 2..K given the true level 1.
3. Amend SPEC 5.1 to feed level 1 some sequence-identifying context (e.g. a learned per-sequence start token or a prefix), which changes the architecture for all heads.
4. Accept a lower `regress` number and record it as the `regress` capability ceiling (~0.36 at this scale) rather than a pass.

## Q5 — Remaining F9 minor items (open, non-blocking)
- `GPT2Scorer.score_sample` still scores one text at a time. Batching is deferred to T4.2, as the review recommends.
- Resume restores model and optimizer (including the new decay/no-decay groups) but not the data iterator or RNG. Acceptable for dev; same batch size and seed required for reproducible resume.
- Weight decay now excludes embeddings, LayerNorm parameters and biases. This changes optimizer parameter groups, so pre-F9 checkpoints cannot be resumed (model-only loading still works).
- `git_sha.txt` is suffixed with `-dirty` when the working tree has uncommitted changes.
