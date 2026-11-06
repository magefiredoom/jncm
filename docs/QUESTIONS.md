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

## Q5 — Remaining F9 minor items (open, non-blocking)
- `GPT2Scorer.score_sample` still scores one text at a time. Batching is deferred to T4.2, as the review recommends.
- Resume restores model and optimizer (including the new decay/no-decay groups) but not the data iterator or RNG. Acceptable for dev; same batch size and seed required for reproducible resume.
- Weight decay now excludes embeddings, LayerNorm parameters and biases. This changes optimizer parameter groups, so pre-F9 checkpoints cannot be resumed (model-only loading still works).
- `git_sha.txt` is suffixed with `-dirty` when the working tree has uncommitted changes.
