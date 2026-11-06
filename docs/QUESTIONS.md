# QUESTIONS — jncm

Record of questions, ambiguities, and assumptions. Spec decisions belong to the researcher; do not change anything marked [DECISION].

## Status
No test contradicts SPEC. The following non-blocking interpretations were made during the 2026-11-06 verification/conformance pass.

## Q1 — VQ `Encoded.codes`: indices or straight-through vectors?
SPEC 4.1 says `codes` are “the codes actually used in the residual chain” and the API comment says “vq: same as targets” (indices). SPEC 4.2 also says `contrib` uses `R_k(e_st)` in training and `R_k(E_k[idx])` when called from codes.
**Chosen:** keep `codes` as straight-through float vectors so decoder CE gradients reach the encoder; keep `targets` as integer indices. `reconstruct(enc.codes)` is bit-exact with `enc.zhat` (measured diff 0.0). Confirm whether the generator should receive indices instead.

## Q2 — Rounding ties in the level schedule
SPEC 2 says `l_k = round(...)` without specifying tie-breaking. Python's `round` uses banker's rounding. The specified dev/smoke schedules are unaffected. Example: L=64, K=5 gives `[1, 17, 32, 48, 64]`; half-up would give `[1, 17, 33, 48, 64]`.
**Chosen:** Python `round` (no test contradicts it).

## Q3 — Q/K/V projections in the “pre-LN transformer”
SPEC 4.1/5.2 say “pre-LN transformer block” but the draft had no learned Q/K/V or output projection (it used the same tensor for Q, K, V).
**Chosen:** standard learned `qkv = Linear(d, 3d)` and `out_proj = Linear(d, d)`, as required for a transformer block. This changes parameter shapes; pre-fix checkpoints are incompatible.

## Q4 — Synthetic data fallback
SPEC 3 says training/eval read prepared memmaps only. The draft silently trained on random synthetic data when `data/tinystories/{train,val}.bin` were missing.
**Chosen:** add `train.allow_synthetic_data` (default false; true only in `smoke.yaml`). Real runs now fail loudly instead of producing meaningless metrics.
