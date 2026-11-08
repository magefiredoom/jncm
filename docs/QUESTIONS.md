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

## Q8 — T3.5 memorization acceptance for `regress` (RESOLVED 2026-11-07)
Decision: do not change SPEC 5.1 or the regress head. A deterministic head at level 1 (one position, zero conditioning) can only emit one code, so the old 8-sequence unconditional acceptance was an invalid test, not a model bug.

T3.5 now accepts: (a) memorization of ONE fixed sequence at >= 0.90 token accuracy; (b) teacher-forced level 1 — supply the true level-1 code and sample levels 2..K from the regress head — reproducing all 8 sequences at >= 0.90. Both arms pass (one sequence: 1000 steps; teacher-forced 8 sequences: 2500 steps). The `xfail` arm is removed, SPEC 5.1 and the regress head are unchanged, and TASKS T3.5 is ticked. SPEC 5.3 documents the unconditional level-1 collapse as expected for this ablation arm.

## Q5 — Remaining F9 minor items (open, non-blocking)
- `GPT2Scorer.score_sample` still scores one text at a time. Batching is deferred to T4.2, as the review recommends.
- Resume restores model and optimizer (including the new decay/no-decay groups) but not the data iterator or RNG. Acceptable for dev; same batch size and seed required for reproducible resume.
- Weight decay now excludes embeddings, LayerNorm parameters and biases. This changes optimizer parameter groups, so pre-F9 checkpoints cannot be resumed (model-only loading still works).
- `git_sha.txt` is suffixed with `-dirty` when the working tree has uncommitted changes.

## Q9 — T4.1 offline variables versus `prep.sbatch` (ASSUMPTION, non-blocking)
TASKS T4.1 asks every Slurm template for the offline environment variables (`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`). `scripts/slurm/prep.sbatch` exists to download `roneneldan/TinyStories` (SPEC 3, AGENTS rule 6: downloads live only in `scripts/prep_*.py`), so setting them would make the job it exists to run fail.
**Assumption:** `tok.sbatch` and `gen.sbatch` (and later `eval.sbatch`) set all three; `prep.sbatch` sets none and documents that it needs Hub access (internet-enabled partition or a login node). Reversible: add the three exports to `prep.sbatch` if the researcher wants prep to fail rather than reach the network.

## Q10 — JEPA E-bar is a full deep copy of the tokenizer (ASSUMPTION, non-blocking)
SPEC 8 says "EMA encoder E-bar (deep copy)". `jncm/jepa/ema.py` copies the whole `MultiscaleTokenizer` — embedding, encoder blocks, all K quantizers and reconstruction operators, and the decoder blocks plus token head — and EMA-updates every parameter. The target chains (`online` and `ema_clean`) use only `encode_trunk` and the quantizers; the decoder copy is never used, but the full copy keeps "deep copy" literal, makes `state_dict` loading symmetric, and costs one extra tokenizer of memory (about the same as the online tokenizer). Reversible: EMA only the encode path if dev-scale memory matters.

## Q11 — VICReg's c-hat_k is the deterministic prediction (ASSUMPTION, non-blocking)
SPEC 8 step 6 applies VICReg "on online c_k and on c-hat_k" but does not define c-hat_k. Implemented as the prediction the head makes for the target, without the channel noise (sampling-time renormalization and noise would make the variance term trivially satisfied):
- `regress`: `c-hat_k` is the raw linear output `heads[k-1](h_k)` — no renorm.
- `flow`: `c-hat_k` is the noiseless Euler integration of the flow ODE, starting from `eps ~ N(0, I)` in [B*l_k, d_k]: `x_0 = eps`, then `x_{i+1} = x_i + (1/flow_steps) * net_k(x_i, t_i, h_k)` for `i = 0..flow_steps-1` with `t_i = i / flow_steps`, and `c-hat_k = x_final` reshaped to [B, l_k, d_k]. The start `eps` comes from the step's single generator (drawn after the flow-matching draws in the loss), so a seed reproduces the term, but the `nets` are called `flow_steps` times per level: a `flow` + VICReg joint step costs more than a Phase-1 flow step.
Recorded as an ASSUMPTION in STATE.md; SPEC.md not edited.

## Q12 — collapse alarm watches the online c_k only (ASSUMPTION, non-blocking)
SPEC 8's diagnostics ask for "min and mean std of c_k and t_k" per level and a collapse alarm "if mean std < 0.1 for 200 consecutive steps" without naming the tensor. `CollapseAlarm` counts consecutive steps with `std_mean(c_k) < 0.1` per level and alarms at 200 (`COLLAPSE_STD`, `COLLAPSE_PATIENCE`); `std_min(c_k)`, `target_std_min(t_k)` and `target_std_mean(t_k)` are logged but never drive the alarm. Rationale: c_k is the representation that must not collapse; t_k is its EMA, and c-hat_k is directly penalised by VICReg, so an early collapsed prediction is expected and should not raise the alarm.

## Q13 — `jepa.enabled` is not consulted (ASSUMPTION, non-blocking)
SPEC does not say how joint mode is selected. `jncm/train_jepa.py` runs joint mode by being invoked; the config flag `jepa.enabled` is never read (it stays `false` in both shipped configs). This keeps the entry point explicit and avoids a second "silently did nothing" path, matching the SPEC 3/4 rule that a misconfiguration must fail loudly. The four SPEC 8 ablation flags (`ema`, `vicreg`, `decoder_through_E`, `head`) and the `target_chain` switch are all consumed and tested. Reversible: require `jepa.enabled = true` in `validate_jepa_config` if the researcher prefers a config-level guard.
