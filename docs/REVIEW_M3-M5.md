# REVIEW — M3–M5 against SPEC v1.4

Scope: `jncm_review.zip` (a `git archive`, so tracked files only): generator (backbone, heads, corruption, sampler, `train_generator.py`), JEPA (`joint.py`, `ema.py`, `vicreg.py`, `train_jepa.py`), `run_eval.py`, `gate_report.py`, and the sbatch templates. T4.5 (launch kit) is not in this zip. Method: line-by-line read plus `py_compile` (clean). **I could not run torch**, so runtime claims come from reading; each finding has a test or a pilot check that confirms or refutes it. Budget numbers are my estimates, not measurements.

## Verdict

The logic in the places that usually go wrong is right: level-k conditioning (no leak, no off-by-one), clean targets, EMA math, detach placement, accounting. Three problems would waste GPU time or bias the headline comparison (F11 to F13), and F14 to F16 affect how far the budget goes and how fair the figures are.

## Findings

### F11 BLOCKER — staged generators train with a frozen random input embedding
`jncm/train_generator.py:176-184` computes `backbone.embed_levels(tokenizer, codes)` inside `torch.no_grad()`. `embed_levels` ends in the generator's own `input_embed` (`backbone.py:334-352`: the Linear from the reconstruction to H, the level embedding, the position projection). Under `no_grad` those three never receive gradients, so every G0 and G1 run keeps them at random initialization. The JEPA path (`joint.py:285`) calls the same function outside `no_grad`, so JEPA trains them. That is a confound: JEPA would get an advantage unrelated to JEPA, in exactly the comparison G2 makes.
Fix: split the frozen part from the trainable part. Compute the pooled cumulative reconstructions (`reps`) under `no_grad`, then apply `input_embed` outside it. Use the same function in both trainers.
Test: after one staged backward pass, every parameter of `backbone` (and `head`) has a non-None gradient, for `ce`+vq and `flow`+noisy. Run the same check on one joint step.

### F12 BLOCKER (GPU only) — the JEPA trainer crashes on its first CUDA step
`jncm/tokenizer/trunk.py:98-100`: `sample_cutoff` calls `torch.rand(1, generator=generator)` and `torch.randint(1, K, (1,), generator=generator)`, which create CPU tensors. `train_jepa.py` builds `rng = torch.Generator(device=device)` (CUDA on FIR) and `joint.py:293` passes it in, so PyTorch raises a device-mismatch error. CPU tests cannot see this. The staged trainer is not affected (it passes no generator to the cutoff).
Fix: make the draw device-aware (`device = generator.device if generator is not None else "cpu"` and pass `device=device`), then `.item()`.
Test: unit test with a CPU generator (passes before and after), plus a `skipif(not torch.cuda.is_available())` test with a CUDA generator. The 200-step JEPA pilot will also expose it.

### F13 BLOCKER (resources) — batch 1024 cannot fit on one GPU, and there is no gradient accumulation
`configs/tinystories_dev.yaml:51` and `jepa_dev.yaml:62` use `batch_size: 1024`. The backbone processes every level padded to 128 positions, so one step is 1024 × 12 × 128 = 1.57M tokens through 12 layers of width 768. By the standard rule of thumb for activation memory (about 34·h bytes per token per layer with fused attention) that is roughly 490 GB, against 80 GB on an H100; 64 sequences is about 31 GB. Nothing in the repo does micro-batching. The pilot would fail with an out-of-memory error after waiting in the queue.
Fix: add `generator.train.grad_accum` (default 1) to `train_generator.py` and `train_jepa.py`: split the batch into equal micro-batches, average the losses, step once, and keep position accounting on the full batch. VICReg statistics become per micro-batch; say so in the config comment.
Test: with dropout 0, `corrupt.mode=none`, `ce`+vq, gradients with `grad_accum=4` equal the full-batch gradients within 1e-5.

### F14 MAJOR — milestone grid makes the G2 comparison mostly "not comparable"
Staged `total_positions` = 655M (tokenizer) + generator positions, so its curve spans about 786M to 1.97B. A JEPA run at batch 1024 and 10k steps spans 131M to 1.31B under the single accounting and 262M to 2.62B under the conservative one. With only four milestones, the single-accounting comparison has one overlapping point (the JEPA final checkpoint against the staged 50% one) and the conservative one has two. `gate_report.py` handles this correctly by printing "not comparable", but the figure and the gate need more points.
Fix: milestones every 10% (0.1 to 1.0) for every run, and save milestone files without the optimizer state (a 100M-parameter checkpoint with Adam state is about 1.2 GB; without it about 0.4 GB).
Test: `run_eval.py` loads an optimizer-free milestone checkpoint.

### F15 MAJOR — the three arms have very different parameter counts
Backbone 85.5M. The `ce` head adds 37.8M (12 levels × 768 × 4096), `flow` adds 15.1M, `regress` under 0.1M. The generators are therefore about 123M, 100M and 85M. The proposal planned matched scale. At minimum, log `n_params` (tokenizer, backbone, head, total) in the first metrics record and report them in the results table; consider whether the comparison needs a parameter-matched control.

### F16 MAJOR (risk, not a bug) — the corruption schedule is a guess with a large effect
Defaults are swap rates 0.8 to 0.2 across levels (SPEC 5.4, my guess). At level 1 that means 80% of sequences condition level 2 and finer on another sequence's coarse code. NCM reports conditioning corruption as important and settled on a moderate mean; its exact schedule is unspecified in what I read. Before the full G0 run, spend a small budget on a sweep of `p_first` in {0.8, 0.5, 0.3} (keep `p_last` 0.2; check the key names in `config.py`) with the `ce` arm at about 2,000 steps each, comparing sampled Gen PPL (64 samples, 1 seed). Use the cheapest configuration that passes the pilot.

### F17 MINOR
- `compute_generator_loss` runs the frozen tokenizer under bf16 autocast, so targets and conditioning carry bf16 rounding. Wrap the frozen part in `torch.autocast(enabled=False)` (cheap).
- Resume restores model and optimizer but not the RNG or the data iterator (known, acceptable).
- Neither trainer's CUDA path has run yet: the pilots are the test, so treat the go/no-go checklist as mandatory.

## What checked out
Backbone: block-diagonal mask with correct key padding, levels from the chain of codes `< k` (level 1 sees zeros), normalized positions, level embedding. Heads: ce, regress (renorm and channel noise at sampling) and flow (velocity target `c - eps`, Euler sampling, per-level nets), `weighted_total` identities. Corruption: per-level linear schedule, swap within level, targets untouched. Sampler: K backbone passes, argmax decode. Trainer: frozen eval-mode tokenizer, corruption once per hierarchy, clean targets, LR schedule, checkpoint and resume arithmetic, position accounting. JEPA: `encode_online_targets` matches SPEC 8 step 2, targets under `no_grad` and detached, EMA update after `optimizer.step`, EMA kept in eval mode, VICReg variance and covariance terms, `decoder_through_E` detach, flow integration for VICReg, `ema_clean` ablation. `run_eval.py`: per-seed generators, sampling in eval mode and fp32, shard merge. `gate_report.py`: interpolation only inside the observed range. Sbatch templates: no account, offline variables, resume from `latest.pt`.

## Budget (rough, to be replaced by pilot `sec_per_step`)
A 12-layer, width-768 generator at batch 1024 is about 8.1e14 FLOPs per step (6·N·tokens, padded). At a realistic 150 to 300 TFLOP/s that is 2.7 to 5.4 s per step, so 7.5 to 15 H100-hours for 10,000 steps, per run. Padding wastes about 2× of that. G0 + G1 (flow, regress) + the JEPA base run is about 5 runs; the six ablations would add more. A scaled-down generator (width 512, 8 layers, batch 256, 10,000 steps) is about 6e13 FLOPs per step, roughly 0.6 to 1.1 H100-hours per run, but covers a smaller position range (0.33B). The choice is yours; the pilots decide it.

## Do next
1. Run the prompt below (fixes F11 to F15, optional lite configs).
2. Pilots: `launch_phase.sh g0 pilot` and `g2 pilot` on both the dev and the lite configuration, then `budget.py`.
3. Only then submit full runs.

## Prompt for OpenCode (fresh session)
```text
Task: T5.5 (review fixes F11 to F17 from docs/REVIEW_M3-M5.md). Follow AGENTS.md. docs/SPEC.md is v1.5; I copied it in. Read only docs/REVIEW_M3-M5.md, jncm/train_generator.py, jncm/train_jepa.py, jncm/jepa/joint.py, jncm/generator/backbone.py, jncm/tokenizer/trunk.py (sample_cutoff), configs/*.yaml. Print at most 30 lines of any output. Tests first, one commit per finding, in this order. If you run low on context, finish the current finding, update STATE.md and stop.
1. F11: frozen part under no_grad, input_embed outside it, shared by staged and joint trainers. Test: every backbone and head parameter has a gradient after one step (ce+vq, flow+noisy, one joint step).
2. F12: device-aware sample_cutoff; CPU test plus a cuda-skipif test.
3. F13: generator.train.grad_accum (default 1) in train_generator and train_jepa; equivalence test (dropout 0, corrupt none, ce+vq, accum 4 vs 1, gradients within 1e-5); position accounting unchanged; sec_per_step logged per optimizer step.
4. F14: milestones [0.1, 0.2, ..., 1.0] in the dev configs; milestone files without optimizer state; run_eval loads them; gate_report still works on the denser curves.
5. F15: log n_params (tokenizer, backbone, head, total) in the first metrics record of every trainer; add a column to gate_report. Test.
6. F17: frozen tokenizer part under torch.autocast(enabled=False) in the staged trainer; test that a CPU bf16-autocast step gives the same targets as fp32.
7. Add configs/tinystories_lite.yaml and configs/jepa_lite.yaml (generator H 512, n_layers 8, heads 8, batch_size 256, everything else as dev) and an optional CONFIG env var (default the dev config) in gen.sbatch, jepa.sbatch and eval.sbatch; update launch_phase.sh and budget.py to accept `--config`. Tests with dry runs.
Add T5.5 to docs/TASKS.md, update STATE.md, tick T5.5 only for the items that pass, commit. Report in at most 10 lines.
```