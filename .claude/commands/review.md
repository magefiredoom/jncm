---
description: Review a commit range against docs/SPEC.md (no code edits)
argument-hint: <base>..<head>   (default: the last 10 commits)
---
You are reviewing code written by other agents. Do not edit code. Read AGENTS.md and docs/SPEC.md, then review `git diff $ARGUMENTS` (if empty, `git diff HEAD~10..HEAD`).
Write docs/REVIEW_<date>.md with findings sorted by severity (BLOCKER, MAJOR, MINOR). For each: file and line, what is wrong, why it matters for the experiment, the fix, and a test that would have caught it.
Check specifically:
1. information leak: anything at level k depending on level >= k; off-by-one in the cumulative reconstruction;
2. parameters that never receive gradients (compare optimizer params with params that have grads after one step);
3. GPU-only failures: autocast dtype mixes, generator device mismatches, memory at the configured batch size;
4. stop-gradient and detach placement (decoder_through_E, JEPA targets, EMA eval mode, update order);
5. losses: normalization, level weights, padding masks; noise and corruption applied in the right place and mode;
6. evaluation: re-tokenization, entropy definition, seeds, diversity metrics;
7. accounting identical across vq, noisy and JEPA runs (SPEC 6.2);
8. anything that makes the vq vs noisy vs JEPA comparison unfair (parameter counts, budgets, configs);
9. script and sbatch issues: wrong run ids, resume behaviour, mode bits, offline variables.
No style nits. If a category has no findings, say so in one line. You cannot run torch on GPU here: mark runtime claims as unverified and name the test or pilot that would confirm them.
