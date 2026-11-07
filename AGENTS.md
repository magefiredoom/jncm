# AGENTS.md — jncm (JEPA-style Next-Concept Modeling)

Research codebase. Goal: compare discrete (VQ) and continuous multiscale hierarchies for coarse-to-fine text generation (baseline: NCM, arXiv 2610.14162), then add JEPA-style joint training. A human researcher runs long jobs on a Slurm cluster (FIR). You write and test code.

## Read order (every session)
1. `docs/STATE.md` — what is done, broken, next. Trust an item only if it says VERIFIED.
2. `docs/TASKS.md` — find the first unchecked task. Work on that one only.
3. `docs/SPEC.md` — read the sections the task refers to. The spec is the source of truth.

## Rules
1. **One task at a time.** Finish it, test it, update STATE.md, commit, stop. Do not start the next task unless told to.
2. **The spec is law.** Never edit `docs/SPEC.md`. Never weaken, skip or delete a test to make it pass (you may change steps or learning rate inside a test, never its threshold). If the spec looks wrong or ambiguous: append the question to `docs/QUESTIONS.md`, take the smallest reversible choice, mark it `ASSUMPTION` in STATE.md, and continue with an independent task.
3. **Never claim something works unless you ran it in this session.** In STATE.md write VERIFIED only together with the command and the last lines of its output. Otherwise write UNVERIFIED. If you cannot run code, everything is UNVERIFIED. Checking a pasted copy of a torch-free function in a sandbox counts as SANDBOX-CHECKED (weaker than VERIFIED; the next session re-runs it as a real test).
4. **Never invent numbers.** Metrics come only from files under `results/` produced by real runs.
5. **CPU first.** Every unit test runs on CPU with `configs/smoke.yaml` in under 60 s. GPU and Slurm are for the human's real runs.
6. **Compute nodes may have no internet.** Training and eval code reads only local files (`data/`, `models/`). Downloads happen only in `scripts/prep_*.py` and `scripts/fetch_models.py`.
7. **Dependencies:** torch, numpy, pyyaml, tokenizers, datasets (prep only), transformers (GPT-2 scoring only), pytest, matplotlib. Nothing else without asking. Use `torch.nn.functional.scaled_dot_product_attention`; no flash-attn, no custom CUDA. `torch.compile` only behind a config flag, default off.
8. **Code quality:** modules under 400 lines, type hints, tensor shapes in docstrings. No stubs (`pass`, `...`, "TODO: implement") in a task you mark done. Prefer boring, standard PyTorch.
9. **Reproducibility:** every run writes `results/<run_id>/{config.yaml, git_sha.txt, metrics.jsonl}`; seed everything; log `tok_positions`, `gen_positions`, `total_positions`.
10. **Commit after every task:** `git commit -m "T2.2: <summary>"`.
11. **When unsure about a library API, test it in a tiny script before building on it.** Do not guess signatures.

## FIR operator rules (only when I say you may use `ssh fir`)
- Use only `ssh fir '<command>'` (shared connection). Never ask for or type passwords or MFA codes. If the connection is down, stop and tell me.
- Allowed: `sbatch`, `squeue -u $USER`, `sacct`, `scancel <id>` for jobs you submitted, `ls`, `cat`, `head`, `tail -n 50`, rsync of this repo's data/ and results/ only. One-time setup: `module load`, `virtualenv`, `pip install --no-index`.
- Never run python, pytest or training on the login node, except one 5-step smoke run as a sanity check. No `rm`. Stay inside <dir>/jncm.
- Print at most 30 lines of any log. Report job ids and states, not full outputs.

## Commands
- All tests: `pytest -x -q`
- One file: `pytest -x -q tests/test_levels.py`
- Slow tests: `pytest -x -q -m slow`
- Smoke tokenizer run (CPU): `python -m jncm.train_tokenizer --config configs/smoke.yaml --steps 50`
- Environment setup: see "Environment" in `docs/STATE.md`.

## Layout
```
jncm/        config.py utils.py data.py levels.py
             tokenizer/ (trunk.py vq.py noisy.py)
             generator/ (backbone.py heads.py corruption.py sample.py)
             jepa/ (ema.py vicreg.py joint.py)
             eval/ (genppl.py)
             train_tokenizer.py train_generator.py train_jepa.py
scripts/     prep_tinystories.py fetch_models.py run_eval.py plot_curves.py slurm/
configs/     smoke.yaml tinystories_dev.yaml
tests/       one test file per module
docs/        SPEC.md TASKS.md STATE.md QUESTIONS.md
data/ results/ models/   (git-ignored)
```

## End of every session (mandatory)
Update `docs/STATE.md` using its template: files touched, commands run and their results, ASSUMPTIONs, bugs found, what the human must do, next task. If you ran out of context or are being replaced by another model, do this first.
