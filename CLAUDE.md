@AGENTS.md

# Claude Code notes — jncm

Research codebase: continuous (and JEPA-style) versions of NCM (Next-Concept Modeling) for coarse-to-fine text generation, compared against NCM's discrete codes. Target: a 4-page ICLR 2027 workshop paper (suggested deadline Feb 1, 2027). Everything in AGENTS.md above applies to every agent that touches this repo. This section adds what is specific to Claude Code.

## Start of every session
1. Read `docs/HANDOFF.md` once (background, decisions, gotchas), then only `docs/STATE.md` "Current position" and "FIR jobs". Do not load all of STATE.md.
2. Run `git status --short` and `git log --oneline -8`. If the tree is dirty or the last commit looks half-done, run `/recover` before anything else.
3. Pick one task from `docs/TASKS.md`; say which and why before editing. `/resume` does steps 1 to 3.

## Conventions that matter here
- `docs/SPEC.md` is the source of truth and is edited only by the researcher. Ambiguity goes in `docs/QUESTIONS.md` with a marked ASSUMPTION, never a silent choice.
- Tests first, one commit per finding, never weaken a threshold. While developing run only the relevant tests; run `python -m pytest -q -m "not slow"` once at the end.
- Print at most 30 lines of any command output. Reports at most 8 to 12 lines. Say VERIFIED only with a command and output tail seen this session.
- Local host is Windows (PowerShell) with a CPU-only torch; GPU work happens on the FIR cluster (Linux). Use the project venv.
- Recurring failure class: code that passes on CPU but fails on GPU (autocast dtypes, `torch.Generator` device, memory). Treat any CUDA path that has not run in a pilot as suspect.
- Position accounting differs for staged and joint (JEPA) runs; read SPEC 6.2 before touching metrics or plots.

## FIR (cluster) rules
- Compute nodes are offline; downloads happen only in `scripts/prep_*.py` and `scripts/fetch_models.py` on a login node.
- Never run training, pytest or heavy Python on a login node (one 5-step smoke run is fine).
- MFA steps (scp, rsync, new ssh logins) are done by the human. If a step needs MFA, give the exact command and stop.
- Account and partition come from `SBATCH_ACCOUNT` / `SBATCH_PARTITION`; never write them into files.
- After every `sbatch` or `scancel`, record job id, run id, STEPS and time limit under "FIR jobs" in `docs/STATE.md`.
- Never reuse a run id that already has `results/<id>/ckpt/latest.pt` unless resuming on purpose.
- Submit only the jobs the researcher asked for; allocation is limited and fair-share is already over-used.

## Tools in this repo
- Slash commands (`.claude/commands/`): `/resume`, `/recover`, `/handoff`, `/review`, `/fir-status`.
- Use plan mode for changes under `jncm/generator/`, `jncm/jepa/` and `jncm/tokenizer/`.

## Do not
- Edit `docs/SPEC.md`, loosen tests, add silent fallbacks (missing data, unknown config keys, wrong shapes must raise), or invent numbers. Metrics come only from `results/` files produced by real runs.
- Start two tasks at once, or work while another agent has uncommitted changes in the tree.
