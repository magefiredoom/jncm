# Second review of the REVIEW_2026-11-19 fixes (2026-11-20)

Reviewer: OpenCode (K2-Horizon-375B-A23B), run headless on a plain snapshot of commit `154e0ba` (no git history, only `FINDINGS.md` allowed as output; the snapshot differed from the commit only by that file, `review.diff` and a `data/` directory its tests created). Brief: independent search for real defects in `e89aa75..154e0ba`, focus list of six items. It ran before T5.9b (`c37fdef`, `de60f3a`) and before the fixes below, so it did not see those two commits.

## Triage by Claude (each finding checked against the code in this session)

| Finding | Verdict | Action |
| --- | --- | --- |
| MINOR-1 stale docstring in `load_online_tokenizer` (said SPEC 6.2 does not charge the tokenizer, but Q14 does) | confirmed by reading; my B2 commit missed it | fixed in `jncm/train_jepa.py` (docs only) |
| MINOR-2 `--sec_per_step=0` passes the positive-number check, then awk fails with "division by zero" | reproduced: `launch_phase.sh g0 full --sec_per_step=0` printed `awk: cmd. line:13: fatal: division by zero attempted`; a pilot record with `sec_per_step` 0 would do the same | fixed: positive check at argument time and in `compute_chunks` for measured values; tests for `0`, `0.0`, `0e3`, `00.00` and a zero pilot record (failed first) |
| MINOR-3 shell tests cannot tell BusyBox ash from bash, so they error instead of skipping | not adopted. It is an artifact of OpenCode's own PATH (its `bash` is BusyBox). On this host the probe tries `C:\Program Files\Git\bin\bash.exe` first and the suite passes. Turning a wrong shell into a silent skip would hide real failures, against the repo's no-silent-fallback rule | none; the loud failure stays |
| UNVERIFIED: CUDA RNG capture/restore | agreed; already listed in STATE | a GPU pilot that is killed and resumed settles it |

Focus items 1 to 6 of the brief: no further finding (planner arithmetic, atomic save, resume state, `tok_offset` accounting, shell scripts, tests that pass for the wrong reason). This agrees with Claude's own pass of the same range, which found nothing beyond the fixes already made. Neither review can substitute for a GPU run.

## Raw report (unedited)

# Second review — jncm snapshot (post-REVIEW_2026-11-19 fixes)

Review rules: only this file is written. Findings are checkable by reading the
shipped code; all claims below were verified this session (grep/read plus test
runs with the project venv python, cwd = snapshot, `timeout≈600000` where
train tests were involved). Shell-dependent tests were run with
`PATH="/c/Program Files/Git/usr/bin:$PATH"` prepended because the PATH bash
here is BusyBox ash, which cannot run bash arrays (environment artifact, not a
repo defect).

**Totals: 0 BLOCKER, 0 MAJOR, 3 MINOR, 1 UNVERIFIED note.** All ≤12 slots
used at 3; sorted by severity then importance.

## Findings

### MINOR-1 — stale docstring contradicts the Q14-charged tok_offset accounting it implements
- File: `jncm/train_jepa.py`, `load_online_tokenizer`, L140-146: docstring says
  ``tok_positions`` is "kept as bookkeeping; SPEC 6.2 does not charge it to a
  joint run".
- But the module docstring (L20-25) and the trainers' accounting DO charge it:
  ``total_positions = tok_offset + joint_positions`` ("the run warm-starts E
  and D from it, so it is charged: QUESTIONS Q14, decided by the researcher
  2026-11-19"), and `load_online_tokenizer` returns ``tok_positions`` precisely
  so the caller *can* charge it.
- Doc is stale from the pre-Q14 accounting; behavior is correct, docs are not.
  Verified by reading, this session.

### MINOR-2 — `--sec_per_step=0` passes launch_phase.sh validation, then dies in awk
- File: `scripts/launch_phase.sh`. CLI regex (L190-191,
  `^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$`) accepts `"0"`, although the die
  message demands a *positive* number.
- With `sec_per_step = 0`, the planner's awk computes `per = interval * sps` =
  0 and hits a fatal division-by-zero on `k = int(0.9 * wall / per)`. The
  awk stderr message names no flag and gives no remediation, and
  `set -euo pipefail` (L59) aborts the script before the friendly ERR branch
  (L345-351) can run.
- Net effect: loud but confusing; user must infer "don't pass 0". One-char fix
  (reject all-zero), but out of scope to apply here.
- No test covers it: `grep -rn "sec_per_step=0"` and "positive number" across
  `tests/` return nothing. Verified by reading + grep, this session.

### MINOR-3 — shell tests can't tell real bash from a BusyBox-ash impostor, so they error instead of skipping
- File: `tests/test_launch_phase.py` `_working_bash` (L18-33), same pattern in
  the other shell-test files: the host probe is only `bash -c "echo ok"`
  plus filename heuristics — any POSIX-ish shell that prints "ok" (including
  BusyBox ash launched here as `bash.EXE`) passes it.
- Then 52 bash-array-dependent tests (arrays, `${parameter:+}` etc.) fail with
  confusing assertion diffs instead of the module-level skip guarded by
  `BASH is None` (L36-38).
- Robustness only: the suite is right to run wherever a real bash exists, but
  the probe should test a bash-ism (e.g. an array round-trip) before declaring a
  winner. Demonstrated this session: prepending Git's real bash made
  test_launch_phase.py / test_done_and_sec_per_step.py / test_review_fixes.py
  pass 71p/1s, and the sbatch/time/chain suites 42p.

## UNVERIFIED (not run, noted per rules)
- CUDA branches of RNG capture/restore in `jncm/utils.py` (`captures host +
  CUDA states; `restore_rng_state` casts to CUDA and calls
  `torch.Generator(device="cuda").set_state()` with the CPU-serialized tensor)
  cannot execute on this CPU-only box. The CPU path is verified (test_rng_state
  suite passed, incl. save/restore round-trips); the CUDA path is plausible
  standard PyTorch but remains UNVERIFIED until a GPU run.
- Trainer train-time tests (`test_train_generator.py`,
  `test_train_jepa.py`) passed in the earlier session of this review (65p,
  bitwise-resume and token-position accounting); re-verify with those exact
  commands after any edit.

## Focus-item disposition (one line each)
1. compute_chunks planner (launch_phase.sh) — verified vs shipped configs
   (planner arithmetic and milestone alignment); sole defect is MINOR-2.
2. atomic_torch_save + all checkpoint read/write paths — verified
   (tmp+os.replace, cleanup, latest.pt every milestone, all state keys present;
   atomic/budget/plot suites pass). No finding.
3. resume correctness (train_generator.py / train_jepa.py) — verified
   (data_state + RNG streams in checkpoints, bitwise-resume tests pass). No finding.
4. position accounting after tok_offset — verified in trainers + run_eval +
   gate_report + plot_curves (Q14 charging consistent); only MINOR-1 docs
   contradict it.
5. shell issues (launch_phase/chain/eval.sbatch/gen.sbatch/jepa.sbatch) — sbatch
   and launch probes all pass in CI-shaped trees; only MINOR-2 defect.
6. tests passing when code is wrong — no instance found this round: every focus
   area's semantics were exercised by reading + targeted runs; the closest
   item is MINOR-3 (probe robustness) and the missing sec_per_step=0 case
   (MINOR-2).

## Test status this session (venv + real-bash PATH prepend)
- `test_launch_phase.py` + `test_done_and_sec_per_step.py` + `test_review_fixes.py`:
  71 passed, 1 skipped (47.85s).
- `test_chain_sh.py` `test_eval_sbatch.py` `test_jepa_sbatch.py`
  `test_sbatch_config.py` `test_sbatch_time.py`: 42 passed (1.24s).
- Earlier this review (same rules): test_launch_phase 52p/1s; train_generator +
  train_jepa 65p (243s); rng/data/atomic_save/budget/utils 49p;
  run_eval/jepa_configs/jepa_diagnostics 20p; train_tokenizer/review fixes /
  done_and_sec_per_step / eval_data_row / config 34p.
