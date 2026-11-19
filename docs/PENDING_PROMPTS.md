# PENDING_PROMPTS — tasks specified in the planning conversation but not confirmed done

Paste or adapt these as needed. They assume CLAUDE.md and AGENTS.md are loaded. Check `git log` and docs/TASKS.md first; skip anything already done. Replace {placeholders}. Output caps ("at most N lines") exist because earlier agents ran out of tokens.

## 1. T5.6 — launch-kit fixes before pilots (local, no GPU)
```text
Task: T5.6 (launch-kit fixes before pilots). Read only scripts/launch_phase.sh, scripts/budget.py, configs/*.yaml (generator.train keys), and the launch-kit tests. Print at most 30 lines of any output. Tests first; run the launch-kit tests plus the non-slow suite once at the end.
1. Pilot mode: g0/g1 generator pilots (and their evals) take TOKENIZER from results/tok_vq_10k or results/tok_noisy_10k when results/<id>/DONE exists; submit a pilot tokenizer only if the real one is missing. Dry-run tests for both cases.
2. Tokenizer run ids are NOT suffixed for lite configs. Add a test comparing the tokenizer sections of the dev and lite configs (they must be equal). Only generator and JEPA run ids get _lite. Full mode skips tokenizer submission when tok_*_10k/DONE exists, for either config.
3. grad_accum: tinystories_dev and jepa_dev 16 (micro-batch 64); lite files 4 (micro-batch 64). Every config must satisfy batch_size % grad_accum == 0; the dry-run plan prints the micro-batch size.
4. `git update-index --chmod=+x scripts/slurm/chain.sh scripts/launch_phase.sh`; verify with `git ls-files -s`; add a test that scripts exec'd by other scripts are mode 100755 in the index.
5. Dry-run output also prints the number of jobs, total requested GPU-hours (sum of --time) and the submission order. Update RUNBOOK section 0, add T5.6 to docs/TASKS.md, update STATE.md, commit and push. Report in at most 8 lines.
```

## 2. FIR sync and pilots
Preconditions (human): MFA steps done; GPT-2-large is in ~/jncm/models/gpt2-large on FIR and symlinked into ~/jncm_git/models; T5.6 pushed.
```text
Task: sync FIR and run the pilots. Follow the FIR rules in CLAUDE.md and the channel recorded in docs/STATE.md "FIR operation notes". Print at most 30 lines of any output.
1. Prove the channel (hostname plus echoed token). In ~/jncm_git: `git checkout -- scripts/slurm/chain.sh` if it shows a local mode change, then `git pull --ff-only`; HEAD must equal local HEAD; check chain.sh and launch_phase.sh are executable; GPT2_LARGE_PATH files exist; with module python/3.12 and the venv, `python -c "import jncm, torch, transformers"` works. If anything fails, stop and tell me; never improvise credentials.
2. Dry-run `scripts/launch_phase.sh g0 pilot`, `g1 pilot`, `g2 pilot` for the dev config and the lite config. The plan must show real tokenizers reused (no pilot_tok_* jobs) and micro-batch 64. Show me the plan only if it has more than 10 jobs or requests more than 4 GPU-hours; otherwise submit all pilots at --time 00:20:00 (at most 20 polls per batch).
3. Apply the "Pilots first" checklist in RUNBOOK.md section 0 to each pilot (no crash, data_source real, finite losses, Gen PPL hook ran offline, samples decode to text, flow and regress report pairwise_agree, sec_per_step recorded). Run `python scripts/budget.py` for dev and lite. Record job ids, run ids and sec_per_step under "FIR jobs" in docs/STATE.md; commit docs only.
4. Report in at most 12 lines: pass/fail per pilot, projected H100-hours per gate for dev versus lite, your recommendation. Submit nothing else.
```

## 3. Corruption sweep (finding F16) — after the pilots, with the dev-or-lite decision made
```text
Task: corruption sweep. Precondition: results/tok_vq_10k is DONE on FIR; config chosen: {CONFIG}.
1. Read the generator.corrupt keys in jncm/config.py. Sync FIR (pull, HEAD check).
2. Submit three ce generator runs of 2000 steps: SETS "generator.corrupt.p_first=0.8", "=0.5", "=0.3" (p_last stays 0.2), RUN_ID corr_p080, corr_p050, corr_p030, GEN_PPL=1, --time = 1.3 x pilot sec_per_step x 2000. Nothing else.
3. When done: evaluate the final checkpoint of each with 64 samples and 1 seed; report Gen PPL, H, pairwise_agree and the final training loss in one table, and recommend a p_first. Do not change any default config; the researcher decides.
```

## 4. Full runs, once per phase (G0, G1 or G2) after the pilots pass
```text
Task: launch full runs for {G0 | G1 | G2}. Config {CONFIG}; wall per job {WALL}; budget cap {N} H100-hours; overrides {SETS or defaults}.
1. Prove the channel; git pull and HEAD check.
2. Preconditions: tokenizers DONE (tok_vq_10k for G0; tok_noisy_10k for G1 and G2); pilots passed in docs/STATE.md "FIR jobs". If anything is missing, stop.
3. Run scripts/budget.py for this phase; if the projection exceeds the cap, stop and show me.
4. `scripts/launch_phase.sh <phase> full --wall {WALL}` as a dry run; show me; wait for "go"; then rerun with --submit. Record job ids, run ids, chunk counts and time limits in docs/STATE.md "FIR jobs"; commit docs only. Do not poll for completion.
```

## 5. Collect results, after a phase's training jobs have finished
```text
Task: collect results for {G0 | G1 | G2}. Print at most 30 lines per output.
1. Prove the channel; squeue and sacct; confirm this phase's training runs have DONE files.
2. Submit eval for each run: `sbatch --export=ALL,RUN_ID=<run>[,TOKENIZER=<ckpt>] scripts/slurm/eval.sbatch` (--time from RUNBOOK); at most 20 polls; then `python scripts/run_eval.py --run_id <run> --merge`.
3. Run `python scripts/gate_report.py` (with the JEPA runs for G2) and scripts/plot_curves.py. Copy only the final table and verdict lines (at most 40) into docs/STATE.md "Results log"; no raw logs. Commit docs only.
4. Print the "paste back" block from RUNBOOK.md. Do not interpret the results.
```
