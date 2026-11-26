# STATE — jncm

Last updated: 2026-11-20 (FIR is back; driven through the user's `newFir` zellij pane, every write announced; the SSH session dropped once and the user re-authenticated with Duo). All eleven findings of docs/REVIEW_2026-11-19.md are fixed (T5.7a/b/c, T5.8, T5.9a/b). **FIR showed that no generator, JEPA or eval job had ever run there; each canary found something the CPU tests cannot** (OOM at Slurm's 256 MB default `5c20748`; silent CPU fallback `3f1eed2`; GPT-2 path, eval `import jncm` and the 7-to-15-minute `import transformers` `40dde9e` and `5bc0904`; the lean tokenizer's missing `encode` `554e7e9`). **Canary 3 (`pilot_gen_vq_ce`, job 4750663, `884a0db`) is the first generator run that completed on FIR:** 300 steps, `COMPLETED` in `00:30:14`, `train_sec_per_step` 4.46, `MaxRSS` 8.3 GB, final loss 5.8538, Gen PPL 783.14, ten 493 MB milestones plus a 1.48 GB `latest.pt`, `DONE` written. Its eval (4750664) failed on the missing `encode` (fixed, tested with the real scorer); the eval was resubmitted alone as 4754470 at `554e7e9` and **COMPLETED** in `00:04:36` (20 eval records, Gen PPL 3760 at step 30 down to 770 at step 300, one seed, 32 samples: a pipeline check). VERIFIED: `.venv/Scripts/python.exe -m pytest -q -m "not slow"` under Git Bash -> `453 passed, 1 skipped, 5 deselected in 106.83s` at `554e7e9`. **2026-11-21: both running sweep jobs failed at 00:06 on a short `torch.save` write: `/home` was full (48 of 48 GiB). Resolved: the results now live in `/scratch/$USER/jncm/results` (symlinked), `/home` is at 8.5 GiB, and the records of the six finished runs are in the local `results/` (VERIFIED below). E3 so far (merged locally, 3 seeds x 64 samples at step 2000): `corr_p080` Gen PPL 99.21 +/- 3.06, H 4.032; `corr_p050` 98.36 +/- 1.81, H 4.014; `pairwise_agree` about 0.015 for both; `corr_p030` pending. The seed-2 shard came from job 4924498 (`md5 8414332dd3cf39963e961875eac44def` on FIR and after decoding). TASKS M6 is drafted (2026-11-21). **E2 passed (packed step 2.94 s against 4.46 s) and E3 is complete: `p_first` stays 0.8 (DECISIONS 7). The researcher chose G0 now, G1 and G2 after T4.15a, and G2 chunks of 06:00:00: all four main chains are submitted (G0 5086493 to 5086497 at `5385690`; G1 5088616 to 5088625 and G2 5088655 to 5088659 at `29d38e5`).** **The E1 pilots (noisy + flow, noisy + regress, JEPA base; jobs 4763744 to 4763749) all pass SPEC 10.2 (VERIFIED below; regress shows an unstable, low-entropy output). The E3 sweep runs `corr_p080` and `corr_p050` on the unpacked `edd6a99` (since 21:46 and 21:50, step 1500 at 23:45, the `p_first` overrides confirmed in their configs) and `corr_p030` is pending, so T4.7 (`c979361`, done locally, half the dev step FLOPs) is pulled only once `corr_p030` has started.** **SPEC v1.6 (written at the researcher's request) and TASKS M5b define the work from here; the researcher chose dev for all long runs; the E1 pilots (g1 flow and regress, g2 JEPA; jobs 4763744 to 4763749) have run.** **Plan and chosen directions: `docs/PLAN_2026-11-20.md`** (level-pair packing T4.7 first, about half of a dev step's FLOPs are padding; the g1/g2 dev pilots for correctness in parallel; dev versus lite by a fixed rule after a packed re-pilot; paper method and setup updated). The 20-minute pilot wall does not fit a dev pilot (30 minutes for 300 steps); CUDA RNG restore on a resume, bf16 behaviour beyond this run and the eval on the cluster are still UNVERIFIED. Paper references verified (`paper/references_verified.md`).

Rule: newest information wins. Rewrite sections; do not append history.

## Current position
- **2026-11-25 (read this first; the bullets below are older):** the four main-study chains (G0, G1 flow, G1 regress, G2 base) are mid-run on FIR, 15 jobs PENDING (see "FIR jobs", poll of 2026-11-25). SPEC is v1.7 (the five items of docs/PLAN_DIRECTION_2026-11-20.md section 8, applied at the researcher's request): 6.1 now requires the T4.9 merge job, and 7 requires T4.10 to T4.12. T4.9's local part is done (`cb770f4`); with the researcher's approval FIR was pulled to `3c24d59` and four merge jobs were submitted afterok on the E4/E5 evals (6262175, 6262176, 6262192, 6262193; "FIR jobs"); T4.9 is ticked once the first of them merges a real eval array. **The queue has not moved today:** all 15 jobs were PENDING at 00:23, 22:47 and 23:07 MST (reason `Priority` for each chain's next chunk); SPEC 10.4 v1.7 gives the priority order if the stall continues (G0, G1 flow, G2, G1 regress), which is the researcher's call. T4.10 is done locally (2026-11-26, not pulled to FIR). Next code task: T4.11 or T4.12.
- 2026-11-20 (T5.7-T5.9a, from docs/REVIEW_2026-11-19.md): generator and JEPA trainers write an atomic `latest.pt` at every 10% milestone and log `train_sec_per_step`; `launch_phase.sh` plans chunks in whole resume intervals and errors when one cannot fit; `run_eval.py --head` / `eval.sbatch HEAD` / `launch_eval` make G1 evals work; resume restores the data stream and every RNG (bitwise-equal to an uninterrupted run, tokenizer trainer excluded); JEPA totals include `tok_offset` (Q14, researcher decision: warm start, charged); collapse-alarm patience is `ceil(200/eval_every)`; ablations are sized from the base pilot; `budget.py` shows DONE rows as 0 h; `launch_phase.sh` refuses leaked `SETS`/`GEN_PPL`/`N_SAMPLES`/`N_SEEDS`. T5.7, T5.8 and T5.9 ticked. T5.9b (plan approved first): `joint_loss` takes an optional `cut` and `train_jepa` draws one nested-dropout cut per optimizer step for all micro-batches (m2); `embed_levels` computes the frozen conditioning chain in fp32 with autocast disabled, the joint path unchanged (m3).
- Milestone / task in progress: M0–M4 complete (T0.2, T1.2, T2.4, T3.1–T3.5, T4.1–T4.5); **T5.1 (EMA)**, **T5.2 (VICReg)**, **T5.3 (JEPA joint trainer)**, **T5.3b (target chain + joint accounting)**, **T5.4 (JEPA configs, `jepa.sbatch`, joint-checkpoint eval, Phase 2 runbook)**, **T5.5a (review fixes F11–F13)** and **T5.5b (review fixes F14/F15/F17, lite configs, CONFIG/--config, gen.sbatch SETS)** and **T5.5c (lite launch kit: run-id suffix rule in `launch_phase.sh`/`budget.py`, Figure C, `RUNBOOK.md` §0.3)** and **T5.6 (launch-kit fixes before pilots: pilot tokenizer reuse, unsuffixed tokenizer ids, `grad_accum` 16/4 with micro-batch 64, executable scripts, plan summary)** complete and ticked. F16 stays a documented risk, not a code change. Next work is the human's gates G0a/G0/G1/G2 on FIR (do not start M6). T4.5 added the launch kit: pilots first, then chunked full runs.
- Last completed tasks: T5.5b (10% milestone grid without Adam state in milestone files, parameter counts in the first metrics record and the gate table, frozen tokenizer outside bf16 autocast, lite configs + optional `CONFIG`/`--config`, `SETS` in `gen.sbatch`); before that T5.5a (F11 shared level-input path with a trainable `input_embed`, F12 device-aware `sample_cutoff`, F13 `generator.train.grad_accum` in both trainers), T4.5 (`DONE` + `sec_per_step` in all three trainers and templates, `scripts/slurm/chain.sh`, `scripts/launch_phase.sh`, `scripts/budget.py`, `RUNBOOK.md` §0 "Pilots first"), T4.4 (`scripts/gate_report.py` G0/G1/G2 table with interpolation, `eval.sbatch` without a required `TOKENIZER`, `RUNBOOK.md` §3B), T5.4 (`configs/jepa_dev.yaml`, `configs/jepa_smoke.yaml`, `scripts/slurm/jepa.sbatch`, JEPA milestone checkpoints in `jncm/train_jepa.py`, joint checkpoints in `scripts/run_eval.py`, `--x` in `scripts/plot_curves.py`, `RUNBOOK.md` §3A). T3.5 acceptance still passes as written in TASKS.md.
- All five Slurm templates now exist (`prep`, `tok`, `gen`, `eval`, `jepa`); `eval.sbatch` evaluates milestone checkpoints with one array task per seed (joint JEPA checkpoints included), `jepa.sbatch` and `gen.sbatch` run arms from `SETS`, and all four training/eval templates accept an optional `CONFIG`.
- T5.5b milestone checkpoints: the dev configs write `ckpt/milestone_<pct>.pt` every 10% (`[0.1, ..., 1.0]`), `smoke` configs keep 10/25/50/100%, milestone files omit Adam state, and `run_eval.py` evaluates exactly those by default.
- T0.2 real data prep and the T1.2 protocol gate are now verified locally on real TinyStories data; `models/gpt2-large` is present in `models/gpt2-large`.
- Gate G0a is done; **the pilot grid is the current step: one pilot (g0 dev) has trained and been evaluated on FIR end to end, the other 14 are not resubmitted**. Canary 3 (4750663, 300 steps, 40-minute wall) completed with `train_sec_per_step` 4.46 (GPU-bound: `nvidia-smi` 99 to 100% in four samples, 41,259 MiB), so a dev generator step costs about 4.5 s and a 10,000-step dev run about 12.4 hours before any budget margin; `budget.py` will say it properly once the pilot is applied (`RUNBOOK.md` section 0.1). FIR `~/jncm_git` is at `554e7e9` (= local), tracked tree clean, `GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large`.
- G0a tokenizer jobs: 2880034 `tok_vq_10k` **COMPLETED** (`02:21:20`, final step 10000, `acc_full 0.99998` -> PASS); 2880035 `tok_noisy_10k` **COMPLETED** 2026-11-09 (`02:16:49`, final step 10000, `acc_full 0.99994` -> PASS). Both now have backfilled `DONE` files (0.848 and 0.8209 s/step). The 24 h job 2800220 was cancelled.

## FIR jobs

| Job ID | Run ID | STEPS | Time limit | State (polled 2026-11-09) |
| --- | --- | --- | --- | --- |
| 2880034 | `tok_vq_10k` | 10,000 | 04:00:00 | **COMPLETED** `02:21:20`, `ExitCode 0:0`; final step 10000, `acc_full 0.9999847` -> G0a vq PASS (>= 0.999); `DONE` backfilled 2026-11-11 (`sec_per_step 0.848`) |
| 2880035 | `tok_noisy_10k` | 10,000 | 04:00:00 | **COMPLETED** `02:16:49`, `ExitCode 0:0`, ended 2026-11-09T09:29:09; final step 10000, `acc_full 0.99993896` -> G0a noisy PASS; `DONE` backfilled 2026-11-11 (`sec_per_step 0.8209`) |

Pilot 2796353 (`pilot_vq`, 500 steps, 20 min limit) COMPLETED in `00:09:05`; the 24 h job 2800220 was cancelled. Resubmission, resume and verdict commands: `RUNBOOK.md` §2.1. Neither tokenizer run wrote a `DONE` file at the time (the FIR `~/jncm` tree predates T4.5), so on 2026-11-11 both were backfilled with the repo's own `jncm.utils.write_done` from the T4.5 checkout `~/jncm_git`: `{"step": 10000, "sec_per_step": 0.848}` and `{"step": 10000, "sec_per_step": 0.8209}` (from `sacct` `Elapsed` `02:21:20` and `02:16:49`). Completion itself is established by `sacct` plus the final metrics record at `step 10000`. Wall clock per step (startup included) is 0.85 / 0.82 s/step; use a real pilot record, not these numbers, to size a chain.

### Pilot grid of 2026-11-11: all 16 jobs FAILED (polled 2026-11-20, FIR back)

Every training job ended `OUT_OF_MEMORY` (exit `0:125`) after 8 s to 5 min and every pilot eval `FAILED` (exit `1:0`) after 2 to 7 s (`afterany` ran them without a checkpoint). Cause: `gen`, `jepa` and `eval.sbatch` declared no `--mem` and no `--cpus-per-task`, so Slurm used its defaults (`ReqMem 256M`, `AllocCPUS 1`); `MaxRSS` was about 230 MB, the size of `import torch`, and the log says `Detected 1 oom_kill event in StepId=3566052.batch` right after `Running: python -m jncm.train_generator`. The 2026-11-10 timing job 3234940 (`pilot_gen_vq_ce`, 100 steps) died the same way (`OUT_OF_MEMORY`, `00:05:00`), so no generator or JEPA metrics record has ever existed. Only `tok.sbatch` declared 6 CPUs and 32000M, which is why only the tokenizers trained. No pilot wrote a results directory (`~/jncm_git/results` holds only `pilot_vq`, `smoke_g0a`, `tok_vq_10k`, `tok_noisy_10k`).

| Gate | Config | Training job (Elapsed) | Eval job |
| --- | --- | --- | --- |
| g0 | dev | 3566052 `pilot_gen_vq_ce` OOM (00:00:43) | 3566053 FAILED |
| g1 | dev | 3566054 `pilot_gen_noisy_flow` OOM (00:01:54); 3566055 `pilot_gen_noisy_regress` OOM (00:02:03) | 3566056, 3566057 FAILED |
| g2 | dev | 3566058 `pilot_jepa_base` OOM (00:03:44) | 3566059 FAILED |
| g0 | lite | 3566090 `pilot_gen_vq_ce_lite` OOM (00:00:08) | 3566091 FAILED |
| g1 | lite | 3566092 `pilot_gen_noisy_flow_lite` OOM (00:05:00); 3566093 `pilot_gen_noisy_regress_lite` OOM (00:00:31) | 3566095, 3566096 FAILED |
| g2 | lite | 3566098 `pilot_jepa_base_lite` OOM (00:00:33) | 3566101 FAILED |

Fix `5c20748`: `gen`, `jepa` and `eval.sbatch` now declare `--cpus-per-task=6` and `--mem=32000M` (copied from `tok.sbatch`, not measured). Billing is unchanged: `sacct` `ReqTRES` shows `billing=12200` both for the 1-CPU/256M pilots and for the 6-CPU/32000M tokenizer job 2880034. Job 3506164 (`bitwise-2b`, account `${SBATCH_ACCOUNT}_`) in the queue belongs to another project; it was not touched.

### Canaries and probes of 2026-11-20 (g0 dev pilot `pilot_gen_vq_ce`, 300 steps; agent submissions, each announced to the user)

Canary command (login2, `~/jncm_git`; each time a dry run without `--submit` first): `GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large scripts/launch_phase.sh g0 pilot [--wall=HH:MM:SS] --submit`. The environment held no `SETS`, `GEN_PPL`, `N_SAMPLES` or `N_SEEDS`. (`--wall` must be written with `=`; the usage text said otherwise until `884a0db`.)

| Jobs | HEAD on FIR | Outcome |
| --- | --- | --- |
| 4740492 gen, 4740493 eval | `5c20748` | canary 1, `--time=00:20:00`: gen `OUT_OF_MEMORY` after `00:04:14` on node fc10513 (`MaxRSS 32723008K`, the new 32000M limit): the log shows `CUDA initialization: CUDA driver initialization failed` and `Starting generator run pilot_gen_vq_ce ... on cpu...`, a silent CPU fallback. Eval cancelled by the agent (`scancel 4740493`). |
| 4742789 gen, 4742790 eval | `3f1eed2` | canary 2, `--time=00:20:00`: gen `FAILED` (exit `1:0`) after `00:12:18` (16:22:09 to 16:34:27): `... on cuda...`, step 0 trained on the H100 (`nvidia-smi` 38,359 MiB used), then the step-0 Gen PPL diagnostics sat in `from transformers import ...` (`ps`: `INl`, `wchan cl_sync_io_wait`, open fd `.../transformers/models/gte/modeling_gte.py`) and ended in `FileNotFoundError: GPT-2 weights not found at models/gpt2-large`. Eval `FAILED` after `00:01:00`: `ModuleNotFoundError: No module named 'jncm'` (`scripts/run_eval.py` line 45). No `metrics.jsonl`, no checkpoint. |
| 4746172 probe 1 (CPU, 2 cores, 8000M) | `40dde9e` | `TIMEOUT` at `00:30:04` on node fc30555. First echo 4m51s after job start (`module load` and venv on a cold node). `from transformers import AutoModelForCausalLM, AutoTokenizer` cold: `real 15m16.191s`; `-X importtime`: `transformers` 792,977,349 us cumulative, `transformers.utils.import_utils` 648,398,566 us self, `transformers.models.auto.auto_factory` 111,014,778 us. The same import in a new interpreter: `import torch 3.9s; transformers 5.22.1 import 446.1s; 8 transformers.models.* modules (2 families) loaded`. The node-local copy stage never finished. |
| 4748974 probe 2 (CPU, 2 cores, 16000M) | `87ef89f` | `COMPLETED` `00:10:43` on fc30555: `module load` 0s, venv 0s, interpreter 1s, `import torch` 93s, `tokenizers.Tokenizer.from_file` 1s, `import transformers` 300s (rc 124, its limit), `from transformers import GPT2LMHeadModel, GPT2TokenizerFast` 220s (9 model modules), `GPT2LMHeadModel.from_pretrained` 28s (774,030,080 parameters). |
| 4750663 gen, 4750664 eval | `884a0db` | canary 3, `--wall=00:40:00` (both `--time=00:40:00`), with the lean scorer (`5bc0904`). Gen `COMPLETED` `00:30:14` (17:25:35 to about 17:55:49), exit `0:0`: `[genppl] tokenizer 0.1s, GPT-2 weights 6.6s`; step 1 `Loss: 8.8073 ... Gen PPL: 8212.38` (26.2 s, CUDA warm-up); milestones every 30 steps at 17:30:31, 17:32:43, 17:34:53, 17:37:12, 17:40:14, 17:42:35, 17:44:42, 17:50:28, 17:53:19, 17:55:31 (the 70 to 80 interval took 5m46s, the others 2.1 to 3.0 minutes); step 300 `Loss: 5.8538 ... Gen PPL: 783.14`; `DONE` `{"step": 300, "sec_per_step": 5.513..., "finished_at": "2026-11-20T17:55:38"}`; final record `train_sec_per_step` 4.462, `total_positions` 694,681,600; `MaxRSS 8341948K`. Eval `4750664_0` `FAILED` after `00:03:16`: `AttributeError: 'LeanGPT2Tokenizer' object has no attribute 'encode'` (`run_eval.py` line 94). |
| 4754470 eval | `554e7e9` | the eval alone, resubmitted after the fix (training is `DONE`, ten milestones exist): `GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large sbatch --account="$SBATCH_ACCOUNT" --gres=gpu:h100:1 --time=00:40:00 --array=0-0 --export=ALL,RUN_ID=pilot_gen_vq_ce,HEAD=ce,TOKENIZER=results/tok_vq_10k/ckpt/latest.pt,N_SAMPLES=32,N_SEEDS=1 scripts/slurm/eval.sbatch`. `COMPLETED` `00:04:36`, exit `0:0`, `MaxRSS 7140020K` (it queued about 5 minutes behind a node reservation): `[genppl] tokenizer 4.9s, GPT-2 weights 9.4s`, `wrote 20 eval records under results/pilot_gen_vq_ce/eval` (ten checkpoints, each an aggregate and a seed shard). Gen PPL by milestone, 32 samples, 1 seed (a pipeline check, not a result): step 30 3759.82, 60 1816.10, 90 1646.89, 120 1558.17, 150 1225.45, 180 911.45, 210 917.12, 240 752.15, 270 974.03, 300 769.73 (H 3.99 at step 300). |

What each failure was and what changed:
- Canary 1: nothing stopped a GPU job from training on the CPU (`device: cuda` silently became `cpu`). Now `jncm.utils.resolve_device` raises when `JNCM_REQUIRE_CUDA=1`, which `tok`, `gen`, `jepa` and `eval.sbatch` export. `gen`, `jepa` and `eval.sbatch` also load `python/3.12` before the venv like `tok.sbatch` (the tokenizer jobs that did so logged `on cuda`); whether that line was the cause or fc10513 was a bad node is NOT established, since canary 2 got a GPU with the line in place and on another node.
- Canary 2, GPT-2 path: `train_generator --gen_ppl` used the config's relative `models/gpt2-large` and ignored `$GPT2_LARGE_PATH` (a checkout on FIR has no `models/`). Now `resolve_gpt2_path` (environment first, then config) and `require_gpt2_weights` run before the scorer and before training.
- Canary 2, eval: `python scripts/run_eval.py` puts `scripts/` on `sys.path`, not the repo root, and the FIR venv has no editable install of `jncm`. `run_eval.py` and `eval_data_row.py` now add the repo root to `sys.path` (the dev venv has an editable install, which is why no CPU test saw it; `tests/test_script_imports.py` hides it first). All eight pilot evals of 2026-11-11 would have hit this even with a checkpoint.
- The `transformers` import: transformers 5.22.1 `utils/import_utils.py` `create_import_structure_from_path` scans the whole package with `os.scandir` and `open()`s every `.py` file at `import transformers` (no cache, no switch), so on Lustre it costs 7 to 15 minutes per job, and model-specific imports do not avoid it (220 s warm). `GPT2Scorer` no longer imports it: `jncm/eval/gpt2_lean.py` is a torch GPT-2 plus a `tokenizers` tokenizer and a 25-line safetensors reader, verified equal to Hugging Face (see VERIFIED). Loading the real weights is not the slow part (`from_pretrained` 28s).

### Pilots g1 and g2, dev (TASKS E1), submitted 2026-11-20 at `554e7e9` (agent, announced; approved by the researcher)

Commands (login1, `~/jncm_git`, after dry runs): `GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large scripts/launch_phase.sh g1 pilot --wall=00:40:00 --submit` and the same with `g2`. Both reuse `results/tok_noisy_10k` (`DONE`). The environment held no `SETS`, `GEN_PPL`, `N_SAMPLES` or `N_SEEDS`; no `pilot_gen_noisy_*` or `pilot_jepa_base` directory existed.

| Job ID | Run ID | STEPS | Time limit | Notes | State |
| --- | --- | --- | --- | --- | --- |
| 4763744 | `pilot_gen_noisy_flow` | 300 | 00:40:00 | gen.sbatch `HEAD=flow`, `GEN_PPL=1` (submission order of the dry run: flow first) | COMPLETED 00:23:35 |
| 4763745 | `pilot_gen_noisy_regress` | 300 | 00:40:00 | gen.sbatch `HEAD=regress`, `GEN_PPL=1` | COMPLETED 00:23:29 |
| 4763746 | `pilot_gen_noisy_flow` eval | 32 x 1 | 00:40:00 | `HEAD=flow`, `afterany` on the flow job | COMPLETED 00:01:52 |
| 4763747 | `pilot_gen_noisy_regress` eval | 32 x 1 | 00:40:00 | `HEAD=regress`, `afterany` on the regress job | COMPLETED 00:01:51 |
| 4763748 | `pilot_jepa_base` | 200 | 00:40:00 | jepa.sbatch, warm start from `tok_noisy_10k` | COMPLETED 00:26:25 |
| 4763749 | `pilot_jepa_base` eval | 32 x 1 | 00:40:00 | joint checkpoint, no `HEAD` | COMPLETED 00:01:51 |

### Corruption sweep, dev (TASKS E3, SPEC 10.3), submitted 2026-11-20 at `edd6a99` (agent, announced)

Submitted while the E1 pilots were still queued (over an hour in `Priority`): the sweep uses only the vq + ce training and eval path, which canary 3 (4750663, 4754470) verified end to end, so its pilot had passed; flow, regress and JEPA, which E1 tests, are not involved. Training: `GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large scripts/slurm/chain.sh scripts/slurm/gen.sbatch 1 --wall=03:13:22 --submit --export=ALL,HEAD=ce,RUN_ID=corr_pXXX,STEPS=2000,GEN_PPL=1,TOKENIZER=results/tok_vq_10k/ckpt/latest.pt,SETS=generator.corrupt.p_first=<p>` (wall = ceil(1.3 x 4.462225 x 2000) s). Eval: `chain.sh scripts/slurm/eval.sbatch 1 --wall=00:20:00 --array=0-2 --dependency=afterany:<training id> --submit --export=ALL,RUN_ID=corr_pXXX,HEAD=ce,TOKENIZER=results/tok_vq_10k/ckpt/latest.pt,N_SAMPLES=64,N_SEEDS=3,CKPT=results/corr_pXXX/ckpt/milestone_100.pt`. Dry runs first; `scontrol show job` confirms `Dependency=afterany:<training id>` for all three evals. The submission output is in `~/jncm_git/results/.submit_corr_pXXX.txt`.

| Job ID | Run ID | STEPS | Time limit | Notes | State (at submission) |
| --- | --- | --- | --- | --- | --- |
| 4775911 | `corr_p080` | 2000 | 03:13:22 | `p_first` 0.8 (the default), `p_last` 0.2 | PENDING; FAILED 02:19:46 (short write of `latest.pt` at step 2000; `milestone_100.pt` complete) |
| 4775912 | `corr_p080` eval | 64 x 3 | 00:20:00 | final milestone (`CKPT`), `--array=0-2` | PENDING (Dependency); `_0`, `_1` COMPLETED; `_2` CANCELLED by the system (01:50) |
| 4893126 | `corr_p080` eval, seed 2 only | 64 x 1 of 3 | 00:20:00 | `--array=2`, same export as 4775912; submitted 2026-11-21 at `edd6a99` | FAILED 00:04:12: `FileNotFoundError: GPT-2 weights not found at models/gpt2-large` (agent error: the shell after the reconnect had no `GPT2_LARGE_PATH`, and `--export=ALL` carried none) |
| 4924498 | `corr_p080` eval, seed 2 only | 64 x 1 of 3 | 00:20:00 | as 4893126, with `GPT2_LARGE_PATH=~/jncm/models/gpt2-large` in `--export` (dry run checked) | COMPLETED 00:05:26 |
| 4775913 | `corr_p050` | 2000 | 03:13:22 | `p_first` 0.5 | PENDING; FAILED 02:16:57 (same; `milestone_100.pt` complete) |
| 4775914 | `corr_p050` eval | 64 x 3 | 00:20:00 | final milestone | PENDING (Dependency); `_0..2` COMPLETED |
| 4775915 | `corr_p030` | 2000 | 03:13:22 | `p_first` 0.3 | PENDING; held and released 2026-11-21 around the move; COMPLETED 02:20:02 (from 15:58:10, on `edd6a99`) |
| 4775916 | `corr_p030` eval | 64 x 3 | 00:20:00 | final milestone | PENDING (Dependency); ran on the packed code `5385690` (sampler unchanged). `_0` COMPLETED on fc10603 (`step 2000 seed 0: Gen PPL 117.66, H 3.98`); `_2` COMPLETED on fc10615 (`Gen PPL 114.04, H 3.98`); `_1` FAILED on fc10609 after 00:03:02: `RuntimeError: device 'cuda' was requested and JNCM_REQUIRE_CUDA=1, but torch.cuda.is_available() is False on this host` (a node fault; the guard worked) |
| 5079385 | `corr_p030` eval, seed 1 only | 64 x 1 of 3 | 00:20:00 | `--array=1`, `GPT2_LARGE_PATH` in `--export`; `SBATCH_EXCLUDE` was not honoured (`ExcNodeList=(null)`), so `scontrol update JobId=5079385 ExcNodeList=fc10609` (rc 0, read back) | COMPLETED 00:00:59 |

Polled 2026-11-20 23:45: 4775911 `RUNNING` since 21:46:23 and 4775913 `RUNNING` since 21:50:09 (both on `edd6a99`, step 1500), 4775915 `PENDING` (`ReqNodeNotAvail, UnavailableNodes:fc10515`), the three eval arrays `PENDING (Dependency)`. The `SETS` question below is answered: both configs carry their `p_first`.

**2026-11-21 00:06 / 00:07: 4775911 and 4775913 `FAILED`** (`sacct`: `02:19:46` and `02:16:57`). Both tracebacks (`tail -n 14 slurm-4775911.out`, `tail -n 6 slurm-4775913.out`) end in `jncm/train_generator.py:646` `atomic_torch_save(state, ckpt_dir / "latest.pt")` -> `torch.save` -> `RuntimeError: [enforce fail at inline_container.cc:672] . unexpected pos 500213376 vs 500213264` (p050: `20616832 vs 20616720`). This is a short write. It happened in both jobs within one minute of each other, at different steps of their schedules. The cause is the `/home` quota: `diskusage_report`, run in the shared pane by the checards session and read by the agent from the pane's scrollback, shows `/home (user $FIR_USER) -> 48GiB/ 48GiB` and `/scratch (user $FIR_USER) 29GiB/ 19TiB`; it lists no `/project`. `ls -la ~/jncm_git ~/jncm`: `~/jncm_git/results -> ~/jncm/results` (also `data` and `.venv` link into `~/jncm`), so the data is in `~/jncm/results`. Training itself finished: `ls -l results/corr_p0{80,50}/ckpt/` shows complete final milestones, `milestone_100.pt` at `493107205` bytes (00:05 and 00:07), and `corr_p080/ckpt/step_002000.pt` at `1479312380`; neither run has `DONE`. The evals 4775912 and 4775914 (`afterany`, `CKPT` = the final milestone) can still produce the E3 numbers, but only once writing works again. No jncm job was submitted or cancelled after this.

Footprint (from the file sizes above): a 10,000-step dev run writes about 9.4 GB (10 milestones of 0.49 GB, `step_005000.pt` and `step_010000.pt` of 1.48 GB each, `latest.pt` 1.48 GB). Results so far are about 33 GB (canary 3 6.4, the sweep about 12, the three pilots about 15; *estimate* from the model sizes). The main study and the ablations come to about 100 GB, which is more than a 50 GB `/home` can hold. `results/` has to move (researcher). `/scratch` is the space available, and Alliance scratch is purged after about 60 days (check Fir's policy). The small records the paper needs (`metrics.jsonl`, `eval/`, `config.yaml`, `git_sha.txt`, `DONE`) are therefore also copied off it from time to time.

Untested before this submission: a `SETS` value containing `=` inside `--export` (if Slurm splits it, the job stops at config load within a minute). After the evals: `python scripts/run_eval.py --run_id corr_pXXX --merge` for each, then the SPEC 10.3 rule.

### Packed re-pilot (TASKS E2, SPEC 10.2), submitted 2026-11-21 at `5385690` (agent, announced)

Pull first: `corr_p030` (4775915) started at 15:58:10 on `edd6a99`. At 17:41 it had run `01:42:51` and logged `Step 1000 | Loss: 3.4646 | ... | Gen PPL: 189.45` (so every module, the lazily imported scorer included, was loaded). Then `git pull --ff-only` on FIR -> head `5385690`, `grep -c 'def pack_pairs' jncm/generator/packing.py` -> 1. All three sweep arms trained on `edd6a99`; their evals run the packed code, whose sampler is unchanged.

`mv results/pilot_gen_vq_ce results/pilot_gen_vq_ce_unpacked` (FIR, and the same rename in the local records), then `GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large scripts/launch_phase.sh g0 pilot --wall=00:30:00` (dry run: 2 jobs, 1.00 requested GPU-hours), then `--submit`. `scontrol show job 5018535` -> `Dependency=afterany:5018534`.

| Job ID | Run ID | STEPS | Time limit | Notes | State |
| --- | --- | --- | --- | --- | --- |
| 5018534 | `pilot_gen_vq_ce` | 300 | 00:30:00 | gen.sbatch `HEAD=ce`, `GEN_PPL=1`, packed code | COMPLETED 00:17:27 |
| 5018535 | `pilot_gen_vq_ce` eval | 32 x 1 | 00:30:00 | `afterany:5018534` | COMPLETED 00:00:35 |

### E4 G0 full run (SPEC 10.4), submitted 2026-11-21 at `5385690` (agent, announced; researcher's choice "G0 now, rest after fix")

Researcher's answers (2026-11-21): submit G0 now; G1 and G2 after T4.15a with SPEC 10.2-scaled timings; G2 chunks of `06:00:00`. Checks: `ls -d results/gen_vq_ce_dev` -> no such directory; head `5385690`. Dry run: `gen_vq_ce_dev: 10000 steps x 2.9400977470242653 s/step [train_sec_per_step] , resume every 1000 steps, 0.9 x 03:00:00 -> 4 chunk(s) (pilot pilot_gen_vq_ce)`, eval with the template defaults (256 samples, `--array=0-4`), plan 5 jobs, 15.00 requested GPU-hours. Command: `GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large scripts/launch_phase.sh g0 full --submit`.

| Job ID | Run ID | STEPS | Time limit | Notes | State |
| --- | --- | --- | --- | --- | --- |
| 5086493 | `gen_vq_ce_dev` | 10000 | 03:00:00 | chunk 1/4, gen.sbatch `HEAD=ce`, `p_first` 0.8 | PENDING |
| 5086494 | `gen_vq_ce_dev` | 10000 | 03:00:00 | chunk 2/4, `afterany` on 5086493 | PENDING |
| 5086495 | `gen_vq_ce_dev` | 10000 | 03:00:00 | chunk 3/4 | PENDING |
| 5086496 | `gen_vq_ce_dev` | 10000 | 03:00:00 | chunk 4/4 | PENDING |
| 5086497 | `gen_vq_ce_dev` eval | 256 x 5 | 03:00:00 | every milestone, `afterany` on 5086496 | PENDING |

**Pull rule while E4 and E5 run:** a chunk runs whatever `~/jncm_git` holds when it starts. Until these chains end, pull only commits that leave `jncm/`, `configs/`, `scripts/slurm/*.sbatch` and `scripts/run_eval.py` untouched (docs, tests and `launch_phase.sh` are fine). T4.14 (it edits `jncm/train_tokenizer.py`) waits until the chains are done.

### E4 G1 and E5 G2 full runs, submitted 2026-11-21 at `29d38e5` (agent, announced; after T4.15a)

FIR pull first: `git diff --name-only HEAD origin/main` -> `docs/STATE.md docs/TASKS.md scripts/launch_phase.sh tests/test_launch_phase.py` (no protected path), `git pull --ff-only` -> head `29d38e5`. Run ids unused (`ls -d` -> no such directory for all three). Dry runs: G1 `10000 steps x 2.97 s/step ... -> 4 chunk(s) (--sec_per_step, overrides pilot pilot_gen_noisy_flow train_sec_per_step 3.99...)` and the same for regress (`3.96...`), 10 jobs, 30.00 requested GPU-hours; G2 `jepa_base_dev: 10000 steps x 5.31 s/step ... 0.9 x 06:00:00 -> 4 chunk(s) (--sec_per_step, overrides pilot pilot_jepa_base train_sec_per_step 6.83...)`, 5 jobs, 30.00 requested GPU-hours. 2.97 = 2.94 x 8.52/8.44 (SPEC 10.2 FLOP ratio, flow the larger G1 arm); 5.31 = 6.83 - (4.46 - 2.94) (no JEPA FLOP count yet; assumes only the backbone part got faster). At the unpacked 6.83 s three intervals take about 5.7 h, still inside the 6 h wall. Commands: `GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large scripts/launch_phase.sh g1 full --sec_per_step=2.97 --submit` and `... g2 full --sec_per_step=5.31 --wall=06:00:00 --submit`. Mapping from `sacct --format=SubmitLine`:

| Job ID | Run ID | STEPS | Time limit | Notes | State |
| --- | --- | --- | --- | --- | --- |
| 5088616 to 5088619 | `gen_noisy_flow_dev` | 10000 | 03:00:00 | chunks 1 to 4, `HEAD=flow` | PENDING |
| 5088624 | `gen_noisy_flow_dev` eval | 256 x 5 | 03:00:00 | `afterany:5088619` | PENDING |
| 5088620 to 5088623 | `gen_noisy_regress_dev` | 10000 | 03:00:00 | chunks 1 to 4, `HEAD=regress` | PENDING |
| 5088625 | `gen_noisy_regress_dev` eval | 256 x 5 | 03:00:00 | `afterany:5088623` | PENDING |
| 5088655 to 5088658 | `jepa_base_dev` | 10000 | 06:00:00 | chunks 1 to 4 | PENDING |
| 5088659 | `jepa_base_dev` eval | 256 x 5 | 06:00:00 | `afterany:5088658`, joint checkpoint | PENDING |

The pull rule of the G0 block applies to all four chains.

**First chunks, 2026-11-22 01:32 (poller and `ls`/`grep` on FIR):** `5086493 TIMEOUT 03:00:15`, `5088616 TIMEOUT 03:00:15`, `5088620 TIMEOUT 03:00:12`, `5088655 RUNNING 03:08:19`. A `TIMEOUT` is the normal end of a non-final chunk: the job trains until the wall, and the next chunk (`afterany`) resumes from `latest.pt`, which is rewritten at every milestone (`gen.sbatch` header). `ls results/<run>/ckpt` for G0, flow and regress: `latest.pt milestone_010.pt milestone_020.pt milestone_030.pt`, last metrics record `step 3500` (so chunk 2 resumes at 3000 and redoes about 600 steps); JEPA `latest.pt milestone_010.pt`, last record `step 1500`. Measured `train_sec_per_step` on the packed code: G0 `2.886`, flow `2.964`, regress `2.779`, JEPA `5.844` (planned 2.94, 2.97, 2.97, 5.31). At 5.84 s, three JEPA intervals take about 4.9 h, inside the 6 h wall, so 4 chunks still cover 10,000 steps. The chunk-2 jobs wait on `ReqNodeNotAvail, UnavailableNodes:fc10515`. The resume check (chunk 2 redoes steps 3001 to about 3600, whose losses chunk 1 logged) runs once G0 chunk 2 has trained 30 minutes.

**2026-11-23/24: four chunks failed on GPU-less nodes; chains repaired (agent, announced; `scontrol` allowed by the researcher).** No poller ran during a usage-limit pause, so these were found on 11-24. `sacct`: `5088617 FAILED 00:02:10` (fc10609), `5088618 FAILED 00:01:06` (fc10608), `5088621 FAILED 00:02:03` (fc10612), `5088656 FAILED 00:02:02` (fc10608), all between 08:09 and 08:31 on 11-23; every log ends in `RuntimeError: device 'cuda' was requested and JNCM_REQUIRE_CUDA=1, but torch.cuda.is_available() is False on this host` (the guard worked; no checkpoint changed). Healthy chunks ran on fc10613, fc10607, fc10614, fc10514 and fc10615; `5086494` (G0 chunk 2) `TIMEOUT 03:00:01` on fc10607 as designed. Repair:
- `scontrol update JobId=<id> ExcNodeList=fc10608,fc10609,fc10612` on the 11 pending jncm jobs (5086495, 5086496, 5086497, 5088619, 5088622, 5088623, 5088624, 5088625, 5088657, 5088658, 5088659), each read back.
- Missing chunks appended after each chain's last chunk, exports copied from `sacct --format=SubmitLine` of the first chunk (dry runs first): flow `chain.sh gen.sbatch 2 --wall=03:00:00 --dependency=afterany:5088619` -> 5994035, 5994036; regress `chain.sh gen.sbatch 1 --wall=03:00:00 --dependency=afterany:5088623` -> 5994037; JEPA `chain.sh jepa.sbatch 1 --wall=06:00:00 --dependency=afterany:5088658` -> 5994038; the same node exclusion on all four. A chunk advances about 3,000 steps from the last milestone, so flow (at 3000, two chunks lost) needs 3 more, regress and JEPA 2 more; requeueing the failed jobs instead could have run two chunks of one run at once.
- Evals repointed: `scontrol update JobId=5088624 Dependency=afterany:5994036`, `5088625 -> afterany:5994037`, `5088659 -> afterany:5994038` (read back `(unfulfilled)`).

| Job ID | Run ID | STEPS | Time limit | Notes | State |
| --- | --- | --- | --- | --- | --- |
| 5994035, 5994036 | `gen_noisy_flow_dev` | 10000 | 03:00:00 | added chunks, after 5088619 | PENDING |
| 5994037 | `gen_noisy_regress_dev` | 10000 | 03:00:00 | added chunk, after 5088623 | PENDING |
| 5994038 | `jepa_base_dev` | 10000 | 06:00:00 | added chunk, after 5088658 | PENDING |

**Poll 2026-11-25 00:23 MST (agent, read-only, `newFir` pane `terminal_0`, channel proved by `hostname -f` -> `login1.int.fir.alliancecan.ca` and an echoed token):** no change since 11-24. `sacct -X`: `5086493 TIMEOUT 03:00:15`, `5086494 TIMEOUT 03:00:01`, `5088616 TIMEOUT 03:00:15`, `5088620 TIMEOUT 03:00:12`, `5088655 TIMEOUT 06:00:10`, the four FAILED chunks as recorded, the other 15 jobs PENDING (`squeue` reason `Priority` for the next chunk of each chain, `Dependency` for the rest). `ls results/<run>/ckpt` and the last metrics record (no run has `DONE`): G0 milestones up to `milestone_060.pt`, last record `step 6500`, `train_sec_per_step 2.940`; flow, regress and JEPA up to `milestone_030.pt`, last record `step 3500` (`2.964`, `2.779`, `5.868`; JEPA `collapse_alarm false`). The full runs log no in-training Gen PPL (it comes from the evals). `scontrol show job` on 5086495, 5088619, 5088622, 5088657, 5994035: `ExcNodeList=fc10608,fc10609,fc10612`. FIR `~/jncm_git` is at `29d38e5`. **No chunk to spare:** G0 needs both remaining chunks (6000 -> 9000 -> 10000), and flow, regress and JEPA need all three of theirs (3000 -> 6000 -> 9000 -> 10000), so one more lost chunk leaves a run short of 10,000 steps while its `afterany` eval still starts. A 20-minute read-only poller runs from the agent session (scratchpad `jnpoll.sh`); it wakes on a hard failure, a TIMEOUT of a final chunk (5086496, 5994036, 5994037, 5994038) or an eval, an eval COMPLETED, `DependencyNeverSatisfied`, a dropped channel, or another poller's output in the pane.

**Results pull checked 2026-11-25 23:27 MST (researcher ran `scripts/pull_results.sh --skip-running` in WSL; agent compared read-only).** Per run, files and bytes (without `*.tmp`) on FIR `~/jncm_git/results/` (`find -L`) against `D:\code\jncm\results`: `tok_vq_10k` 7 files / 392,798,004 B and `tok_noisy_10k` 7 / 298,076,104 B identical on both sides (each has `ckpt/latest.pt`, 130,920,073 B and 99,344,757 B, plus `step_005000.pt`, `step_010000.pt`, `DONE`); the six pilots and `pilot_vq`, `smoke_g0a` identical; the three sweep arms identical except one file each that exists only on D:, the merged `eval/step_002000.json` built locally on 11-21 (no merge ran on FIR); the four main-study runs not copied (still training, `--skip-running`). Byte counts only, no checksums.

### T4.9 merge jobs for the E4/E5 evals, submitted 2026-11-25 about 23:10 MST at `3c24d59` (agent; researcher's approval "Yes, push and submit")

The poller was stopped first. `git push` (`ab7ebd2..3c24d59`). FIR `~/jncm_git`: tracked tree clean (`git status --short --untracked-files=no` -> 0 lines), `git diff --name-only HEAD origin/main` -> docs, RUNBOOK, tests, `scripts/launch_phase.sh`, `scripts/pull_results.sh`, `scripts/slurm/chain.sh`, new `scripts/slurm/merge.sbatch` (nothing under `jncm/` or `configs/`, no `run_eval.py`, no existing template), then `git pull --ff-only` -> `3c24d59`. `scontrol show config` -> `DependencyParameters = kill_invalid_depend` (a merge whose eval seed fails is cancelled). `sbatch --test-only ... merge.sbatch` -> 1 processor in partition `cpubase_bycore_b1`. Command per run: `sbatch --account=$SBATCH_ACCOUNT --dependency=afterok:<eval id> --export=ALL,RUN_ID=<run> scripts/slurm/merge.sbatch`. Read back with `scontrol show job`: `jncm_merge`, PENDING `Dependency`, `TimeLimit=00:15:00`, `TRES=cpu=1,mem=4G,node=1`, partitions `cpubase_bycore_b1,cpubackfill,c12hbackfill`.

| Job ID | Run ID | STEPS | Time limit | Notes | State |
| --- | --- | --- | --- | --- | --- |
| 6262175 | `gen_vq_ce_dev` merge | - | 00:15:00 | `Dependency=afterok:5086497_*` | PENDING |
| 6262176 | `gen_noisy_flow_dev` merge | - | 00:15:00 | `Dependency=afterok:5088624_*` | PENDING |
| 6262192 | `gen_noisy_regress_dev` merge | - | 00:15:00 | `Dependency=afterok:5088625_*` | PENDING |
| 6262193 | `jepa_base_dev` merge | - | 00:15:00 | `Dependency=afterok:5088659_*` | PENDING |

The 1024 x 5 final evals (SPEC 10.4, `CKPT`) are not submitted yet; each needs its own merge afterok on it, and because its shards share the `step_<n>.seed_<i>.json` names of the 256-sample final milestone, it must start only after the 256-sample merge of that run has finished.

## VERIFIED (command + last lines of output)

### T4.10 storage guard and disk line - VERIFIED 2026-11-26 (local CPU, Git Bash; not pulled to FIR)
- Files: `scripts/slurm/chain.sh` and `scripts/launch_phase.sh` resolve `results` with `realpath -m` (links followed; `cygpath -m` normalisation only where it exists, i.e. Git Bash) and compare it with `JNCM_FORBID_RESULTS_UNDER` (default `/home`) on a path boundary; `--submit` under the prefix exits 2 naming SPEC 7 and the resolved root (launch_phase checks before its first chain, so nothing is half-submitted); a dry run prints `results root ...` (chain: in its `DRY-RUN` line; launch_phase: a `results root:` line) and only warns. `scripts/budget.py` prints a `DISK <run>:` line per run after the unchanged table, plus `DISK TOTAL`: whole run, GB = 1e9 B and GiB, milestones 4 B per stored parameter, resume files (latest.pt + step files) 4 B + 8 B Adam per trainable parameter, cadence from `generator.train` of the run's config family, JEPA with the EMA tokenizer copy when the pilot ran with `ema`, ablations marked upper bound (base pilot), tokenizer rows and pilots without `n_params_*` `n/a` with the reason (exit status unchanged). The existing chain and launch test envs now set `JNCM_FORBID_RESULTS_UNDER=/jncm-test-no-such-prefix` (hermetic).
- Red first: `pytest -q tests/test_results_guard.py tests/test_budget.py` -> `14 failed, 21 passed` (the three allowed-submission guard tests passed already, as they should). Green: `tests/test_results_guard.py` `9 passed`, `tests/test_budget.py` `26 passed`, `tests/test_chain_sh.py` `17 passed`.
- On the real local pilot records: `python scripts/budget.py` -> `DISK gen_vq_ce_dev: 9.37 GB = 8.72 GiB (whole run: 10 milestone(s) x 0.493 GB + 3 resume file(s) x 1.479 GB; ...)` (acceptance about 9.4 GB; FIR measured 493,107,205 B per milestone and 1,479,308,875 B for latest.pt), flow 0.402 / 1.206 GB and regress 0.342 / 1.026 GB against 402,210,533 / 1,206,630,362 and 342,210,501 / 1,026,618,827 B measured, JEPA 0.468 / 1.339 GB against 468,437,855 / 1,339,093,729 B; `DISK TOTAL: 84.40 GB = 78.60 GiB over 10 run(s)`.
- FIR (read-only, 2026-11-26 00:01 MST): `realpath -m -- results` in `~/jncm_git` -> `/scratch/$USER/jncm/results`, `realpath -m -- /home` -> `/home`, no `cygpath`; so the default prefix lets today's layout submit.
- Full suite: `.venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `1 failed, 525 passed, 1 skipped, 5 deselected in 489.29s`. The failure is `tests/test_train_jepa.py::test_joint_train_sec_per_step_excludes_diagnostics` (`assert 0.5148 < 0.5 * 0.7261`, rerun `0.6874 < 0.5 * 0.9049`): a host-speed timing test (a 0.3 s sleep in two diagnostics against the CPU training step), unrelated to T4.10 (no file under `jncm/` changed; the same code passed it in the 509-test runs of 2026-11-25). **Finding, not fixed:** the test is flaky when the host is loaded; its threshold stays. Raising the sleep would make it robust, but that is a test change for the researcher to approve.

### T4.9 merge job, local part - VERIFIED 2026-11-25 (local CPU, Git Bash; FIR use UNVERIFIED)
- Files: new `scripts/slurm/merge.sbatch` (1 CPU, `--mem=4G`, `--time=00:15:00`, no GPU or array, `RUN_ID` unset or empty -> exit 2, missing `results/<RUN_ID>/eval` -> exit 1, runs `python scripts/run_eval.py --run_id "$RUN_ID" --merge`); `scripts/slurm/chain.sh --merge-after` (after the last chunk, `sbatch --account --dependency=afterok:<last id> --export=<same export> scripts/slurm/merge.sbatch`; refuses an export without `RUN_ID`; `CHAIN_LAST_ID` unchanged, `MERGE_JOB_ID` added); `scripts/launch_phase.sh` passes it for every eval and prints `+ N CPU merge job(s)` apart from the GPU-hours; RUNBOOK merge note (manual merge for the evals already queued, `DependencyNeverSatisfied` after a failed seed).
- Red first: `pytest -q tests/test_chain_sh.py tests/test_merge_sbatch.py` -> `12 failed, 13 passed`; `pytest -q tests/test_launch_phase.py` -> `11 failed, 56 passed` (four existing tests changed on purpose: their sbatch-line or call counts grew 3 -> 4 by the merge, each now also checks the merge line). Two of my new tests had wrong fixtures (a g1 full run without the finished tokenizers; a guessed job count) and were corrected to the real dry-run plan (`5 job(s), 1.67 requested GPU-hour(s)` + `2 CPU merge job(s)`). `tests/test_sbatch_time.py` lists the templates by name and gained `merge.sbatch`.
- Green: `.venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `509 passed, 1 skipped, 5 deselected in 224.59s`.
- Not done: the first FIR use (TASKS acceptance), so T4.9 stays unticked. Pulling it to FIR adds `merge.sbatch` and changes no template the queued chunks use; the pull rule above lists `scripts/slurm/*.sbatch`, so the pull is the researcher's call. The four queued E4/E5 evals (5086497, 5088624, 5088625, 5088659) were submitted without merges; each needs one `sbatch ... --dependency=afterok:<id> --export=ALL,RUN_ID=<run> scripts/slurm/merge.sbatch` (RUNBOOK).

### GPU resume is bitwise exact - VERIFIED 2026-11-24 (FIR, `gen_vq_ce_dev`)
- G0 chunk 1 (5086493, fc10613) trained to about step 3600 and saved `latest.pt` at 3000; chunk 2 (5086494, fc10607) resumed from 3000 and redid the steps chunk 1 had logged. `sed -n 8p` and `9p` of `results/gen_vq_ce_dev/metrics.jsonl`: `"step": 3500 "train_loss": 3.8287889659404755 "sec_per_step": 2.893...` (chunk 1) and `"step": 3500 "train_loss": 3.8287889659404755 "sec_per_step": 2.958...` (chunk 2, after the resume). Identical to the last digit: model, optimizer, data stream and RNG state (CUDA included) are restored exactly on the GPU. This closes the "CUDA RNG restore on a resume" item that was UNVERIFIED since T5.8.

### E2 packed re-pilot passes SPEC 10.2; E3 sweep complete - VERIFIED 2026-11-21 (FIR jobs; records copied and merged locally)
- `sacct` (poller, 20:39:58): `5018534 jncm_gen COMPLETED 00:17:27`, `5018535_0 jncm_eval COMPLETED 00:00:35`, `5079385_1 jncm_eval COMPLETED 00:00:59` (the `corr_p030` seed-1 rerun, `ExcNodeList=fc10609`).
- Records: `tar czf /scratch/$USER/jncm/rec_20261121b.tgz --exclude=ckpt results/corr_p030 results/pilot_gen_vq_ce` -> md5 `ea386fcde1acd5ddb49aaf950ded93bd`, 59459 bytes; decoded locally `lines 81, bytes 59459, md5 ea386fcde1acd5ddb49aaf950ded93bd`, 44 entries; copied into `results/` (no name existed locally).
- E2 (`pilot_gen_vq_ce`, packed, `git_sha.txt` `5385690...`): `DONE` `{"step": 300, "sec_per_step": 3.08584269285202, ...}`; last `train_sec_per_step` `2.9401` against `4.4622` for `pilot_gen_vq_ce_unpacked` (canary 3), a ratio of 0.659 (the FLOP ratio is 0.514; the GPU does not turn all of it into time). Wall `sec_per_step` 3.086 against 5.514. Training loss at step 300 `5.8563` against `5.8538`; Gen PPL per milestone (32 samples, 1 seed) `3744, 1804, 1629, 1672, 1056, 870, 750, 1033, 724, 574` against `3760, 1816, 1647, 1558, 1225, 911, 917, 752, 974, 770`; H 4.02 to 4.50 in both. In line with the canary; the dropout streams differ (T4.7).
- E3 (`run_eval.py --run_id corr_p030 --merge` -> `wrote results\corr_p030\eval\step_002000.json`; 3 seeds x 64 samples at step 2000): `corr_p080` Gen PPL `99.21 +/- 3.06`, H `4.032`; `corr_p050` `98.36 +/- 1.81`, H `4.014`; `corr_p030` `115.42 +/- 1.60` (`117.66, 114.56, 114.04`), H `4.023`; `pairwise_agree` `0.0147`, `0.0152`, `0.0153`. All three meet the H and `pairwise_agree` conditions; p050 is lowest but beats the default by 0.85 < 2 x 3.06 = 6.12 (DECISIONS 7).
- Every FIR `git_sha.txt` ends in `-dirty` (also `884a0db...` and `edd6a99...`): `get_git_sha` (`jncm/train_tokenizer.py:33`) uses `git status --porcelain`, which counts the untracked `slurm-*.out` files in `~/jncm_git`. It does not show a modified tracked file (TASKS T4.14).
- Dry runs on FIR (`launch_phase.sh g0 full`, `g1 full --sec_per_step=2.97`, `g2 full --sec_per_step=5.31`, nothing submitted): G0 `gen_vq_ce_dev` 10000 steps x 2.9401 s/step [train_sec_per_step] -> 4 chunks, plan 5 jobs, 15 requested GPU-hours; G1 flow and regress 10000 x 3.9907 / 3.9602 (the unpacked pilots) -> 5 chunks each, 12 jobs, 36 h; G2 `jepa_base_dev` 10000 x 6.8279 -> 10 chunks, 11 jobs, 33 h. `--sec_per_step` is ignored when a pilot exists (the pilot value wins), so the SPEC 10.2 scaling is not applied (TASKS T4.15). The ablation list (`--ablations` submits all six) is not in SPEC 10.6 order (T4.15).

### Results moved to /scratch, records copied locally - VERIFIED 2026-11-21 (FIR, agent, announced; `scontrol` allowed by the researcher)
- `scontrol hold 4775912,4775914,4775915,4775916` -> rc 1: 4775912 and 4775914 had already run. `sacct`: `4775912_0` and `_1` `COMPLETED` (00:18 and 01:32), `4775912_2` `CANCELLED by 0` on `fc10517` at 01:50:00 (by the system, not by a session); `4775914_0..2` `COMPLETED`. 4775915 and 4775916 `JobHeldUser`.
- `du -sh ~/jncm/results/*`: corr_p050 6.0G, corr_p080 7.4G, pilot_gen_noisy_flow 6.0G, pilot_gen_noisy_regress 5.1G, pilot_gen_vq_ce 7.4G, pilot_jepa_base 6.9G, pilot_vq 250M, smoke_g0a 27M, tok_noisy_10k 285M, tok_vq_10k 375M.
- `mkdir -p /scratch/$USER/jncm && nohup mv ~/jncm/results /scratch/$USER/jncm/results` (about 9 minutes; log: `nohup: ignoring input` only), then `ln -s /scratch/$USER/jncm/results ~/jncm/results` (rc 0). `ls ~/jncm_git/results/` resolves through both links to all ten runs; `du -sh /scratch/$USER/jncm/results` -> `40G`. `diskusage_report`: `/home (user $FIR_USER) 8475MiB/ 48GiB`, `/scratch (user $FIR_USER) 69GiB/ 19TiB`.
- `scontrol release 4775915,4775916` -> rc 0, both `PENDING None`; FIR head still `edd6a99` (not pulled, so `corr_p030` runs the code of the other two arms).
- Records: `tar czf /scratch/$USER/jncm/rec_20261121.tgz --exclude=ckpt results/{pilot_gen_vq_ce,pilot_gen_noisy_flow,pilot_gen_noisy_regress,pilot_jepa_base,corr_p080,corr_p050}` -> 136941 bytes, 140 entries, md5 `dfc3f4abb51c627ce79c9c1c957d7216`. Printed through the pane as base64 and decoded locally: `lines 184, bytes 136941, md5 dfc3f4abb51c627ce79c9c1c957d7216`. The six runs were copied into the local `results/` (no name existed locally, `cp -n`).
- Local merge: `.venv/Scripts/python.exe scripts/run_eval.py --run_id corr_p050 --merge` -> `wrote results\corr_p050\eval\step_002000.json` (local only). Final milestone (step 2000, 64 samples per seed): corr_p050 Gen PPL `98.36 +/- 1.81` over 3 seeds (`100.91, 97.10, 97.06`), H `4.014 +/- 0.008`, `pairwise_agree` `0.0152`; corr_p080 seeds 0 and 1 `103.29` and `95.91`, H `4.036` and `4.044`, `pairwise_agree` `0.0148` and `0.0145`. Both arms meet the H and `pairwise_agree` conditions of SPEC 10.3; the decision waits for corr_p080 seed 2 and corr_p030.

### E1 pilots g1 and g2 pass SPEC 10.2 - VERIFIED 2026-11-20 (FIR, code `edd6a99`, unpacked)
- `sacct -j 4763744,...,4763749 -X -n --format=JobID,JobName,State,Elapsed,Start,End`: `4763744 jncm_gen COMPLETED 00:23:35`, `4763745 jncm_gen COMPLETED 00:23:29`, `4763748 jncm_jepa COMPLETED 00:26:25` (all started 21:31:33, wall 00:40:00), evals `4763746_0`, `4763747_0`, `4763749_0` `COMPLETED` in `00:01:5x`.
- `ls results/<run>`: `DONE ckpt config.yaml eval git_sha.txt metrics.jsonl` for all three (`samples` too for the two generators); `git_sha.txt` `edd6a99e`. Last `train_sec_per_step`: flow `3.9907`, regress `3.9602`, JEPA `6.8279` (canary 3 vq + ce: 4.46). These timings are void after T4.7 (SPEC 10.2).
- `grep -oE '(ppl|h_gpt2|pairwise_agree)_mean' results/<run>/eval/step_??????.json`, every milestone finite (32 samples, 1 seed: pipeline checks, not results):
  - noisy + flow: Gen PPL `5524.09` (step 30) falling steadily to `2951.80` (step 300); H `4.769` to `4.534`; `pairwise_agree` <= `0.009`.
  - noisy + regress: Gen PPL alternates between about 36 and 390 (`294.3, 39.4, 387.3, 35.9, 297.7, 72.1, 288.6, 220.2, 390.0, 37.3`); H `2.06` to `3.17` (data row 4.03); `pairwise_agree` up to `0.418`. A low-entropy, repetitive output that GPT-2 scores well; the G1 flag (`pairwise_agree` > 0.5) and the H condition are the checks that matter for this arm.
  - JEPA base: Gen PPL `6560.89` (step 20) to `4319.28` (step 200), lowest `3796.45` at step 120; H `4.815` to `4.663`; `pairwise_agree` <= `0.005`.
- E3 check on the same login: `grep -E 'p_first|p_last|mode:' results/corr_p0{80,50}/config.yaml` -> `mode: swap p_first: 0.8 p_last: 0.2` and `mode: swap p_first: 0.5 p_last: 0.2`, so the `SETS` value with `=` reaches the config. Both at `step: 1500` at 23:45 (about 4.05 s/step).

### T4.7 level-pair packing - VERIFIED 2026-11-20 (local CPU; commit `c979361`; NOT yet on FIR)
- `tests/test_backbone_packing.py` (21 tests) failed first (`ImportError` on the missing helper), then passed. With dropout off the packed and padded layouts give equal outputs and equal parameter and input gradients within 1e-5 (smoke, dev, odd-K and uneven schedules), garbage past a level's length never enters, a group with padding stays finite in both directions, and the staged `ce`/`flow`/`regress` and joint JEPA losses have equal values and gradients under both layouts. The FLOP ratio at the smoke size is at most 0.55.
- Dev step FLOPs with the packed default (same scratch `FlopCounterMode` script as before): `micro-batch 4: 1.694e+12 FLOPs fwd+bwd; per step (batch 1024): 4.336e+14`, with `mlp 2.693e+14`, `qkv 1.010e+14` and `out_proj 3.366e+13`. That is 0.514 of the padded 8.440e+14. The GPU speed is not measured yet.
- Changed tests, with no threshold moved:
  - `test_one_packed_backbone_pass_per_loss` now expects the v1.6 shape `(N_SEQ*2, 33, H)`.
  - `test_regress_overfits_fixed_batch` runs at 4000 steps and lr 1e-3, instead of 1200 steps and 5e-3. Lock-step runs of both layouts (scratch script) agree to `4e-12` for 200 steps in float64, and the gap then grows to `2.03e-02` by step 400, even in float64. A search over steps and lr found every old-style setting failing for some seed with either layout. 4000 steps at 1e-3 passed for 6 seeds x 2 layouts, worst loss `4.5e-04`.
- Focused tests: `140 passed, 4 deselected`. Full suite under Git Bash at `c979361`: `.venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `478 passed, 1 skipped, 5 deselected in 125.66s (0:02:05)`. `jncm/generator/backbone.py` 396 lines (packing code in the new `jncm/generator/packing.py`, 131 lines).
- FIR: the pull of `c979361` did not happen. The SSH session had dropped again (the pane showed `C:\Users\aston>`), and the guarded pull line and one poll line went to the local `cmd`, where nothing ran: `cmd` split them at `|` into failing fragments, and the home directory is not a git repository. FIR is at `edd6a99`, which has T4.8 (`CKPT`) and not T4.7. The poller was stopped.

### Step cost and model sizes - VERIFIED 2026-11-20 (local CPU analysis of the model definitions; FIR `stat`)
- FLOPs of one dev step (`vq` + `ce`), `torch.utils.flop_counter.FlopCounterMode` over `compute_generator_loss` plus `backward` on a random micro-batch of 4, scaled to batch 1024 (scratch script, nothing in the repo changed): `micro-batch 4: 3.297e+12 FLOPs fwd+bwd; per step (batch 1024): 8.440e+14`; by module `LevelTransformerBlock.mlp 5.344e+14`, `.qkv 2.004e+14`, `.out_proj 6.680e+13`, everything else below `1e+12`; `level lengths: [1, 13, 24, 36, 47, 59, 70, 82, 93, 105, 116, 128]`, `real positions 774, padded slots 1536, utilization 0.504`; `achieved 189 TFLOP/s at 4.462 s/step = 19.1% of bf16 dense peak 989 TFLOP/s`. Cause of the padding: `GeneratorBackbone.forward_levels` pads every level to `max(l_k)` and runs `B*K` sequences (`jncm/generator/backbone.py`). Every complementary pair has 129 positions, which is what TASKS T4.7 uses.
- Parameter counts and step FLOPs of all six arms (same method, micro-batch 2): dev `vq+ce` tokenizer 11,932,544, backbone 85,459,200, head 37,797,888, total 135,189,632, `8.44e+14`; dev `noisy+flow` head 15,065,184, tokenizer 8,270,048, `8.52e+14`; dev `noisy+regress` head 73,824, `8.29e+14`; lite backbone 25,488,896 with heads 25,214,976 (`ce`), 13,492,320 (`flow`), 49,248 (`regress`), `6.51e+13`, `6.78e+13`, `6.25e+13` per step (batch 256). The dev `ce` counts equal the canary's logged first record (`"n_params_tokenizer": 11932544 "n_params_backbone": 85459200 "n_params_head": 37797888 "n_params_total": 135189632`).
- Checkpoint sizes on FIR, `stat -c '%s %n'`: `1479308875 results/pilot_gen_vq_ce/ckpt/latest.pt`, `493107205 .../milestone_010.pt`, `1479312380 .../step_000300.pt`.
- T1.2 data row re-run with the lean scorer (local CPU, same command as the original gate plus `--device cpu` and a new output file): `GPT2_LARGE_PATH=models/gpt2-large python scripts/eval_data_row.py --data data/tinystories/val.bin --tokenizer data/tinystories/tokenizer.json --gpt2 models/gpt2-large --n_samples 256 --device cpu --out results/t1_2_protocol_gate_lean.json` -> `Gen PPL: 10.80 +/- 0.26`, `H (GPT-2 tokens): 4.03 +/- 0.01`, `Verdict: PASS`. Against `results/t1_2_protocol_gate.json` (Hugging Face scorer, 2026-11-06): `ppl_mean` 10.804267692346627 vs 10.804267656614687 (difference 3.6e-08), `ppl_std` 0.2611787899 vs 0.2611787213, `h_gpt2_mean` and `h_dataset_mean` identical. The original file was not touched.

### Canary 3: the first completed generator run on FIR - VERIFIED 2026-11-20 (FIR through the `newFir` pane)
- `sacct -j 4750663,4750664 -X --format=JobID,State%12,Elapsed,ExitCode -n` -> `4750663 COMPLETED 00:30:14 0:0`, `4750664_0 FAILED 00:03:16 1:0`. `cat results/pilot_gen_vq_ce/DONE` -> `{"step": 300, "sec_per_step": 5.513463085492452, "finished_at": "2026-11-20T17:55:38"}`; the last `metrics.jsonl` record has `train_loss 5.853775709867477`, `sec_per_step 5.513463085492452`, `train_sec_per_step 4.462225482883174`, `total_positions 694681600`, `gen_ppl 783.140171900873`; `sacct -j 4750663` -> `4750663.batch MaxRSS 8341948K`.
- Log: `Starting generator run pilot_gen_vq_ce (head=ce, quantizer=vq, corrupt=swap) for 300 steps on cuda...`, `[genppl] tokenizer 0.1s, GPT-2 weights 6.6s (~/jncm/models/gpt2-large)`, `Step 1 | Loss: 8.8073 | gen_pos: 131072 | total_pos: 655491072 | Gen PPL: 8212.38`, `Step 300 | Loss: 5.8538 | gen_pos: 39321600 | total_pos: 694681600 | Gen PPL: 783.14`, `Training completed. Results written to results/pilot_gen_vq_ce`. The step-1 record shows `sec_per_step 26.16` (CUDA warm-up), so the `sec_per_step` in `DONE` (5.51) includes it and the startup of about three minutes.
- Checkpoint sizes (`stat -c '%s %n'`): `latest.pt` 1,479,308,875 bytes, `step_000300.pt` 1,479,312,380, each milestone 493,107,205 (milestones carry no Adam state). CORRECTION: an earlier version of this line said 93 MB and 79.3 MB; an `ls | cut -c28-` had cut the leading digits off the size column, and the 1.5 GB writes it called unnecessary do happen (they fitted: the pilot wrote ten of them in 30 minutes). `srun --jobid=4750663 --overlap nvidia-smi`: `100 %, 41259 MiB`, `100`, `99`, `100` (four samples 2 s apart); `ps`: `SNl`, 1.5% CPU. So the 4.46 s/step is GPU time, not I/O; WHY a 23M-parameter generator needs that is not diagnosed (suspects, unmeasured: the fp32 frozen conditioning chain of m3 and F17 with TF32 off, 16 micro-batches of 64, 12 levels per sample).
- The eval failure: `slurm-4750664_0.out` ends `File ".../scripts/run_eval.py", line 94, in _gpt2_ids: ids = gpt2_tokenizer.encode(text)` -> `AttributeError: 'LeanGPT2Tokenizer' object has no attribute 'encode'`. Fixed in `554e7e9`; both new tests fail without the fix with that exact error (module stashed) and pass with it. Full suite at `554e7e9`: `.venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `453 passed, 1 skipped, 5 deselected in 106.83s (0:01:46)`.
- Eval 4754470 (resubmitted at `554e7e9`): `sacct -j 4754470` -> `4754470_0 COMPLETED 00:04:36 0:0`, `4754470_0.batch 7140020K`; `ls results/pilot_gen_vq_ce/eval` lists `step_000030.json` and `step_000030.seed_0.json` through step 300; the log ends `step 300 seed 0: Gen PPL 769.73, H 3.99`, `step 300 (1.0): Gen PPL 769.73 +/- 0.00, total_positions 694681600`, `wrote 20 eval records under results/pilot_gen_vq_ce/eval`. The staged G0 pipeline (tokenizer checkpoint -> generator -> milestones -> eval with the real GPT-2 scorer) therefore runs end to end on FIR; `--merge` and `gate_report.py` have not been run on it.
- The connection: during the wait the SSH session was reset (`client_loop: send disconnect: Connection reset`) and one command line was typed into the local Windows `cmd` before the pane showed the problem; nothing ran there (`head: invalid number '3;'`, no `.git` in the home directory, project repo clean). The user re-authenticated; the channel was re-proven (`login1.int.fir.alliancecan.ca`, token echoed, `ACC=$SBATCH_ACCOUNT`) before the next write.

### Import cost and the lean GPT-2 scorer - VERIFIED 2026-11-20 (FIR probes 4746172 and 4748974; local CPU)
- Probe numbers: the table in "FIR jobs" (all from `slurm-4746172.out` and `slurm-4748974.out`; the `real 15m16.191s` and the `-X importtime` lines are verbatim). From the installed source, `.venv/Lib/site-packages/transformers/utils/import_utils.py` lines 3072 to 3160: `create_import_structure_from_path` calls `os.scandir` recursively and `open(...).read()` on every `.py` file; the local version is `5.22.1`, the same as on FIR (`torch-2.14.1+computecanada`, `tokenizers-0.23.2`, `safetensors-0.8.0` there).
- Equivalence on the real `models/gpt2-large` (CPU, fp32): `pytest -q -m slow tests/test_gpt2_lean.py::test_real_gpt2_large_logits_and_nll_match_the_hugging_face_model` -> `1 passed in 13.26s`; and a script printing `max |dlogit| = 1.33514404296875e-05  tokens: 21`, `NLL hf = 35.519798  lean = 35.519798  ppl hf = 5.9061  lean = 5.9061`, `same ids: True`.
- `pytest -q tests/test_gpt2_lean.py tests/test_gpt2_path.py tests/test_genppl.py tests/test_run_eval.py -m "not slow"` -> `31 passed in 13.05s`, including a fresh interpreter that scores two texts with `transformers` absent from `sys.modules`.
- Full suite at `5bc0904`: `.venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `450 passed, 1 skipped, 5 deselected in 129.57s (0:02:09)`. `tests/test_launch_phase.py` at `884a0db` (usage text fix) -> `59 passed in 48.56s`.

### FIR canaries 1 and 2: findings and fixes - VERIFIED 2026-11-20 (FIR through the `newFir` pane on login2; local tests under Git Bash)
- Canary 1: `sacct -j 4740492,4740493 -X --format=JobID,State%12,Elapsed,ExitCode,NodeList%10` -> `4740492 OUT_OF_MEMO+ 00:04:14 0:125 fc10513`; `sacct -j 4740492.batch --format=JobID,MaxRSS` -> `32723008K`; the log has `CUDA driver initialization failed` and `Starting generator run pilot_gen_vq_ce (head=ce, quantizer=vq, corrupt=swap) for 300 steps on cpu...`. `grep -l 'CUDA initialization' slurm-*.out ../jncm/slurm-*.out` lists only `slurm-4740492.out`; `grep -m1 Starting ../jncm/slurm-2880034.out ...` -> `... on cuda...` for `tok_vq_10k`, `tok_noisy_10k` and `pilot_vq`.
- Canary 2: `sacct -j 4742789,4742790 -X --format=JobID%14,State%12,Elapsed,ExitCode,Start,End -n` -> `4742789 FAILED 00:12:18 1:0 2026-11-20T16:22:09 2026-11-20T16:34:27`, `4742790_0 FAILED 00:01:00 1:0`. The training traceback ends `genppl.py line 64, in _load: raise FileNotFoundError(... GPT-2 weights not found at models/gpt2-large ...)` via `train_generator.py` `run_diagnostics` -> `sample_and_score` -> `gen_ppl` -> `score_texts` -> `score_batch`; the eval log ends `File ".../scripts/run_eval.py", line 45, in <module>: from jncm.config import ... ModuleNotFoundError: No module named 'jncm'`.
- While 4742789 ran (`srun --jobid=4742789 --overlap`, read-only): `nvidia-smi` -> `NVIDIA H100 80GB HBM3, 0 %, 38359 MiB`; `ps -o stat,wchan` -> `INl cl_sync_io_wait`; `ls -l /proc/<pid>/fd` showed `~/.nv/ComputeCache/.gclock`, `train.bin`, `val.bin` and `.../site-packages/transformers/models/gte/modeling_gte.py`; `/proc/<pid>/maps` had `train.bin` and `val.bin` but no `model.safetensors`. A sequential `head -c 268435456 model.safetensors > /dev/null` on the login node took `real 0m2.476s` (`/home` is Lustre, `lfs getstripe -c` = 1), so raw bandwidth is not the limit. `transformers/models` holds 541 entries. That the stall is the transformers import (and not, say, a slow first CUDA kernel) is an INFERENCE from the traceback order (`_load` imports first, checks the path second) and the open descriptor; the probe measures it.
- Fixes: `tests/test_require_cuda.py` (17 tests, red first on the missing helper), `tests/test_gpt2_path.py` (7) and `tests/test_script_imports.py` (2; red first with the cluster's exact `ModuleNotFoundError`). Full suite at `40dde9e`: `.venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `443 passed, 1 skipped, 4 deselected in 114.00s (0:01:53)`.
- FIR after the pulls: `git rev-parse --short HEAD` -> `3f1eed2`, later `40dde9e`; `grep -c 'module load python/3.12'` -> `1` in gen, jepa, eval; `grep -c 'JNCM_REQUIRE_CUDA=1'` -> `1` in all four templates.

### FIR pilot failure: diagnosis and fix - VERIFIED 2026-11-20 (FIR through the `newFir` zellij pane on login2; local tests under Git Bash)
- Channel: `hostname -f; echo ZNCM-9c42-RT; echo ACC=$SBATCH_ACCOUNT` -> `login2.int.fir.alliancecan.ca`, `ZNCM-9c42-RT`, `ACC=$SBATCH_ACCOUNT`.
- `sacct -X -j 3234940,3566052,...,3566101 --format=JobID,JobName%26,State%12,Elapsed,Timelimit,ExitCode` -> every training job `OUT_OF_MEMO+` with `0:125` (Elapsed `00:00:08` to `00:05:00`), every eval `FAILED` with `1:0`.
- `sacct -j 3234940,3566052,3566054,3566058,3566092 --format=JobID%16,State%12,ReqMem,MaxRSS,AllocCPUS,...` -> `ReqMem 256M`, `AllocCPUS 1`, `.batch` `MaxRSS` `230008K` to `230088K`.
- `tail -n 6 ~/jncm_git/slurm-3566052.out` -> `... 3897200 Killed "${cmd[@]}"` and `error: Detected 1 oom_kill event in StepId=3566052.batch`.
- `sacct -j 2880034,3566052 -X --format=JobID,ReqMem,AllocCPUS,ReqTRES%75` -> both `billing=12200` (`cpu=6,mem=32000M` versus `cpu=1,mem=256M`).
- Fix `5c20748`: `pytest -q tests/test_sbatch_resources.py` failed first (`6 failed, 3 passed`); then, under Git Bash, `tests/test_sbatch_resources.py tests/test_sbatch_config.py tests/test_sbatch_time.py tests/test_eval_sbatch.py tests/test_chain_sh.py tests/test_launch_phase.py` -> `109 passed in 41.84s`. Under the PowerShell tool the PATH finds w64devkit's `bash.EXE` first, which cannot parse bash arrays, so `tests/test_chain_sh.py` fails there (19 failures, an environment artifact; `tests/test_sbatch_config.py` picks its bash differently). Run the suite from Git Bash.
- FIR after the pull: `git rev-parse --short HEAD` -> `5c20748`; `grep -c '^#SBATCH --mem' scripts/slurm/{gen,jepa,eval}.sbatch` -> `1` each.
- `scripts/launch_phase.sh g0 pilot` (dry run) printed the plan (2 jobs, 0.67 requested GPU-hours); with `--submit`: `Submitted batch job 4740492` (gen) and `Submitted batch job 4740493` (eval). Whether they run is UNVERIFIED until they finish.

### T5.7-T5.9b (REVIEW fixes) - VERIFIED 2026-11-20 (CPU, Windows, `.venv/Scripts/python.exe`)
- Final full suite at `5c81e17`: `.venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `406 passed, 1 skipped, 4 deselected in 101.28s (0:01:41)` (`398 passed` at `de60f3a`, `403` at `1324c03`; the last three tests cover the `eval.sbatch` GPT-2 path fix `d603888`: `GPT2_MODEL_PATH`, else `GPT2_LARGE_PATH`, else `models/gpt2-large`).
- Second review (OpenCode on a snapshot, `docs/REVIEW_2026-11-20_second_reviewer.md`): 0 blocker, 0 major, 3 minor. Fixed after checking each: a zero `sec_per_step` (reproduced: awk `division by zero attempted`) and a stale Q14 docstring; the shell-probe finding is not adopted (an artifact of the reviewer's PATH). `tests/test_launch_phase.py tests/test_budget.py tests/test_chain_sh.py tests/test_train_jepa.py -m "not slow"` -> `121 passed in 73.89s`.
- Paper references (`paper/references_verified.md`): all six arXiv IDs of HANDOFF section 12 exist and match, LLM-JEPA, VL-JEPA, HDLM, VAR and AURORA-LM were found; the NCM claims of the draft (4 to 5x fewer training tokens counting tokenizer training, one decode, end-to-end as future work, TinyStories reference numbers) and the Base-LCM comparison hold in the full texts (read through a page-fetching helper, so quotes should be re-read once before submission). Q15 records that NCM replaces codes uniformly from the codebook while SPEC `swap` draws another position's code.
- m2 (`c37fdef`): `tests/test_jepa_joint.py tests/test_train_jepa.py tests/test_tokenizer.py -m "not slow"` -> `65 passed, 1 skipped in 39.29s`; the new tests failed first (`TypeError: joint_loss() got an unexpected keyword argument 'cut'`). The bitwise-resume tests still pass.
- m3 (`de60f3a`): `tests/test_generator_backbone.py tests/test_train_generator.py tests/test_generator_heads.py tests/test_jepa_joint.py tests/test_train_jepa.py tests/test_generator_sample.py -m "not slow"` -> `119 passed, 4 deselected in 74.40s`; before the fix the contributions were bfloat16 under CPU autocast (`assert [torch.bfloat16 ...] == [torch.float32 ...]`).
- Full suite at `09086d0`: `.venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `387 passed, 1 skipped, 4 deselected in 112.52s (0:01:52)`.
- B1/m1 (`e89aa75`): trainer, launch and budget tests -> `149 passed, 4 deselected in 61.10s`. The old chunk expectations changed because they described plans that restarted from step 0 (4.0 s/step tokenizer, 5 chunks); the new seeds and counts are explained in the commit message.
- M1 (`643806b`): `tests/test_run_eval.py tests/test_eval_sbatch.py tests/test_launch_phase.py tests/test_sbatch_config.py tests/test_sbatch_time.py tests/test_budget.py tests/test_chain_sh.py` -> `101 passed in 33.90s`; `bash -n scripts/launch_phase.sh` OK; the head-mismatch guard test is unchanged. The failure was reproduced first: `ValueError: checkpoint milestone_010.pt head='flow' does not match config head 'ce'`.
- M2 (`94d0292`): `tests/test_train_generator.py tests/test_train_jepa.py tests/test_data.py tests/test_rng_state.py tests/test_atomic_save.py -m "not slow"` -> `77 passed, 4 deselected in 39.42s`. Mutation check (restore step removed one at a time): global RNG -> CAUGHT, step generator -> CAUGHT, data stream -> CAUGHT; file restored.
- B2 + m7 (`cdfe293`): `tests/test_train_jepa.py tests/test_run_eval.py tests/test_gate_report.py tests/test_plot_curves.py` -> `50 passed in 44.91s`; `tests/test_train_jepa.py tests/test_jepa_diagnostics.py` -> `37 passed in 37.17s`.
- m4/m5/m6 (`09086d0`, written by OpenCode in an isolated worktree, reviewed and re-run here): `tests/test_launch_phase.py tests/test_budget.py` -> `71 passed in 49.25s` in the worktree before the cherry-pick; the cosmetic review fixes are listed in the commit message.

### T5.6 launch-kit fixes before pilots — VERIFIED 2026-11-11 (CPU, Git bash; nothing submitted)
- Configs: `configs/tinystories_dev.yaml` and `configs/jepa_dev.yaml` `generator.train.grad_accum` 1 -> 16 (batch 1024), `configs/tinystories_lite.yaml` and `configs/jepa_lite.yaml` 1 -> 4 (batch 256); `configs/smoke.yaml` stays 1 (batch 16). Micro-batch is 64 in all four dev/lite files and every config satisfies `batch_size % grad_accum == 0`.
- `scripts/launch_phase.sh`: tokenizer ids are literals (`tok_vq_10k`, `tok_noisy_10k`, pilots `pilot_tok_vq`, `pilot_tok_noisy`) and never pass through `rid()`; `select_tokenizer` picks the real id when `results/<id>/DONE` exists. Pilot mode then prints `REUSE <real>: results/<real>/DONE exists, no pilot tokenizer` and launches no tokenizer chain; otherwise it submits the pilot tokenizer as before. Full mode still calls `launch_tok`, which skips on `DONE` for either config. g2 pilot applies the same rule (real `tok_noisy_10k` when `DONE`, else `pilot_tok_noisy`).
- `scripts/budget.py`: `is_tokenizer`/`suffixed_run_id` keep `tok_*` and `pilot_tok_*` rows unsuffixed; generator/JEPA rows still get the config stem. A lite projection therefore measures the two shared tokenizer rows and reports only generator/JEPA rows as `MISSING` without lite pilots.
- Plan summary: every run ends with `== plan: N job(s), H requested GPU-hour(s) (1 GPU x <wall> each)`, the config's micro-batch line and a numbered submission order.
- Real-tree dry run (`export SBATCH_ACCOUNT=$SBATCH_ACCOUNT; bash scripts/launch_phase.sh g0 pilot`; the local `results/` has no `tok_vq_10k/DONE`, so this is the "real missing" case): `== plan: 3 job(s), 1.00 requested GPU-hour(s) (1 GPU x 00:20:00 each)` / `   micro-batch 64 (batch 1024 / grad_accum 16, configs/tinystories_dev.yaml)` / order `1. tok.sbatch pilot_tok_vq`, `2. gen.sbatch pilot_gen_vq_ce`, `3. eval.sbatch pilot_gen_vq_ce`; `G0_PILOT_EXIT=0`.
- Real-tree lite dry run (`bash scripts/launch_phase.sh g1 pilot --config configs/tinystories_lite.yaml`): `RUN_ID=pilot_tok_noisy` (unsuffixed) -> `pilot_gen_noisy_flow_lite` + `pilot_gen_noisy_regress_lite` -> two evals; `== plan: 5 job(s), 1.67 requested GPU-hour(s)` / `   micro-batch 64 (batch 256 / grad_accum 4, configs/tinystories_lite.yaml)`; `G1_LITE_EXIT=0`. The `REUSE` branch is covered by tests (`test_pilot_reuses_a_finished_real_tokenizer`, `test_lite_pilot_reuses_the_shared_real_tokenizer`) because no local `results/<tok>/DONE` exists and none was invented.
- `bash -n scripts/launch_phase.sh` -> `BASH_N=0`.
- Focused: `.venv/Scripts/python.exe -m pytest -x -q tests/test_launch_phase.py tests/test_budget.py tests/test_jepa_configs.py` -> `52 passed in 28.87s` (EXIT=0).
- Full non-slow suite: `.venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `304 passed, 1 skipped, 4 deselected in 79.66s` (EXIT=0); the skip is the CUDA-only `sample_cutoff` test.
- New tests: both tokenizer cases per phase in pilot mode plus lite full skip; dev/lite `tokenizer:` section equality (yaml); `batch_size % grad_accum == 0` over `configs/*.yaml` plus micro-batch 64 for the four dev/lite files; plan output (jobs, 1.00 GPU-hour, micro-batch, order); mode-100755 test that reads `git ls-files -s -- scripts` and requires every shell script referenced by another script to be `100755`.
- `git update-index --chmod=+x scripts/slurm/chain.sh scripts/launch_phase.sh` then `git ls-files -s` -> `100755 ... scripts/launch_phase.sh`, `100755 ... scripts/slurm/chain.sh`. This is the fix for the FIR `Permission denied` (`RC=126`) recorded above.
- `test_config.py::test_lite_configs_scale_only_the_generator` now also normalizes `grad_accum` (lite 4 / dev 16) before comparing the two configs; its assertions were not weakened elsewhere.

### Figure C paper polish — VERIFIED 2026-11-11
- `scripts/plot_tokenizer_levels.py`: figsize 7.0 x 2.8 in via module constants `FIG_SIZE`/`DPI` (300), constrained layout; C2 legend moved outside the data area (`bbox_to_anchor=(0.5, -0.32)`, below the right axis, two columns, frame off); the dotted reference is now `LEVEL1_CAP = 512/4096 = 0.125` labelled `level-1 cap (512 codes / 4096)` and it appears in the legend (it did not before: `axhline` was drawn after `legend`); the vq level-1 point carries a hollow marker overlay (`markerfacecolor="none"`, zorder 5); C1 got headroom (`ylim 0..1.12`, ticks 0..1.0) and 4.5 pt point labels so the level 9-12 labels no longer sit on the top spine, and C2 `ylim 0..1.08`.
- `.venv/Scripts/python.exe scripts/plot_tokenizer_levels.py` -> `wrote figures\fig_c_tokenizers.png`, `records: tok_vq_10k step 10000, tok_noisy_10k step 10000`, `PLOT_EXIT=0`; the printed 12-row table is identical to the T5.5c one above (same records, same numbers).
- PNG check: `size (2100, 840) dpi (299.9994, 299.9994)` = 7.0 x 2.8 in at 300 dpi, 148,314 bytes. The rendered image was read back and inspected: legend fully below the C2 axes, cap line at 0.125 above the level-1 vq point (0.1042), hollow level-1 vq marker visible, no clipped text at the margins.
- Fixture tests: `.venv/Scripts/python.exe -m pytest -x -q tests/test_plot_tokenizer_levels.py` -> `6 passed in 1.27s`, `PYTEST_EXIT=0` (no test was changed).
- Module length 251 lines (< 400). The full non-slow suite was not re-run: only this one script's rendering changed.

### FIR sync, DONE backfill and pilot dry runs — VERIFIED 2026-11-11 (zellij `attach`, pane `terminal_0`, login2)
- Channel: `hostname -f` -> `login2.int.fir.alliancecan.ca`; `echo ACC=$SBATCH_ACCOUNT` -> `ACC=$SBATCH_ACCOUNT`.
- Sync: `cd ~/jncm_git && git pull --ff-only` -> `Already up to date.`, `RC=0`; `git rev-parse HEAD` -> `e43ba47fe2a5f440e8cce4221c8bf4081cd171a9` = local HEAD; `git status --short` -> empty.
- Empty status needed a session-local `.git/info/exclude` (`.venv`, `data`, `results`, `models`): `.gitignore` lists `data/`, `models/`, `results/`, `.venv/` with a trailing slash, which never matches a symlink, so the four docked symlinks show as `??` on an otherwise clean checkout. One leftover `slurm-3234940.out` was moved to `results/.slurm-3234940.out` (nothing deleted, no `rm`).
- Symlinks: `data -> ~/jncm/data`, `results -> ~/jncm/results`, `.venv -> ~/jncm/.venv` all resolve; **`models` is absent** (`ls: cannot access 'models'`), and `~/jncm/models` itself does not exist.
- Env: `export GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large` -> `ls: cannot access '~/jncm/models/gpt2-large': No such file or directory`, `LS1=2`; `find ~ -maxdepth 4 -iname "*gpt2*"` -> nothing; `find ~ -maxdepth 6 -type d -name "*models*"` -> only racerbot `dynamic_models` dirs; `find /scratch/$USER /project/$SBATCH_ACCOUNT -maxdepth 3 -iname "*gpt*"` -> nothing. `module load python/3.12; source .venv/bin/activate; python -c "import jncm, torch, transformers"` -> `jncm ~/jncm_git/jncm/__init__.py`, `torch 2.14.1`, `tf 5.22.1`, exit 0 (transformers import takes ~3 min cold on GPFS).
- Job 2880035: `scontrol update JobId=2880035 TimeLimit=03:00:00` -> `Invalid job id specified` / `SRC=1`; `sprio -j 2880035` -> `slurm_load_jobs error: Invalid job id specified`. `sacct -j 2880035` -> `2880035 jncm_tok COMPLETED 02:16:49 0:0 2026-11-09T09:29:09`. The job had already finished, so the requested 3 h limit was moot; nothing was cancelled.
- DONE backfill with the repo writer `jncm.utils.write_done` (no rerun): `results/tok_vq_10k/DONE` -> `{"step": 10000, "sec_per_step": 0.848, "finished_at": "2026-11-10T22:27:21"}`; `results/tok_noisy_10k/DONE` -> `{"step": 10000, "sec_per_step": 0.8209, "finished_at": "2026-11-10T22:42:50"}`. Noisy final record: `n_records 21 step 10000 acc_full 0.99993896484375 data_source real` -> PASS.
- Bug found (launch kit, not fixed in git): `scripts/slurm/chain.sh` is committed `100644` while `launch_phase.sh:244` runs `"$CHAIN" ...`; on Linux `bash scripts/launch_phase.sh g0 pilot` -> `line 244: scripts/slurm/chain.sh: Permission denied`, `RC=126`. Workaround: `chmod +x scripts/slurm/chain.sh` in the FIR working tree only (repo and HEAD untouched; `git status` then shows that one mode change). Fix later with `git update-index --chmod=+x` on `scripts/slurm/chain.sh` (`launch_phase.sh` itself is invoked via `bash`).
- Dry runs, six green after the chmod (each prints `dry-run (pass --submit)`; no `sbatch` is executed): `g0 pilot` -> `pilot_tok_vq` -> `pilot_gen_vq_ce` (`GEN_PPL=1`) -> eval `N_SAMPLES=32,N_SEEDS=1`; `g1 pilot` -> `pilot_tok_noisy` -> flow + regress in parallel -> 2 evals; `g2 pilot` -> `pilot_jepa_base` (`STEPS=200`) + 6 `LIST` ablations, with `WARNING: results/pilot_tok_noisy/ckpt/latest.pt not found (dry-run only)`. `--config configs/tinystories_lite.yaml` yields `pilot_tok_vq_lite`, `pilot_gen_vq_ce_lite`, `pilot_tok_noisy_lite`, `pilot_gen_noisy_flow_lite`, `pilot_gen_noisy_regress_lite`, `pilot_jepa_base_lite` and forwards `CONFIG=` to every export. **Nothing was submitted.**
- Blocker: every g0/g1 gen pilot exports `GEN_PPL=1` and `train_generator.py` builds `GPT2Scorer` at startup, so those jobs cannot start until `~/jncm/models/gpt2-large` exists (local copy 2.89 GB: `model.safetensors` 2952.7 MB, `tokenizer.json` 3.4 MB). Human action: rsync it, `ln -s ~/jncm/models models` in `~/jncm_git`, then re-run the six dry runs and submit on "go". `budget.py` was not run: with no pilot records under `results/pilot_*` it can only print `MISSING`.

### T5.5c lite launch kit + Figure C — VERIFIED 2026-11-11
- `bash -n` on `scripts/slurm/{prep,tok,gen,eval,jepa}.sbatch`, `scripts/slurm/chain.sh` and `scripts/launch_phase.sh` (Git bash first on PATH): `exit=0` for all seven.
- Configs: `diff configs/tinystories_dev.yaml configs/tinystories_lite.yaml` differs only in the header comment and `generator.H 768 -> 512`, `n_layers 12 -> 8`, `n_heads 12 -> 8`, `generator.train.batch_size 1024 -> 256`; `jepa_dev.yaml` vs `jepa_lite.yaml` the same. Both lite files keep the 10% milestone grid and `grad_accum`.
- Focused: `.venv/Scripts/python.exe -m pytest -q tests/test_plot_tokenizer_levels.py tests/test_budget.py tests/test_launch_phase.py tests/test_sbatch_config.py tests/test_config.py tests/test_jepa_configs.py` -> `62 passed in 17.83s` (EXIT=0). Full non-slow suite: `.venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `292 passed, 1 skipped, 4 deselected in 65.16s` (EXIT=0); the skip is the CUDA-only `sample_cutoff` test.
- Run-id rule (`scripts/launch_phase.sh`): `SUFFIX` comes from `--config` (empty for `*_dev`, `tinystories_lite`/`jepa_lite` -> `lite`); `rid()` replaces a trailing `_dev` or appends `_<stem>`; `base_id()` recovers the dev-shaped id for the pilot lookup. Dry runs give exactly the expected id sets for g0/g1/g2 under both configs (`pilot_tok_vq_lite`, `pilot_gen_vq_ce_lite`, `pilot_jepa_base_lite`, ...), and `g0 full --config configs/tinystories_lite.yaml` sizes its chunks from `pilot_tok_vq_lite`/`pilot_gen_vq_ce_lite` and exports `TOKENIZER=results/tok_vq_10k_lite/ckpt/latest.pt`. One bug of mine was caught by these tests: `run_g2` used the unsuffixed `pilot_tok_noisy`/`tok_noisy_10k` tokenizer path; fixed before the final run.
- `scripts/budget.py --config configs/tinystories_lite.yaml`: prints `run-id suffix: _lite`, projects `gen_vq_ce_lite`, `tok_vq_10k_lite`, ... from `_lite` pilot records only; with only dev pilots present every lite run is `MISSING` (`TOTAL: 0.00 H100-hours over 0 run(s)`), never a dev number. `test_accepts_and_prints_the_config` (T5.5b) was updated to seed suffixed pilots, since the projection it asserts now reads lite records by design.
- Figure C: `python scripts/plot_tokenizer_levels.py` -> `wrote figures\fig_c_tokenizers.png`, `records: tok_vq_10k step 10000, tok_noisy_10k step 10000`, PNG 111,608 bytes, EXIT 0. Real numbers (last `acc_full` record of each run):

| level | vq acc_cum | noisy acc_cum | vq codebook_usage | noisy eff_rank | noisy eff_rank/7 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0776 | 0.0775 | 0.1042 | 6.1512 | 0.8787 |
| 2 | 0.1442 | 0.1083 | 0.7297 | 6.9560 | 0.9937 |
| 3 | 0.2911 | 0.2176 | 0.8491 | 6.9581 | 0.9940 |
| 4 | 0.6144 | 0.4105 | 0.8811 | 6.9087 | 0.9870 |
| 5 | 0.7533 | 0.6157 | 0.8850 | 6.9700 | 0.9957 |
| 6 | 0.8854 | 0.7797 | 0.9192 | 6.9772 | 0.9967 |
| 7 | 0.9370 | 0.8876 | 0.9231 | 6.9873 | 0.9982 |
| 8 | 0.9715 | 0.9504 | 0.9365 | 6.9791 | 0.9970 |
| 9 | 0.9856 | 0.9776 | 0.9700 | 6.9793 | 0.9970 |
| 10 | 0.9929 | 0.9898 | 0.9915 | 6.9524 | 0.9932 |
| 11 | 0.9945 | 0.9948 | 0.7886 | 6.9532 | 0.9933 |
| 12 | 1.0000 | 0.9999 | 0.5708 | 6.8201 | 0.9743 |

  Reading: the noisy tokenizer is behind vq at every level until 11 (level 4: 0.41 vs 0.61) but ends equal on `acc_full`; vq loses codebook usage at the two coarsest levels late in training (0.79 / 0.57) while noisy `eff_rank/7` stays >= 0.97 everywhere — the late coarsest-level degradation is a vq phenomenon.
- Figure C fixtures (`tests/test_plot_tokenizer_levels.py`, 6 tests): the last record containing `acc_full` wins over a trailing record without it; PNG written (>1 KB); the table carries one row per level with the record's numbers and `eff_rank/7`; missing `metrics.jsonl` -> exit 1 with `missing .../<run>/metrics.jsonl`; a file without any `acc_full` -> exit 1 (`no record with 'acc_full'`); differing level counts -> exit 1.
- `RUNBOOK.md` §0.3 "Lite configuration" added: when to use, the `CONFIG`/`--config` commands, and the suffix rule with the `budget.py` filter.

### T5.5 trainer review fixes F11–F15/F17 — re-verified 2026-11-11
- Audit at HEAD `a07f29e`: `GeneratorBackbone.embed_levels` keeps `pooled_reconstructions` under `no_grad` when the tokenizer is frozen and applies `input_embed` outside; `compute_generator_loss` and `joint_loss` share it (F11). `sample_cutoff` draws on `generator.device` (F12). `generator.train.grad_accum` is validated and loops equal micro-batches in both trainers; `jepa_dev.yaml` notes per-micro-batch VICReg statistics (F13). Both dev configs carry the 10-point milestone grid and milestone files omit `optimizer_state_dict` (F14). Both trainers log the four `n_params_*` fields in their first record and `gate_report.py` prints `n_params` (F15). `frozen_codes` wraps encode+corruption in `no_grad` plus `autocast(enabled=False)` (F17).
- Targeted F11–F15/F17 node IDs: `.venv/Scripts/python.exe -m pytest -q <14 node ids>` -> `13 passed, 1 skipped in 9.53s`; the skip is `tests/test_tokenizer.py::test_sample_cutoff_accepts_a_cuda_generator` (`needs CUDA`; this host is CPU-only, so F12's CUDA draw path stays UNVERIFIED).
- Full non-slow suite: `$env:PATH = 'C:\Program Files\Git\bin;' + $env:PATH; .venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `276 passed, 1 skipped, 4 deselected in 63.20s` (EXIT=0).
- No code or test was changed in this re-verification; `docs/TASKS.md` T5.5a/T5.5b stay ticked.

### F16 sweep preflight: FIR sync and environment — 2026-11-10 (zellij `attach`, pane `terminal_0`, login3)
- Local HEAD, `origin/main` and FIR HEAD are identical: local `git rev-parse HEAD origin/main` -> `252fb3b8c6bda7cc5b2620853c8b2f09e2e92084` for both; FIR `cd ~/jncm_git && GIT_TERMINAL_PROMPT=0 git pull --ff-only && git rev-parse HEAD` -> `Already up to date.` / `252fb3b8c6bda7cc5b2620853c8b2f09e2e92084`. Nothing to push; the pull was a no-op. FIR `git status --short` shows only `?? data` (the pre-existing data symlink).
- `~/jncm_git` now has `.venv -> ~/jncm/.venv` and `results -> ~/jncm/results` (created this session; `.gitignore` covers both). `data -> ~/jncm/data` already existed.
- `results/tok_vq_10k/ckpt/latest.pt` exists (130,920,073 bytes) next to `step_005000.pt`/`step_010000.pt`; the final metrics record is `step 10000`, `data_source real`, `acc_full 0.9999847` — the run is complete even though the old tar tree writes no `DONE` file.
- Corruption keys: `jncm/config.py` has `p_first: float = 0.8` and `p_last: float = 0.2` under `CorruptionConfig` (`generator.corrupt.*`); the dev and lite YAMLs both set `mode: "swap"`, `p_first: 0.8`, `p_last: 0.2`, `s_first: 0.5`, `s_last: 0.1`.
- Blocker A (Gen PPL): `ls -ld ~/jncm/models ~/jncm/models/gpt2-large` -> both `No such file or directory`; `echo P=$GPT2_LARGE_PATH` -> `P=`; `find /scratch/$USER /project/$SBATCH_ACCOUNT -maxdepth 4 -iname "*gpt2*"` -> no output. `train_generator.py` builds `GPT2Scorer` at startup when `--gen_ppl` is passed (lines 651-657), so a `GEN_PPL=1` job fails before training.
- Blocker B (wall clock): no `pilot_gen_*` directory exists (`find ~ -maxdepth 4 -type d -name "pilot_gen*"` -> no output) and `sacct -S 2026-11-01` lists no `jncm_gen`/`jncm_eval` job. No generator has run on FIR, so there is no pilot `sec_per_step` for `1.3 * sec_per_step * 2000`.
- Blocker C (memory risk, not measured): `configs/tinystories_dev.yaml` has `batch_size: 1024` and `grad_accum: 1`; the backbone packs `[B*K, L, H]` = 12,288 x 128 tokens through 12 x 768 layers, which is the F13 case estimated at about 490 GB of activations on an 80 GB H100. A 2,000-step dev run with only the `p_first` overrides would most likely OOM.
- Decisions (user, 2026-11-10): keep `configs/tinystories_dev.yaml`, add `generator.train.grad_accum=16` as a second `SETS` entry (effective batch stays 1024), and measure the wall clock with `pilot_gen_vq_ce` first.
- Timing pilot: `sbatch` job **3234940** (`jncm_gen`, 100 steps, `SETS=generator.train.grad_accum=16`, `GEN_PPL=0` because no GPT-2, `--time=0:20:00`) submitted 2026-11-10; first poll `PD (Priority)`, 1 CPU, `gres/gpu:h`, no output file yet. `GEN_PPL=0` does not affect per-step timing here: `generator.train.eval_every=500` is above the 100-step pilot, so no scoring hook runs; the 1.3x factor is meant to cover the full run's evals.
- Slurm quirk found: on this cluster a command-line `--time=0:20:00` does **not** override the template's invalid `#SBATCH --time=HH:MM:SS` header (`sbatch: error: Invalid --time specification`, reproduced with `--test-only` and with `-t`). A `--wrap` job with the same `-t` parses fine. Workaround used: `sed -e s/HH:MM:SS/0:20:00/ scripts/slurm/gen.sbatch > results/.gen_pilot.sbatch` and submit that copy (git-ignored, repo config untouched). The same workaround is needed for the three real runs with their computed wall clock.
- FIR venv packages for Gen PPL: `source .venv/bin/activate && pip install --no-index transformers` -> `Successfully installed ... transformers-5.22.1+computecanada ... tokenizers-0.23.2+computecanada ... safetensors-0.8.0+computecanada ... huggingface-hub-1.33.0+computecanada` (all from the CVMFS wheelhouse, about 4 min). `transformers`, `tokenizers`, `safetensors` and `huggingface_hub` were absent before this; the pilot's `GEN_PPL=0` path does not need them, but the three `GEN_PPL=1` runs do.

### FIR channel, sync check and tokenizer poll — 2026-11-09 (zellij `attach`, pane `terminal_0`)
- Channel proof: `hostname -f; echo ZNCM-1109-QP4; whoami` -> `login1.int.fir.alliancecan.ca` / `ZNCM-1109-QP4` / `$FIR_USER`.
- Sync check: `cd ~/jncm_git && pwd && hostname -f && echo ZNCM-1109-QP4` -> `-bash: cd: ~/jncm_git: No such file or directory`; `ls -d ~/jncm*` -> `~/jncm  ~/jncm_agent`; `test -d ~/jncm/.git && echo ISGIT || echo NOGIT` -> `NOGIT`; `ls -d /scratch/$USER/jncm*` -> nothing. So no clone exists to pull, HEAD could not be compared, and nothing was improvised (no credentials typed).
- Data/models: `ls -l ~/jncm/data ~/jncm/models` -> `ls: cannot access '~/jncm/models': No such file or directory`, `data/tinystories` present (`du -sh` -> `900M`); `echo P=$GPT2_LARGE_PATH` -> `P=` and `ls -d models models/gpt2-large` -> `No such file or directory` for both; `find ~ -maxdepth 3 -name "gpt2*"` -> nothing. `GPT2_LARGE_PATH` does not resolve on FIR, so the Gen PPL hook cannot run there yet.
- Job state: `squeue -u $USER -o "%.8i %.14j %.8T %.10M %.11l"` -> `2880035 jncm_tok PENDING 0:00 4:00:00` (other rows are another project's `nr-*` jobs, untouched); `sacct -j 2880034,2880035 --format=JobID,JobName%20,State,Elapsed,ExitCode` -> `2880034 jncm_tok COMPLETED 02:21:20 0:0` and `2880035 jncm_tok PENDING 00:00:00 0:0`; `ls -l results/tok_vq_10k/DONE results/tok_noisy_10k/DONE` -> `No such file or directory` for both (expected: that tree has no `write_done`), `results/tok_noisy_10k` absent.
- `tok_vq_10k` last record (`tail -n1 results/tok_vq_10k/metrics.jsonl`, 21 records): `"step": 10000`, `"data_source": "real"`, `"acc_full": 0.9999847412109375`, `"acc_cum": [0.0776, 0.1442, 0.2911, 0.6144, 0.7533, 0.8854, 0.9370, 0.9715, 0.9856, 0.9929, 0.9945, 0.99998]`, `"codebook_usage": [0.104, 0.730, 0.849, 0.881, 0.885, 0.919, 0.923, 0.937, 0.970, 0.991, 0.789, 0.571]`, `"codebook_perplexity": [401.5, 2426.9, 2645.8, 2577.2, 2661.6, 2655.7, 2744.7, 2775.8, 2707.0, 3185.6, 1689.9, 437.4]`. Gate verdict: PASS vs the 0.999 vq threshold, `acc_cum` rises gradually, but the two coarsest levels (11/12) lose codebook usage and perplexity late in training — worth watching on the noisy arm.

### T5.5b review fixes F14/F15/F17, lite configs, CONFIG/SETS — VERIFIED 2026-11-09
- F14 focused (earlier this session): `.venv/Scripts/python.exe -m pytest -q` over the new milestone/config/gate-report tests -> `5 passed`. The dev configs now list `[0.1, ..., 1.0]`; a milestone checkpoint has no `optimizer_state_dict` while `latest.pt` still has it; `run_eval.py` loads an optimizer-less milestone; `gate_report.py` interpolates the denser curve.
- F15 focused (earlier this session): trainer parameter-count tests -> `5 passed`; `.venv/Scripts/python.exe -m pytest -q tests/test_run_eval.py tests/test_gate_report.py tests/test_utils.py` -> `22 passed`. The first metrics record carries `n_params_tokenizer`/`n_params_backbone`/`n_params_head`/`n_params_total`; later records do not; eval metadata carries the same four fields and the gate table prints `n_params` as e.g. `123.4M`.
- F17 focused (earlier this session): `.venv/Scripts/python.exe -m pytest -q` over the new autocast test -> `6 passed`. `frozen_codes` runs encode+corruption under `no_grad` and `torch.autocast(enabled=False)`; on CPU a bf16-autocast staged step produces targets `torch.equal` to the fp32 step.
- Lite/CONFIG focused: `.venv/Scripts/python.exe -m pytest -q tests/test_config.py::test_lite_configs_scale_only_the_generator tests/test_jepa_configs.py::test_jepa_lite_is_the_lite_phase2_run tests/test_sbatch_config.py tests/test_launch_phase.py::test_config_is_forwarded_to_every_template tests/test_launch_phase.py::test_g2_forwards_the_jepa_config tests/test_budget.py::test_accepts_and_prints_the_config tests/test_budget.py::test_missing_config_fails_loudly` -> `12 passed in 4.08s` (EXIT=0).
- Related files: `.venv/Scripts/python.exe -m pytest -q tests/test_launch_phase.py tests/test_budget.py tests/test_jepa_sbatch.py tests/test_eval_sbatch.py tests/test_config.py tests/test_jepa_configs.py` -> `46 passed in 12.53s` (EXIT=0).
- SETS: `.venv/Scripts/python.exe -m pytest -q tests/test_sbatch_config.py tests/test_jepa_sbatch.py` -> `11 passed in 0.41s` (EXIT=0). `gen.sbatch` strips an optional `--set=`, requires `key=value`, appends one `--set` per entry, and exits 2 on `SETS=not-an-override`.
- Full non-slow suite: `$env:PATH = 'C:\Program Files\Git\bin;' + $env:PATH; .venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `276 passed, 1 skipped, 4 deselected in 61.50s` (EXIT=0).
- Lite configs: `configs/tinystories_lite.yaml` and `configs/jepa_lite.yaml` equal their dev configs except H 512, 8 layers, 8 heads and batch 256; the JEPA lite file keeps `head: flow`, `target_chain: online` and the dev ablation flags. `CONFIG` defaults to the template's dev config; an explicitly set missing path exits 1. `launch_phase.sh --config` forwards `CONFIG=` to every sbatch export; `budget.py --config` prints the file's POSIX path.

### T5.5a review fixes F11–F13 — VERIFIED 2026-11-09
- New tests: `.venv/Scripts/python.exe -m pytest -q <11 node ids>` (the F11 staged/joint gradient tests, F12 CPU/CUDA cutoff tests, both F13 accumulation tests, staged rejection/accounting tests, JEPA accumulation/validation tests, config test) -> `11 passed, 1 skipped in 4.47s`; the skip is `tests/test_tokenizer.py::test_sample_cutoff_accepts_a_cuda_generator` (`needs CUDA`; this host runs torch `2.15.0+cpu`, so the CUDA draw path is UNVERIFIED until a GPU host runs it).
- Focused files: `.venv/Scripts/python.exe -m pytest -q tests/test_train_generator.py tests/test_train_jepa.py tests/test_jepa_joint.py tests/test_jepa_configs.py tests/test_tokenizer.py tests/test_generator_backbone.py` -> `78 passed, 1 skipped in 187.60s` (EXIT=0).
- Full non-slow suite: `$env:PATH = 'C:\Program Files\Git\bin;' + $env:PATH; .venv/Scripts/python.exe -m pytest -q -m "not slow"` -> `257 passed, 1 skipped, 4 deselected in 51.76s` (EXIT=0).
- Environment note: the same command without the PATH prefix selects `C:\w64devkit\w64devkit\bin\bash.exe` through `shutil.which("bash")`; `tests/test_launch_phase.py` then fails 28 tests with `syntax error: unexpected "("` at `POSITIONAL=()` (that shell reports no `BASH_VERSION`). With Git bash first the suite is green; no test or script was changed for this.
- F11: before the fix the staged test failed with exactly the five `input_embed.*` parameters missing a gradient, for both `ce`+vq and `flow`+noisy. After it, every backbone and head parameter has a gradient after one staged step in both cases and after one joint step. In joint mode the pooled reconstructions stay in the graph, so with `decoder_through_E=False` and `vicreg=False` the prediction path still leaves `tokenizer.tok_emb.weight.grad` nonzero (SPEC 8 step 3); when the tokenizer is frozen (staged), they run under `no_grad`.
- F12: `sample_cutoff` draws `torch.rand`/`torch.randint` on `generator.device` and then calls `.item()`; the CPU generator test passes and replays a seed exactly.
- F13: `generator.train.grad_accum` defaults to 1. Staged `ce`+vq with dropout 0 and `corrupt.mode=none`: `grad_accum=4` vs 1 give a max gradient difference <= 1e-5, with 4 `compute_generator_loss` calls and one AdamW step per batch. Joint: 2 micro-batches -> 4 `joint_loss` calls and 2 optimizer steps over 2 steps; `joint_positions` still charges the full batch and every record has `sec_per_step > 0`. Both trainers reject a zero or non-divisible `grad_accum`.

### T4.5 launch kit + pilots — VERIFIED 2026-11-08
- `bash -n` on `scripts/slurm/{chain.sh,launch_phase.sh,tok.sbatch,gen.sbatch,jepa.sbatch,eval.sbatch}`: all exit 0.
- Focused: `.venv/Scripts/python.exe -m pytest -q tests/test_chain_sh.py tests/test_launch_phase.py tests/test_budget.py tests/test_done_and_sec_per_step.py`
  `38 passed in 16.40s`  (EXIT=0)
- Full non-slow suite: `.venv/Scripts/python.exe -m pytest -q -m "not slow"`
  `245 passed, 4 deselected in 62.79s (0:01:02)`  (EXIT=0, `set -o pipefail`)
- Trainers: `jncm/{train_tokenizer,train_generator,train_jepa}.py` log `sec_per_step` (wall time since the loop start / steps done in this process) in every metrics record and call `jncm.utils.write_done(...)` after the loop, which writes `results/<run_id>/DONE` (`step`, `sec_per_step`, `finished_at`). Two-step CPU runs of all three trainers: DONE present with `step == 2`, and every record's `sec_per_step` finite and > 0 (vq and noisy tokenizers both checked).
- Templates: `tok/gen/jepa.sbatch` exit 0 with `DONE exists` in the output before a venv, module or dataset is touched (fixture runs; a recording `python` is never called, and a missing TOKENIZER is not checked). `gen.sbatch` gained optional `STEPS` and `GEN_PPL=1` -> `--gen_ppl`, both needed by the 300-step pilots.
- `chain.sh` (fake `sbatch` on PATH): 3 chunks -> ids 1001/1002/1003, chunk 1 without `--dependency`, chunks 2-3 with `afterany:<previous>`; an external `--dependency=afterany:900` survives on chunk 1 and merges on chunk 2; `--array`/`--gres` forwarded; unset `SBATCH_ACCOUNT` -> exit 2.
- `launch_phase.sh` dry runs: g0 pilot prints tok(500 steps, `--time=00:20:00`) -> gen(300 steps, `GEN_PPL=1`) -> eval(`N_SAMPLES=32,N_SEEDS=1`, `--array=0-0`) and no full run id; g1 pilot prints 5 sbatch lines (tok + flow + regress + 2 evals); g2 pilot prints only `pilot_jepa_base` plus 6 `LIST` lines and 14 sbatch lines with `--ablations` (7 arms x train+eval); `g0 full` with no pilot record and no `--sec_per_step` exits 1 with the `--sec_per_step` message.
- `budget.py` (no results on this host): one MISSING line per pilot and exit 1, as designed. Fixtures: 12 pilots at 1.0 s/step -> `TOTAL: 33.33 H100-hours over 12 run(s)`; a 5.0 s/step pilot -> `FLAG jepa_base_dev` and exit 1; the latest pilot record wins over an earlier one.
- `RUNBOOK.md` §0.1 paste-back snippet extracted verbatim and compiled: `SNIPPET_LINES 21`, `SNIPPET_EXIT=0`.

### T4.4 gate report + joint eval support — VERIFIED 2026-11-08
- Focused: `.venv/Scripts/python.exe -m pytest -q tests/test_gate_report.py tests/test_eval_sbatch.py`
  `11 passed in 0.28s`  (EXIT=0)
- Full non-slow suite: `.venv/Scripts/python.exe -m pytest -q -m "not slow"`
  `207 passed, 4 deselected in 52.11s`  (EXIT=0, `set -o pipefail`)
- `scripts/gate_report.py` CLI on this host (no cluster results yet): prints the header, four `MISSING (missing results/<run>/eval)` lines and exit 1, as designed.
- `bash -n scripts/slurm/eval.sbatch` -> `BASH_N_EXIT=0`.
- `scripts/slurm/eval.sbatch`: `TOKENIZER` is optional (the path is only checked when `[[ -n "${TOKENIZER:-}" ]]`) and `--tokenizer` is appended to the `run_eval.py` argv only when it is set, so a joint `train_jepa` checkpoint evaluates without it. The tests run the script body in a fixture tree whose `.venv/bin/activate` defines a recording `python` function: unset `TOKENIZER` -> exit 0 with no `--tokenizer` in the argv; a tokenizer path -> `--tokenizer <path>` in the argv; a missing tokenizer path -> exit 1 with `not found` and no python call. Before this task the unset case aborted at the `:?` guard, so `test_joint_run_needs_no_tokenizer` is the regression test.
- `scripts/gate_report.py` fixture tests (`tests/test_gate_report.py`, 6 tests, tmp_path only): one table row per milestone carrying `18.40 +/- 0.10`, `1.704` (18.40/10.80) and `-0.080` (H delta), with per-seed shards ignored; G0 PASS/FAIL per milestone (`G0 step 100 (50%): PASS`, `G0 step 200 (100%): FAIL`, `G0: FAIL`, exit 1); G1 flow interpolating G0 between two milestones (17.00 vs interpolated 15.00 -> `ratio 1.133` PASS) and printing `not comparable at total_positions 1000 (outside gen_vq_ce_dev [100, 300])`; regress reported only with `FLAG pairwise_agree > 0.5`; ratio 1.200 -> `G1 (gen_noisy_flow_dev): FAIL` and exit 1; G2 under both accountings (`G2 (accounting: total_positions): PASS (1 compared)` with `best Phase-1 gen_noisy_flow_dev 12.00 +/- 0.50 (limit 12.50)`, and `G2 (accounting: total_positions_conservative): not comparable`), a worse JEPA arm FAILing, and missing runs printing `MISSING` with exit 1.
- No real numbers were produced: every gate-report fixture lives under pytest `tmp_path`, and this host still has no real eval aggregates (no cluster runs).

### T5.4 JEPA configs, sbatch, joint eval, Phase 2 runbook — VERIFIED 2026-11-08
- Focused: `.venv/Scripts/python.exe -m pytest -q tests/test_jepa_configs.py tests/test_jepa_sbatch.py tests/test_plot_curves.py tests/test_run_eval.py tests/test_train_jepa.py`
  `37 passed in 29.84s`  (EXIT=0)
- Full non-slow suite: `.venv/Scripts/python.exe -m pytest -q -m "not slow"`
  `198 passed, 4 deselected in 48.27s`  (EXIT=0, `set -o pipefail` + `PYTEST_EXIT=0`)
- `bash -n` on all five templates (`prep tok gen eval jepa`): each `exit=0`.
- `configs/jepa_dev.yaml` loads and `validate_jepa_config` accepts it: head flow, quantizer noisy, `target_chain online`, EMA/VICReg/decoder-through-E on, batch 1024, 10,000 steps, tokenizer d 384 / K 12 and generator H 768 / 12 layers identical to `configs/tinystories_dev.yaml`. `configs/jepa_smoke.yaml` loads on CPU (L 32, V 64, K 4, head regress, batch 16) and is accepted too; both files state that `generator.head` must be `regress` or `flow`.
- `scripts/slurm/jepa.sbatch`: `RUN_ID`/`TOKENIZER` required, `STEPS` optional (config budget otherwise), `SETS` a space-separated `key=value` list each passed as `--set`, resume from `results/<RUN_ID>/ckpt/latest.pt` only when it exists, no account/partition in any `#SBATCH` line, the three `HF_*_OFFLINE` variables.
- `jncm/train_jepa.py` writes `ckpt/milestone_{010,025,050,100}.pt` behind `generator.train.save_milestones` (joint state plus `milestone`); with the flag off it writes none (`test_milestones_are_saved_behind_the_flag`).
- Joint eval: a 20-step CPU `train_jepa` run (configs/jepa_smoke.yaml shrunk to d 32 / H 32 / batch 4) wrote the four milestones; `run_eval` with `tokenizer_ckpt=None` and a fake scorer evaluated them, and `eval/step_000020.json` carries `ppl_mean`/`ppl_std`, `h_gpt2_mean`, `h_dataset_mean`, `unique_frac_mean`, `pairwise_agree_mean`, `joint_positions = total_positions = 2560` and `total_positions_conservative = 5120`, with `tok_positions`/`gen_positions` absent; `--merge` preserves the conservative field.
- `plot_curves.py --x total_positions_conservative` places a joint record at `2 x joint_positions`, leaves staged records on their own `total_positions`, rejects any other key, and renders a PNG under both accountings.
- `RUNBOOK.md` §3A.5 paste-back snippet extracted verbatim and compiled: `SNIPPET_SYNTAX_EXIT=0`, 27 lines.

### T5.3b online target chain + joint accounting — VERIFIED 2026-11-08
- Focused: `.venv/Scripts/python.exe -m pytest -q tests/test_jepa_joint.py tests/test_train_jepa.py`
  `25 passed in 22.50s`  (EXIT=0)
- Full non-slow suite: `.venv/Scripts/python.exe -m pytest -q -m "not slow"`
  `188 passed, 4 deselected in 44.64s`  (EXIT=0, `set -o pipefail` + `PYTEST_EXIT=0`)
- `jncm/jepa/joint.py`: `encode_online_targets` builds `zbar = Ebar.encode_trunk(x)` once and `t_k = LayerNorm(Ebar.proj_k(down(zbar - detach(zhat_{k-1}), l_k)))` from the online chain computed before corruption. With E-bar a fresh `EMAModule` copy of E, the `online` targets are `torch.equal` to `tokenizer.encode(x, generator=seed).targets` at every level K (L=16, V=16, K=4, d=16) for the same seed, and they carry `requires_grad False` and `grad_fn None` in both chains.
- `ema_clean`: targets are `torch.equal` to `encode_clean(E-bar, x)`. Distinguishing probe: with E-bar = E and the online sigma raised from 0.4 to 2.0, `ema_clean` targets stay bit-identical while the `online` targets change; level 1 is identical on both chains (`zhat_0 = 0`), levels 2..K differ at sigma 0.4.
- Accounting: every joint metrics record carries `joint_positions`, `total_positions = joint_positions` and `total_positions_conservative = 2 * joint_positions`; `tok_positions`/`gen_positions` are absent (they stay on staged tokenizer/generator runs). A 3-step run records `[64, 128, 192]` (`B*L = 64`); the checkpoint stores all three fields and keeps the tokenizer budget as `tok_offset = 1234`, uncharged.
- 50 CPU steps with `head=flow` run for both `target_chain` values (VICReg on): finite loss/pred, `nets.*` in the checkpoint, `joint_positions = 50*B*L`, `target_chain` logged in every record. `validate_jepa_config` rejects an unknown `jepa.target_chain`.

### T5.3 JEPA joint trainer — VERIFIED 2026-11-07
- Focused: `.venv/Scripts/python.exe -m pytest -x -q tests/test_jepa_joint.py tests/test_jepa_diagnostics.py tests/test_train_jepa.py`
  `23 passed in 16.71s`  (EXIT=0)
- Full suite: `.venv/Scripts/python.exe -m pytest -x -q`
  `186 passed in 198.71s (0:03:18)`  (EXIT=0)
- `jncm/jepa/joint.py` + `jncm/jepa/diagnostics.py`: `encode_clean` equals the noisy chain at sigma = 0 (`torch.equal` for both `targets` and `zhat` at all K levels), and with sigma 0.4 every chain code differs; raising the online sigma from 0.4 to 2.0 leaves the EMA targets bit-identical (they come from E-bar's own noise-free chain) while the online chain codes change. `L = L_dec + lambda_pred * L_pred + L_vic` holds exactly; VICReg off gives `loss_vic = 0`; zeroed regress heads (collapsed c-hat) give `loss_vic >= K * 0.95`; `decoder_through_E=False` leaves the tokenizer trunk gradient zero/None while the decoder still trains; `predicted_codes` equals the raw regress output and is seed-reproducible for `flow`.
- Diagnostics: iid N(0,I) `[4096, 8]` effective rank > 7, a constant gives 1.0, rank-one data < 1.2; R2(pred=target) = 1, R2(zeros, zero-mean unit-var target) ~ 0, constant target -> 0.0; `code_diagnostics` returns K-length finite lists with `eff_rank` in [1, d_k]; the alarm fires on the 3rd consecutive sub-0.1 step with patience 3, one healthy step resets it, default patience 200.
- `jncm/train_jepa.py`: all four (`decoder_through_E`, `vicreg`) combinations run 50 CPU steps at smoke overrides (L=16, V=16, K=4, d=16, H=16), writing `config.yaml`, `git_sha.txt`, `metrics.jsonl`, `ckpt/latest.pt`; every record carries `loss_dec`/`loss_pred`/`loss_vic`, `std_min`/`std_mean` of c_k and t_k, both effective ranks, `r2`, `collapse_steps`, `collapse_alarm` and the four flags; its position accounting was superseded by T5.3b (SPEC v1.4 section 6.2); the checkpoint has `tokenizer`, `backbone`, `head`, `optimizer_state_dict`, `ema`, `collapse_alarm`, `tok_offset`.
- EMA wiring: after one joint step the saved target `tok_emb.weight` equals `0.996 * initial + 0.004 * online` exactly; with `jepa.ema=false` the checkpoint saves `ema=None` and no `ema_momentum` field.
- Smoke overfit (the TASKS acceptance): 300 CPU steps on one fixed batch of 4 sequences recorded CE 3.02 -> 0.73 and mean R2 -0.384 -> +0.338 (300-400 step probes spanned CE 0.26-0.98 and mean R2 +0.19..+0.39); the test asserts CE < 40% of the first record, mean R2 gain > 0.4 and mean R2 > 0.15.
- Flow head runs 10 CPU steps with finite loss/R2 and a `nets.*` checkpoint; resume from step 2 to 4 continues at step 4 with `tok_positions = 99 + 4*B*L`; `validate_jepa_config` rejects `vq` + `ce` and negative weights, and `train` rejects before touching the checkpoint path.
- CLI smoke (throwaway `.tmp_t53`, deleted afterwards; nothing written to `results/`): 3-step synthetic run printed
  `Step     1 | Loss: 10.7432 | dec 4.2952 | pred 1.6979 | vic 4.7500 | R2 -0.658 | tok_pos: 228 | gen_pos: 128`
  `Step     3 | Loss: 10.5794 | dec 4.4024 | pred 1.5154 | vic 4.6617 | R2 -0.620 | tok_pos: 484 | gen_pos: 384`
  `2` metrics records, `latest.pt` and `step_000003.pt`, exit 0.

### T5.1 / T5.2 / g0a_report --runs — VERIFIED 2026-11-07
- `.venv/Scripts/python.exe -m pytest -q tests/test_g0a_report.py tests/test_jepa_ema.py tests/test_jepa_vicreg.py`
  `16 passed in 2.21s`  (EXIT=0)
- Full non-slow suite: `.venv/Scripts/python.exe -m pytest -x -q -m "not slow"`
  `159 passed, 4 deselected in 25.91s`  (EXIT=0)
- `scripts/g0a_report.py` fixtures: the default runs are `tok_vq_10k`/`tok_noisy_10k`; an explicit `--runs` pair is judged with 0.999 (vq) / 0.995 (noisy) and prints `PASS|FAIL <run_id> (<variant>) ... vs <threshold>`; a run without `config.yaml` falls back to its id; an unknown variant exits 1.
- `jncm/jepa/ema.py`: hand arithmetic `0.996*[1,-2] + 0.004*[2,3]`; target parameters `requires_grad=False` and independent of the online module; eval mode survives `parent.train()` and `ema.train(True)`; `cosine_momentum` = 0.996 at step 0, 0.998 at T/2, 1.0 at T and beyond, non-decreasing; `update` uses the scheduled momentum.
- `jncm/jepa/vicreg.py`: iid N(0,I) `[4096, 8]` gives var < 0.02 and cov < 0.01; a constant gives var = gamma (0.9999 with eps 1e-8, exactly 1 with eps 0) and cov = 0; two correlated dimensions give cov > 0.1; gradients through both terms are finite, including a fully collapsed input.

### T4.3 runbook — VERIFIED 2026-11-07
- `.venv/Scripts/python.exe -m pytest -q -m "not slow"`
  `143 passed, 4 deselected in 22.26s`  (EXIT=0)
- Both paste-back snippets in `RUNBOOK.md` §4.2 and §4.4 were run verbatim on throwaway fixtures under `.tmp_runbook/` (deleted afterwards; nothing written to `results/`): the tokenizer dump printed `step 10000 acc_full 0.9994 ... data_source real` then `PASS vs 0.999`; the eval dump printed `step 100 total_positions 12345678 ppl 18.400 std 0.500 ppl/10.80 1.704 h_gpt2 3.910 h-4.03 -0.120` and `n/a` for a record without `ppl`.
- Wall-clock audit: `SBATCH_LINES=8 WITHOUT_WALL=0 WALL_FIELDS=10` — every `sbatch` submit line in `RUNBOOK.md` carries `--time=<WALL_CLOCK>`, and no wall clock is guessed.
- Runbook commands are taken from the four `scripts/slurm/*.sbatch` files and the CLIs: `gen.sbatch`/`eval.sbatch` take `HEAD`/`RUN_ID`/`TOKENIZER` (passed via `--export=ALL,...`), `eval.sbatch` uses `--array=0-4`, `run_eval.py --merge` needs only `--run_id`, `plot_curves.py` writes `curves.png`.
- Limitation: `scripts/g0a_report.py` keys its thresholds on `tok_vq_dev`/`tok_noisy_dev`, so the 10k gate verdict uses the §4.2 record dump (KNOWN ISSUES 10); no code was changed in T4.3.

### T0.2 real data prep — VERIFIED 2026-11-07
- `data/tinystories/meta.json`: `train_windows 3647424`, `val_windows 36658`, `L 128`, `V 8192`; `train.bin` 933,740,544 bytes = 3,647,424*128*2 and `val.bin` 9,384,448 bytes = 36,658*128*2.
- Token-id scan over the first and last 1,000,000 tokens of both bins: max id 8191 (train) and 8191 (val, tail max 8190), all < V.
- Val windows 0 and 36657 decode with `tokenizer.json` to readable English prose.
- `python scripts/prep_tinystories.py --max_stories 2000 --out data/tiny_test`
  `Train: 3210 windows (410880 tokens)` / `Val:   637 windows (81536 tokens)`; `data/tiny_test/meta.json` has L 128, V 8192 and both bin sizes match exactly. (The shell exit 1 was PowerShell treating the HF auth warning on stderr as an error; the script printed success and the files are complete.)
- `pytest -q tests/test_data_prep.py tests/test_eval_data_row.py tests/test_genppl.py`
  `13 passed in 2.03s`

### T1.2 protocol gate — VERIFIED 2026-11-07
- `GPT2_LARGE_PATH=models/gpt2-large python scripts/eval_data_row.py --data data/tinystories/val.bin --tokenizer data/tinystories/tokenizer.json --gpt2 models/gpt2-large --n_samples 256 --out results/t1_2_protocol_gate.json`
  `Gen PPL:           10.80 +/- 0.26 (Target: ~9.67, tol 15% [8.22, 11.12])`
  `H (GPT-2 tokens):  4.03 +/- 0.01 (Target: ~3.96, tol 0.15 [3.81, 4.11])`
  `H (Dataset tokens):4.03 +/- 0.01`
  `Verdict: PASS`
- `results/t1_2_protocol_gate.json`: `n_samples 256`, `n_seeds 5`; PPL runs 10.46–11.23, H std <= 0.008. Metric, tokenizer and thresholds unchanged.

### T4.2 eval and plots — VERIFIED 2026-11-07
- `bash -n` on all four templates from the repo root (`for f in prep tok gen eval; do bash -n scripts/slurm/$f.sbatch; done`): all exit 0.
- Focused: `.venv/Scripts/python.exe -m pytest -q tests/test_genppl_batched.py tests/test_run_eval.py tests/test_plot_curves.py tests/test_eval_sbatch.py tests/test_genppl.py`
  `27 passed in 6.10s`  (EXIT=0)
- Full suite: `.venv/Scripts/python.exe -m pytest -x -q`
  `147 passed in 176.23s (0:02:56)`  (EXIT=0)
- Batched vs one-at-a-time scoring with the real `models/gpt2-large` (4 texts of different lengths, CPU):
  `tokens ref 35 batch 35`
  `mean_nll ref 3.18554666 batch 3.18554709 diff 4.360e-07`
  The batched path is padding-safe and matches `score_sample` far inside the 1e-4 requirement.
- End-to-end CLI smoke on a throwaway 4-step smoke run (real GPT-2; temp dir deleted afterwards, no numbers written to `results/`):
  `milestones: ['milestone_010.pt', 'milestone_050.pt', 'milestone_100.pt']`
  `step 4 (1.0): Gen PPL 259.84 +/- 19.05, total_positions 2560`
  `wrote 9 eval records under .tmp_t42_e2e/results/gen_tiny/eval`
  `gen_tiny: 3 checkpoints` followed by `wrote .tmp_t42_e2e/curves.png` (113,500 bytes).
- Tests cover: batching (padding-safe token counts, NLL equal to `score_sample`, `score_texts` delegation, `gen_ppl` agreement); eval JSON fields (PPL mean/std = exp(2), H on both token variants, `unique_frac`, `pairwise_agree`, positions, seed shards); checkpoint-config validation and rebuilding from the checkpoint's own architecture; milestone saving behind the flag plus `milestone_steps`; shard merging; plot fixtures (tokenizer offset, shard filtering, PNG); `bash -n` and the T4.1 conventions.
- T1.2 was not re-run: the batched path is its only changed code, and the real-model probe above bounds the difference at 4.4e-7 mean NLL.

### G0a cluster run — in progress (2026-11-07)
- Local tree: `git status` clean after committing `AGENTS.md` (FIR operator rules); one empty untracked `docs/test.py` remains.
- Fast suite: `.\.venv\Scripts\python.exe -m pytest -q -m "not slow"`
  `122 passed, 4 deselected in 19.36s`  (EXIT=0)
- Real data already present: `data/tinystories/{train,val}.bin` + `tokenizer.json` + `meta.json`; meta `train_windows 3647424`, `val_windows 36658`, `L 128`, `V 8192`. `scripts/prep_tinystories.py` was not re-run.
- `scripts/slurm/tok.sbatch` placeholders filled: `--time=12:00:00` (template default, overridden per submit), `--cpus-per-task=6`, `--mem=32000M`, `module load python/3.12`, venv `.venv` unchanged; optional `STEPS` / `RUN_ID` overrides (`RUN_ID` defaults to `tok_<quant>_dev`, `STEPS` unset = config budget). `bash -n` (Git bash) exit 0.
- FIR target (from user): host `login2.int.fir.alliancecan.ca`, user `$FIR_USER`, repo `~/jncm`, python module `python/3.12`, no partition flag. FIR rejects the literal account `${SBATCH_ACCOUNT}_`; the valid account is `$SBATCH_ACCOUNT` (Slurm error lists it).
- FIR requires an explicit GPU model: `gpu:h100` exists (`sinfo -o "%G"` shows `gpu:h100:4(S:2-3)`), so `tok.sbatch` now requests `--gres=gpu:h100:1`; a generic `--gres=gpu:1` is rejected at submit time.
- Venv on FIR: `module load python/3.12` (3.12.4), `virtualenv --no-download .venv`, `pip install --no-index torch numpy pyyaml` -> torch 2.14.1+computecanada, numpy 2.5.3, pyyaml 6.0.3.
- Smoke: `python -m jncm.train_tokenizer --config configs/smoke.yaml --steps 5 --set L=128 --set V=8192 --run_id smoke_g0a` -> 5 steps, real data, exit 0. (The first attempt failed loudly as designed: real meta L=128 vs smoke config L=32.)
- Pilot submission: `sbatch --account=$SBATCH_ACCOUNT --gres=gpu:h100:1 --time=00:20:00 --export=ALL,QUANT=vq,STEPS=500,RUN_ID=pilot_vq scripts/slurm/tok.sbatch` -> job 2796353 (partition `gpubase_b`). It later ran on 2026-11-07 and `COMPLETED` in `00:09:05` (sacct); see "G0a tokenizer reconciliation" below.
- `scripts/g0a_report.py` added: reads `results/tok_vq_dev|tok_noisy_dev/metrics.jsonl`, prints step, `acc_full`, rounded `acc_cum`, SPEC 4.5 diagnostics, and PASS/FAIL vs 0.999 (vq) / 0.995 (noisy) on the final record, requiring `data_source=real`. Checked on fabricated temp metrics (pass -> exit 0, below-threshold -> exit 1); no result numbers were written to `results/`.

### G0a FIR operation notes (zellij control)
- Send: `zellij --session attach action write-chars -p terminal_0 '<cmd>'` then `zellij --session attach action send-keys -p terminal_0 Enter`. After a while `action write 10`/`13` stopped submitting the line; `send-keys Enter` kept working.
- Read: `zellij --session attach action dump-screen -p terminal_0 -f --path <file>`. Without `-p terminal_0` the dump came from a blank/plugin pane; even the right pane pads the viewport with empty lines, so filter to non-empty lines before `tail`.
- A long (~210 char) submit line lost the space between `RUN_ID=pilot_vq` and the script path once (`RUN_ID=pilot_vqscripts/slurm/tok.sbatch`); always read the echoed command on screen after long submits.
- Transfer: `ssh -o BatchMode=yes fir true` is denied, so the human ran `scp <repo tar> fir:jncm_agent/` (944,109,056 bytes); extraction was `mkdir -p ~/jncm && tar -xf ~/jncm_agent/jncm_g0a.tar -C ~/jncm`. The tar is still at `~/jncm_agent/jncm_g0a.tar` (no `rm` was run).
- Account `${SBATCH_ACCOUNT}_` is rejected ("You may not be a member"); use `$SBATCH_ACCOUNT`. Generic `--gres=gpu:1` is rejected; `--gres=gpu:h100:1` is accepted and Slurm selects partition `gpubase_b`.
- Venv: `module load python/3.12`, `virtualenv --no-download .venv`, `pip install --no-index torch numpy pyyaml`; smoke `--config configs/smoke.yaml --steps 5 --set L=128 --set V=8192` (real data wins over the smoke L=32 and fails loudly on mismatch).
- FIR channel proven through the zellij pane (`hostname -f` = `login2.int.fir.alliancecan.ca`). `ssh -o BatchMode=yes` from this host is denied (publickey,keyboard-interactive), so every new ssh/rsync needs human MFA.
- G0a numbers: not run yet (pilot 2796353 completed its 500 steps; no full runs).

### G0a tokenizer reconciliation — VERIFIED 2026-11-07 (zellij `attach`, pane `terminal_0`)
- Channel proof: `hostname -f; echo ZNCM-7f31-QP` -> `login1.int.fir.alliancecan.ca` and `ZNCM-7f31-QP`; `echo ACC=$SBATCH_ACCOUNT` -> `ACC=$SBATCH_ACCOUNT`. The zellij pane sits on `login1` this session (the notes above say `login2`; the ssh target is still `fir`). No account literal was typed.
- Inventory before acting: `squeue -u $USER -o "%.10i %.22j %.8T %.10M %.12l"` -> the only `jncm_tok` row was `2800220 ... PENDING 0:00 1-00:00:00`; the other 14 rows are `nr-league-v1-*` / `nr-train` jobs from another project and were left untouched. `sacct -S today -u $USER --format=JobID,JobName%22,State,Elapsed,Timelimit,WorkDir%40` -> `2796353 jncm_tok COMPLETED 00:09:05 00:20:00 ~/jncm` and `2800220 jncm_tok PENDING 00:00:00 1-00:00:00 ~/jncm`; 9 job rows today, 2 ours, none RUNNING.
- Cancellation (the 24 h pending job): `scancel 2800220` -> `SCANCEL=0`; `squeue -u $USER -n jncm_tok` then printed only its header.
- No run-id collision: `ls results/tok_vq_10k/ckpt/latest.pt results/tok_noisy_10k/ckpt/latest.pt` -> `No such file or directory` for both.
- The remote `scripts/slurm/tok.sbatch` is the pre-h100 copy (`#SBATCH --gres=gpu:1`), so both submits pass `--gres=gpu:h100:1` on the command line, exactly as pilot 2796353 did; the file on FIR was not edited.
- Submissions, from `~/jncm` (the account comes from `$SBATCH_ACCOUNT`):
  `sbatch --account=$SBATCH_ACCOUNT --gres=gpu:h100:1 --time=04:00:00 --export=ALL,QUANT=vq,STEPS=10000,RUN_ID=tok_vq_10k scripts/slurm/tok.sbatch` -> `Submitted batch job 2880034`
  `... --export=ALL,QUANT=noisy,STEPS=10000,RUN_ID=tok_noisy_10k ...` -> `Submitted batch job 2880035`
- Scheduler record: `scontrol show job 2880034|2880035` -> `Account=${SBATCH_ACCOUNT}_gpu`, `TimeLimit=04:00:00`, `ReqTRES=cpu=6,mem=32000M,node=1,billing=12200,gres/gpu=1,gres/gpu:h100=1`, `WorkDir=~/jncm`, SubmitLine carrying `QUANT`/`STEPS`/`RUN_ID`; `squeue -u $USER -n jncm_tok -o "%.8i %.8T %.10M %.12l"` -> `2880034 PENDING 0:00 4:00:00` / `2880035 PENDING 0:00 4:00:00`.
- Status: the submissions are VERIFIED; the runs are UNVERIFIED (PENDING, not polled after submit by design). At the pilot's 500 steps in `00:09:05` wall (startup included), a 4 h limit covers roughly 2.5-3k steps, so neither job finishes the 10,000-step budget in one wall clock; resubmitting the same command resumes from `results/<run_id>/ckpt/latest.pt`.

### T4.1 Slurm templates — prep, tok, gen (eval moved to T4.2)
- One command, `set -o pipefail`, from the repo root: `bash -n` on each script, then the full suite.
  `bash -n scripts/slurm/prep.sbatch exit=0`
  `bash -n scripts/slurm/tok.sbatch exit=0`
  `bash -n scripts/slurm/gen.sbatch exit=0`
  `.venv/Scripts/python.exe -m pytest -x -q`
  `123 passed, 1 xfailed in 144.20s (0:02:24)`  (EXIT=0)
- Requirement audit from the same command's `grep -n` output:
  - `TODO(placeholder)` lines: `prep.sbatch` 6 (wall clock), 7 (modules), 9 (venv); `tok.sbatch` 7 (wall clock), 8 (GPU type), 10 (modules), 12 (venv); `gen.sbatch` same four at 7/8/10/12.
  - `^#SBATCH` lines are only `--job-name`, `--nodes`, `--ntasks`, `--gres=gpu:1`, `--time=HH:MM:SS`. No account, no partition, no GPU type anywhere in the three files (`--gres=gpu:1` is generic; the only `gpu:A100:1` text is the example inside a TODO comment).
  - `HF_HUB_OFFLINE=1` + `HF_DATASETS_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`: 3/3 in `tok.sbatch` and `gen.sbatch`, 0 in `prep.sbatch` (deliberate, Q9).
  - `set -euo pipefail` at `prep.sbatch:29`, `tok.sbatch:33`, `gen.sbatch:41`; each script then checks `.venv/bin/activate`, sources it, and exits 1 with a hint if it is missing.
- `python -m jncm.train_tokenizer --help` lists `--resume RESUME` and `--run_id RUN_ID`; loading `configs/tinystories_dev.yaml` with `--set tokenizer.quantizer=noisy` gives `quantizer 'noisy' L 128 V 8192 max_steps 40000 save_every 5000 device cuda`.
- Option plumbing: `tok.sbatch` requires `QUANT` (`: "${QUANT:?...}"`, case-checked to `vq|noisy`) and runs `--set tokenizer.quantizer=${QUANT} --run_id tok_${QUANT}_dev`; `gen.sbatch` requires `HEAD`, `TOKENIZER` and `RUN_ID`, maps `ce -> vq` and `regress|flow -> noisy` (SPEC 5.3), passes `--tokenizer ${TOKENIZER}`, `--set generator.head=${HEAD}`, `--set tokenizer.quantizer=${quant}`, `--run_id ${RUN_ID}`, and fails with exit 1 if the tokenizer checkpoint is absent. Both append `--resume results/<run_id>/ckpt/latest.pt` only when that file exists.
- Not submitted: this host has no Slurm, so the jobs are syntax-checked and inspected only (UNVERIFIED as cluster jobs; see "Human must fill").
- T4.1 TODO markers appear only in `scripts/slurm/*.sbatch` (no `TODO`/`pass`/`...` stub in `jncm/`), and this section lists every one of them.

### T3.5 generator training — frozen tokenizer, all levels per pass, accounting, resume, diversity
- `.venv/Scripts/python.exe -m pytest -q -m "not slow" tests/test_train_generator.py`
  `.............                                                            [100%]`
  `13 passed, 4 deselected in 8.66s`
  Covers (a) `load_tokenizer` returns an eval-mode, `requires_grad`-off tokenizer plus the checkpoint's `tok_positions`; wrong `L`, wrong variant and a missing file raise; (b) the weighted loss equals a manual SPEC 5.1/5.4/5.5 recomputation for `corrupt.mode` `none` and `swap`, and with `swap` the chain codes really change while the targets stay clean; (c) exactly one packed backbone pass per loss, shape `[B*K, L, H]`; (d) records carry `step`, `train_loss`, `per_level_loss`, `lr`, `data_source`, `tok_positions` (offset from the tokenizer checkpoint), `gen_positions = step*B*L`, `total_positions = sum`; (e) resume continues from the checkpoint's step and positions; (f) the eval hook writes `samples/step_<n>.txt` (<= 32 lines), decodes every sample and logs `gen_ppl` (fake scorer gives exactly exp(1) = 2.718), `unique_frac` and `pairwise_agree`; (g) head/variant pairing is enforced (`ce`+vq, `regress`/`flow`+noisy) and `corrupt.mode=noise` on vq raises; (h) the SPEC 6.1 endpoints: identical samples give `unique_frac = 1/n` and `pairwise_agree = 1`, fully different samples with no shared tokens give 1 and 0.
- Slow acceptance (`pytest -m slow`, CPU): tokenizer overfit 700 steps on the 8 fixed sequences (round-trip accuracy 1.000, vq and noisy), generator batch 8, `corrupt.mode=none`.
  - `ce` 600 steps, lr 1e-3: 8 sequences at >= 0.90 (previous run 1.000).
  - `flow` 1200 steps, lr 2e-3: 8 sequences at >= 0.80 (previous run 0.961, distinct matches 4).
  - `regress` one-sequence arm, 1000 steps: >= 0.90.
  - `regress` teacher-forced arm, 2500 steps: the true level-1 code is supplied and levels 2..K are sampled; all 8 sequences >= 0.90 per sequence.
- The old `regress` xfail arm is removed (Q8 resolved, DECISIONS 4). `pytest -q -m slow tests/test_train_generator.py -k teacher_forced` -> `1 passed, 16 deselected in 485.66s (0:08:05)`.
- Full suite: `.venv/Scripts/python.exe -m pytest -x -q`
  `........................................................................ [ 57%]`
  `......................................................                   [100%]`
  `126 passed in 188.08s (0:03:08)`  (EXIT=0)

### T3.4 sampler — exactly K backbone calls, shapes, seeded determinism
- `pytest -q tests/test_generator_sample.py`
  `..............                                                           [100%]`
  `14 passed in 3.00s`
  Covers (a) a forward hook on the backbone sees exactly K calls, with inputs `[B, l_k, H]` for the SPEC schedule, at B = 1 and B = 3; (b) the count is K for K = 1 as well as K = 4; (c) tokens `[B, L]`, dtype long, ids in `[0, V)` for `ce`/`regress`/`flow`; (d) same `torch.Generator` seed gives `torch.equal` samples and a different seed gives different samples, for all three heads; (e) the sampler reproduces the SPEC 5.6 pseudocode step for step (manual `embed_level` -> `backbone` -> `head.sample_level` -> `contrib` loop with the same seed, `flow` head, `torch.equal`); (f) mismatched head K and `batch_size=0` raise; (g) the sampler leaves module train/eval modes untouched.
- Test-teeth probes (temporary source mutants, then restored; `grep -c MUTANT` = 0 afterwards):
  `MUTANT 1 (drop zhat accumulation)`: `4 failed, 10 passed in 4.18s` (all three seeded tests plus the manual-loop test)
  `MUTANT 2 (extra backbone call at level 1)`: `4 failed, 10 passed in 2.42s` (both call-count tests)
- Full suite from repo root: `.venv/Scripts/python.exe -m pytest -x -q`
  `........................................................................ [ 66%]`
  `.....................................                                    [100%]`
  `109 passed in 18.94s`  (EXIT=0)

### T3.3 corruption — swap rate, clean targets, identity `none`
- `pytest -q tests/test_generator_corruption.py`
  `.........                                                                [100%]`
  `9 passed in 2.00s`
  Covers (a) empirical swap rate per level within 0.03 of `p_j` (levels 1/11/22/32, B=1024); (b) every swapped code is another (batch, position) of its own level, and at rate 1.0 nothing stays put; (c) targets untouched through the real frozen tokenizer (and `enc.codes` unmodified; the corrupted conditioning does change the backbone output, so the check is not vacuous); (d) `mode="none"` returns the identical tensors; (e) seeded determinism plus out-of-place behaviour; (f) `noise` empirical std/mean per level; (g) `noise` rejected unless `quantizer="noisy"` and rejected for integer codes; (h) `linear_schedule` endpoints (0.5/0.1 for noise), wrong rate count, rate out of [0,1] and negative sigma raise.
- Probe used for the numbers above (B=1024, seeds 3/11, schedule [1, 11, 22, 32]):
  `swap target [0.8, 0.6, 0.4, 0.2]`
  `swap observed [0.8164, 0.6009, 0.3957, 0.1959]`
  `noise target [0.5, 0.3667, 0.2333, 0.1]`
  `noise observed std [0.5001, 0.3662, 0.2338, 0.1001]`
- Full suite from repo root: `.venv/Scripts/python.exe -m pytest -x -q`
  `........................................................................ [ 75%]`
  `.......................                                                  [100%]`
  `95 passed in 16.90s`  (EXIT=0)

### T3.2 heads and losses — ce/regress overfit, flow bimodal toy, determinism
- `pytest -q tests/test_generator_heads.py`
  `.............                                                            [100%]`
  `13 passed in 11.82s`
  Covers (a) `ce` overfits a fixed batch through the real frozen-tokenizer → backbone → head path: all 66 code indices at all 4 levels predicted exactly, total loss < 1e-2; (b) `regress` overfits the same path to total MSE 4.62e-4 (per level <= 3.1e-4), cosine >= 0.99; (c) `flow` learns the 2-D two-Gaussian toy (modes +-(1,1), std 0.1): 97.3% of 512 samples within 0.5 of a mode and a 0.453 / 0.547 split (thresholds 35% each, 80% near a mode); (d) seeded determinism for all three samplers (same generator → `torch.equal`, different seed → different samples) and for the flow training loss; (e) `ce` loss equals `F.cross_entropy` of the head logits, `regress` cosine option matches `1 - cos`; (f) `regress` renorm gives per-position unit variance; (g) SPEC 5.5 weights (`alpha=0` position-weighted, `alpha=1` equal per level, `sum_k w_k l_k = 1`) and the factory's argument validation.
- Overfit-batch sizing probe (batch size 1 for a reason, see ASSUMPTIONs): `H=32 L=1 steps=1200 lr=0.005: loss=4.62e-04 per=[1e-06, 0.000107, 0.000308, 4.7e-05] t=4.8s`; `H=64 L=2 steps=600: loss=1.72e-02`. Toy probe at 600 steps: `loss 0.8926 frac+ 0.453 near<0.5 0.973 xmean [-0.069, -0.07] xstd [0.951, 0.957]` (the two-mode target has an irreducible ambiguity floor near 0.86, so the toy test checks bimodality and a 40% loss drop, not a small absolute loss).
- Full suite from repo root: `pytest -x -q`
  `........................................................................ [ 83%]`
  `..............                                                           [100%]`
  `86 passed in 16.56s`  (EXIT=0)

### T3.1 backbone — no-leak, padding invariance, shapes
- `pytest -q tests/test_generator_backbone.py`
  `.........                                                                [100%]`
  `9 passed in 2.04s`
  Covers (a) for k = 1..K, level-k outputs are bit-identical (`torch.equal`) when the chain codes at levels >= k are replaced, for both `vq` and `noisy`; (b) level-k output does change when only level k-1 codes change, so the no-leak test is not vacuous; (c) garbage at padded positions (`randn*100`) leaves every valid output bit-identical through `forward_levels`; (d) shapes `[B, l_k, H]` for all K levels plus the single-level `embed_level`/`forward` path, which matches the packed path to 1e-5; (e) `embed_levels` raises when fewer than K-1 prefix codes are given.
- Test-teeth probe (monkeypatched mutants, then restored):
  `no-leak (level k sees its own code): mutant killed: vq: level 1 output changed when only codes of levels >= 1 changed`
  `padding (mask dropped): mutant killed: padding leaked into level 1`
- Full suite from repo root: `pytest -x -q`
  `........................................................................ [ 98%]`
  `.                                                                        [100%]`
  `73 passed in 8.43s`

### T2.4 regression tests and full suite
- Baseline before fixes: `pytest -q tests/test_review_fixes.py`
  `6 failed, 4 passed in 1.85s`
  Failures: F2[vq] (`index_add_` Float vs BFloat16), F3 missing `WindowPacker`, F4 explicit-synthetic kwarg, F4 meta mismatch, F6 unknown key, F10 schedule `[1, 2, 4]`.
  Already passing at baseline: F1 (QKV was fixed last session), F2[noisy], F4 no-silent-fallback, F7 (straight-through VQ is the intended design).
- Final: `pytest -q tests/test_review_fixes.py`
  `............                                                             [100%]`
  `12 passed in 6.29s`
- Final full suite from repo root (venv on PATH): `pytest -q`
  `................................................................         [100%]`
  `64 passed in 8.67s`
- F1 right-reason probe:
  `params 49984 expected 49984 has_qkv_out True`
  (exact 12*d^2+13*d parameter count; learned `qkv` and `out_proj` exist.)
- F4 right-reason probe:
  `missing: FileNotFoundError Prepared data not found at missing (need train.bin and val.b`  
  `explicit synthetic: [128, 32]`  
  `meta: ValueError meta.json L=128 does not match config L=32`

### T2.3 / T2.4 smoke run (CPU)
- `.\.venv\Scripts\python.exe -m jncm.train_tokenizer --config configs/smoke.yaml --steps 50 --allow_synthetic --run_id tok_vq_t24`
  `Starting training run tok_vq_t24 (vq) for 50 steps on cpu...`  
  `Step     1 | Loss: 4.3445 | acc_full: 0.0176 | tok_pos: 512`  
  `Step    50 | Loss: 4.0646 | acc_full: 0.0703 | tok_pos: 25600`  
  `Training completed. Results written to results\tok_vq_t24`
- `results/tok_vq_t24/git_sha.txt` = `373f9141c7b5a2b067c5598ca832c6082618ddd6` (clean tree).
- Last `metrics.jsonl` record: `data_source: "synthetic"`, `acc_cum: [0.015625, 0.017578125, 0.02734375, 0.0703125]`, `acc_full: 0.0703125`, `tok_positions: 25600`, `gen_positions: 0`, `total_positions: 25600`, `nominal_bits: 6.0`, codebook usage/perplexity present.
- `python -m pip install -e .` finished with `Successfully installed jncm-0.1.0`.
- `python scripts/eval_data_row.py --help` starts with:
  `usage: eval_data_row.py [-h] [--data DATA] [--tokenizer TOKENIZER] ...`
  `Evaluate Gen PPL and H on real validation data`

## DECISIONS
1. **VQ codes stay straight-through vectors.** `Encoded.codes` are float vectors with the straight-through estimator; `Encoded.targets` are the integer indices. The generator conditions from `targets`, never `codes` (SPEC 4.1; F7 test `reconstruct(enc.targets)` passes).
2. **Keep the learned QKV/output projections and delete old smoke checkpoints.** All pre-existing `.pt` files under `results/` were deleted; only checkpoints from the current architecture remain (`results/tok_vq_t24`). Old Gemini/pre-QKV checkpoints are not loadable.
3. **Level schedule uses the exact half-up integer formula** `l_k = 1 + (2*(k-1)*(L-1) + (K-1)) // (2*(K-1))` (SPEC 2 / v1.2, F10). No Python `round`.
4. **Q8 resolved: keep SPEC 5.1 and the deterministic `regress` head.** Level 1 has one position and zero conditioning, so a deterministic head can only emit one code. The old 8-sequence unconditional acceptance was an invalid test, not a model bug. T3.5 now accepts (a) one unconditional fixed sequence at >= 0.90 and (b) the true level-1 code supplied, sampling levels 2..K, all 8 sequences at >= 0.90. SPEC 5.1 and the regress head are unchanged.
5. **G0a tokenizer budget cut to 10,000 steps (allocation saving, 2026-11-07).** The dev tokenizers run as `tok_vq_10k` (job 2880034) and `tok_noisy_10k` (job 2880035) with `STEPS=10000` and `--time=04:00:00` instead of the 40,000-step SPEC 4.4 dev budget; the 24 h `jncm_tok` job 2800220 was cancelled. SPEC 4.4 and the G0a thresholds are unchanged (`acc_full` >= 0.999 vq / 0.995 noisy on `data_source=real`, checked by `scripts/g0a_report.py` on the final record); these two runs are the budget-limited pair the gate is judged on.
6. **Directions after the first FIR runs (2026-11-20; `docs/PLAN_2026-11-20.md`; SPEC v1.6 section 10).** (a) TASKS T4.7, level-pair packing in the backbone, before the full runs. (b) The g1 and g2 dev pilots for correctness on a GPU (submitted as TASKS E1). (c) **Size: dev for every long run, decided by the researcher** ("use dev if we're running long jobs"); this replaces the agent's dev-versus-lite rule, so no lite pilot is run. (d) The corruption sweep (E3) before the full runs, with the selection rule of SPEC 10.3; then G0 + G1 (E4) and G2 base (E5); ablations in the SPEC 10.6 order (E6). (e) **SPEC v1.6 was written by the agent at the researcher's explicit request** ("define the SPEC and tasks moving forward"); it carries the Q14 text, the packed-backbone layout, the scorer rule, the cluster runtime rules and the new section 10. The researcher reviews it; the usual rule (agents do not edit SPEC) applies again from here.
7. **E3: `p_first` stays 0.8 (SPEC 10.3, 2026-11-21).** At step 2000, 3 seeds x 64 samples: p080 `99.21 +/- 3.06`, p050 `98.36 +/- 1.81`, p030 `115.42 +/- 1.60`; H 4.01 to 4.03, `pairwise_agree` about 0.015. All qualify; p050 is lowest but its margin over the default (0.85) is below twice the larger seed std (6.12), so the default stays. The dev configs already hold 0.8; no config change. The three arms trained on `edd6a99`; their evals ran on `edd6a99` (p080, p050) and `5385690` (p030; the sampler is unchanged).

## Files touched in T5.5b
- Code: `jncm/utils.py` (`parameter_counts`), `jncm/train_generator.py` (10% dev milestones, optimizer-free milestone state, first-record parameter counts, `frozen_codes` with `no_grad` + `autocast(enabled=False)`), `jncm/train_jepa.py` (optimizer-free milestone state, first-record parameter counts), `scripts/run_eval.py` (parameter counts in eval metadata and `META_KEYS`), `scripts/gate_report.py` (`n_params` column, denser curve interpolation).
- Configs: `configs/tinystories_dev.yaml`, `configs/jepa_dev.yaml` (dense `[0.1, ..., 1.0]` milestones); `configs/tinystories_lite.yaml`, `configs/jepa_lite.yaml` (new: H 512, 8 layers, 8 heads, batch 256).
- Slurm/scripts: `scripts/slurm/{tok,gen,jepa,eval}.sbatch` (optional `CONFIG`), `scripts/slurm/gen.sbatch` (`SETS`), `scripts/launch_phase.sh` (`--config`), `scripts/budget.py` (`--config`).
- Tests: `tests/test_config.py`, `tests/test_jepa_configs.py`, `tests/test_gate_report.py`, `tests/test_run_eval.py`, `tests/test_train_generator.py`, `tests/test_train_jepa.py`, `tests/test_sbatch_config.py` (new), `tests/test_launch_phase.py`, `tests/test_budget.py`.
- Docs: `docs/TASKS.md`, `docs/STATE.md` (SPEC.md not edited).

## Files touched in T5.5a
- Code: `jncm/generator/backbone.py` (`pooled_reconstructions`; `embed_levels` freezes the pooled part only when the tokenizer is frozen and always applies `input_embed` outside), `jncm/train_generator.py` (`compute_generator_loss` now calls `embed_levels` outside the encode/corruption `no_grad`; `validate_grad_accum`; micro-batch accumulation loop), `jncm/train_jepa.py` (accumulation loop, averaged `JointStep` diagnostics, `grad_accum` validation), `jncm/config.py` (`GeneratorTrainConfig.grad_accum = 1`), `jncm/tokenizer/trunk.py` (device-aware `sample_cutoff`).
- Configs: `configs/smoke.yaml`, `configs/tinystories_dev.yaml`, `configs/jepa_smoke.yaml`, `configs/jepa_dev.yaml` (`grad_accum: 1`; the JEPA files state that VICReg statistics are per micro-batch).
- Tests: `tests/test_train_generator.py`, `tests/test_jepa_joint.py`, `tests/test_train_jepa.py`, `tests/test_tokenizer.py`, `tests/test_jepa_configs.py`.
- Docs: `docs/TASKS.md`, `docs/STATE.md` (SPEC.md not edited).

## Files touched in T4.5
- Code: `jncm/utils.py` (`DONE_FILE`, `write_done`), `jncm/train_tokenizer.py`, `jncm/train_generator.py`, `jncm/train_jepa.py` (`sec_per_step`, DONE after the loop).
- Slurm/scripts: `scripts/slurm/{tok,gen,jepa}.sbatch` (DONE early exit; `gen.sbatch` `STEPS`/`GEN_PPL`), `scripts/slurm/chain.sh` (new), `scripts/launch_phase.sh` (new), `scripts/budget.py` (new).
- Tests: `tests/test_chain_sh.py`, `tests/test_launch_phase.py`, `tests/test_budget.py`, `tests/test_done_and_sec_per_step.py` (new, 38 tests).
- Docs: `RUNBOOK.md` (§0 Pilots first), `docs/TASKS.md` (T4.5), `docs/STATE.md` (SPEC.md not edited).

## Files touched in T4.4
- Scripts: `scripts/gate_report.py` (new: G0/G1/G2 table + verdicts, interpolation, both accountings), `scripts/slurm/eval.sbatch` (`TOKENIZER` optional, `--tokenizer` only when set).
- Tests: `tests/test_gate_report.py` (new, 6 fixture tests), `tests/test_eval_sbatch.py` (fixture-tree runs: joint run without `TOKENIZER`, `--tokenizer` when set, missing tokenizer fails loudly).
- Docs: `RUNBOOK.md` (§3A.3 `TOKENIZER` optional, new §3B gate report), `docs/TASKS.md` (T4.4), `docs/STATE.md` (SPEC.md not edited).

## Files touched in T5.4
- Configs: `configs/jepa_dev.yaml`, `configs/jepa_smoke.yaml` (new).
- Code: `jncm/train_jepa.py` (milestone checkpoints via `milestone_steps`), `scripts/run_eval.py` (`is_joint_checkpoint`, `tokenizer_from_joint_checkpoint`, joint metadata, optional `--tokenizer`), `scripts/plot_curves.py` (`--x`, `x_position`).
- Slurm: `scripts/slurm/jepa.sbatch` (new).
- Docs: `RUNBOOK.md` (§3A Phase 2: base, ablations, eval, both plots, G2 under both accountings, paste-back), `docs/STATE.md`, `docs/TASKS.md` (SPEC.md not edited).
- Tests: `tests/test_jepa_configs.py`, `tests/test_jepa_sbatch.py` (new), `tests/test_plot_curves.py`, `tests/test_run_eval.py`, `tests/test_train_jepa.py`.

## Files touched in T5.3b
- Code: `jncm/config.py` (`JEPAConfig.target_chain`), `jncm/jepa/joint.py` (`TARGET_CHAINS`, `encode_online_targets`, target-chain branch in `joint_loss`), `jncm/jepa/__init__.py` (exports), `jncm/train_jepa.py` (`jepa.target_chain` validation, `joint_positions`/`total_positions`/`total_positions_conservative` records and checkpoints, `target_chain` record field).
- Configs: `configs/smoke.yaml`, `configs/tinystories_dev.yaml` (`jepa.target_chain: "online"`).
- Tests: `tests/test_jepa_joint.py` (online-target property, chain-difference and detachment tests; the old target test now pins `ema_clean`), `tests/test_train_jepa.py` (3-step accounting test, 50-step flow run for both chains, updated 2x2/resume assertions, unknown-chain validation).
- Docs: `docs/STATE.md`, `docs/TASKS.md` (T5.3b), `docs/QUESTIONS.md` (Q10-Q13) (SPEC.md not edited).

## Files touched in T5.3
- Code: `jncm/jepa/joint.py` (new: `encode_clean`, `integrate_flow`, `predicted_codes`, `JointStep`, `joint_loss`, `JEPA_HEADS`), `jncm/jepa/diagnostics.py` (new: `std_summary`, `effective_rank`, `explained_variance`, `code_diagnostics`, `CollapseAlarm`, `COLLAPSE_STD`, `COLLAPSE_PATIENCE`), `jncm/train_jepa.py` (new: `validate_jepa_config`, `joint_named_parameters`, `load_online_tokenizer`, `run_diagnostics`, `train`, CLI), `jncm/jepa/__init__.py` (exports).
- Tests: `tests/test_jepa_joint.py`, `tests/test_jepa_diagnostics.py`, `tests/test_train_jepa.py` (new, 23 tests).
- Docs: `docs/STATE.md`, `docs/TASKS.md` (SPEC.md not edited).

## Files touched in T5.1 / T5.2 / g0a_report
- Code: `jncm/jepa/__init__.py`, `jncm/jepa/ema.py` (new: `cosine_momentum`, `EMAModule`), `jncm/jepa/vicreg.py` (new: `variance_term`, `covariance_term`, `vicreg_loss`), `scripts/g0a_report.py` (`--runs`, variant-keyed thresholds).
- Tests: `tests/test_g0a_report.py`, `tests/test_jepa_ema.py`, `tests/test_jepa_vicreg.py` (new, 16 tests).
- Docs: `docs/STATE.md`, `docs/TASKS.md` (SPEC.md not edited).

## Files touched in T0.2 / T1.2 (verification only)
- Docs: `docs/STATE.md`, `docs/TASKS.md`. No code changes; `data/tiny_test` was generated by the existing prep script, and `results/t1_2_protocol_gate.json` is the gate record (both git-ignored).

## Files touched in T3.5
- Code: `jncm/train_generator.py` (new: `load_tokenizer`, `build_generator`, `build_optimizer`, `compute_generator_loss`, `diversity_metrics`, `sample_and_score`, `run_diagnostics`, `train`, `build_decode_fn`, CLI), `jncm/config.py` (`GeneratorTrainConfig` with the SPEC 5.7 defaults, exposed as `generator.train`).
- Configs: `configs/smoke.yaml`, `configs/tinystories_dev.yaml` (`generator.train` block; smoke uses batch 16 / 200 steps, dev uses the SPEC 5.7 values).
- Tests: `tests/test_train_generator.py` (new: 13 fast tests + 4 slow acceptance arms; SPEC 6.1 diversity endpoints).
- Docs: `docs/STATE.md`, `docs/TASKS.md`, `docs/QUESTIONS.md` (Q8 resolved) (SPEC.md not edited; the researcher replaced it with v1.3).

## Files touched in T3.4
- Code: `jncm/generator/sample.py` (new: `sample_tokens`, the SPEC 5.6 coarse-to-fine loop), `jncm/generator/__init__.py` (exports `sample_tokens`, `LevelHead`).
- Tests: `tests/test_generator_sample.py` (new, 14 tests).
- Docs: `docs/STATE.md`, `docs/TASKS.md`, `docs/QUESTIONS.md` (Q7) (SPEC.md not edited).

## Files touched in T3.3
- Code: `jncm/generator/corruption.py` (new: `linear_schedule`, `swap_codes`, `add_code_noise`, `corrupt_codes`, `MODES`), `jncm/generator/__init__.py` (exports).
- Tests: `tests/test_generator_corruption.py` (new, 9 tests).
- Docs: `docs/STATE.md`, `docs/TASKS.md` (SPEC.md not edited; no docs/QUESTIONS.md entry needed — the three ASSUMPTIONs below are non-blocking).

## Files touched in T3.2
- Code: `jncm/generator/heads.py` (new: `CEHead`, `RegressHead`, `FlowHead`/`FlowMLP`, `SinusoidalTimeEmbedding`, `build_head`, `level_weights`, `weighted_total`, `sample_codes`), `jncm/generator/__init__.py` (exports), `jncm/config.py` (`generator.regress_cosine`).
- Configs: `configs/smoke.yaml`, `configs/tinystories_dev.yaml` (`regress_cosine: false`).
- Tests: `tests/test_generator_heads.py` (new, 13 tests).
- Docs: `docs/STATE.md`, `docs/TASKS.md`, `docs/QUESTIONS.md` (Q6) (SPEC.md not edited).

## Files touched in T3.1
- Code: `jncm/generator/backbone.py` (was untracked scratch from an earlier session; reviewed, no behavioural change needed), `jncm/generator/__init__.py`.
- Tests: `tests/test_generator_backbone.py` (new, 9 tests).
- Docs: `docs/STATE.md`, `docs/TASKS.md` (SPEC.md not edited).

## Files touched in T4.2
- Code: `jncm/eval/genppl.py` (`score_batch`, batched `score_texts`, `tokenizer` property; `score_sample` unchanged as the reference), `jncm/config.py` (`GeneratorTrainConfig.save_milestones`/`milestones`, `config_from_dict`), `jncm/train_generator.py` (`milestone_steps`, milestone checkpoint saving), `configs/smoke.yaml`, `configs/tinystories_dev.yaml`.
- Scripts: `scripts/run_eval.py` (new), `scripts/plot_curves.py` (new), `scripts/slurm/eval.sbatch` (new).
- Tests: `tests/test_genppl_batched.py`, `tests/test_run_eval.py`, `tests/test_plot_curves.py`, `tests/test_eval_sbatch.py` (new).
- Docs: `docs/STATE.md`, `docs/TASKS.md` (SPEC.md not edited).

## Files touched in T4.3
- `RUNBOOK.md` (new), `docs/STATE.md`, `docs/TASKS.md`. No code changes; no wall clock or metric is guessed (all `<WALL_CLOCK>`, reference row from the T1.2 gate).

## Files touched in T4.1
- `scripts/slurm/prep.sbatch`, `scripts/slurm/tok.sbatch`, `scripts/slurm/gen.sbatch` (new), `docs/STATE.md`, `docs/TASKS.md`, `docs/QUESTIONS.md` (Q9). No `eval.sbatch` — T4.2 owns it.

## Files touched in T2.4
- Code: `jncm/config.py`, `jncm/levels.py`, `jncm/tokenizer/vq.py`, `jncm/train_tokenizer.py`, `scripts/prep_tinystories.py`.
- Packaging: `pyproject.toml` (new), `.gitignore` (zip archives).
- Tests: `tests/test_review_fixes.py` (F5/F8 added), `tests/test_train_tokenizer.py` (F8 `results_dir`, F9 dirty/optimizer tests), `tests/test_data_prep.py` (F9 byte alphabet).
- Docs: `docs/STATE.md`, `docs/TASKS.md`, `docs/QUESTIONS.md` (SPEC.md not edited).

## UNVERIFIED / not run
- Still unverified after canary 3: resume after a kill on a GPU (CUDA RNG state restore; no job was killed), bf16 beyond this single run, the JEPA trainer and the flow/regress heads on FIR, the lite config on FIR, the GPU speed-up of T4.7 (the 4.46 s itself is explained by FLOPs, see VERIFIED), and whether `module load python/3.12` mattered for CUDA (canary 1 had none and saw no GPU on fc10513; canaries 2 and 3 had it and ran on cuda). `import torch` took 93 s on a cold node (probe 2), which every job pays.
- `--cpus-per-task=6` / `--mem=32000M` in `gen`, `jepa` and `eval.sbatch` are copied from `tok.sbatch`, not measured: the canary's `MaxRSS` (`sacct -j <id> --format=JobID,MaxRSS,ReqMem`) replaces them. That a 6-CPU/32000M request starts as fast as the 1-CPU one is UNVERIFIED (the billing weight is identical).
- T5.7-T5.9a on a GPU: `torch.cuda.get_rng_state_all`/`set_rng_state_all` and `Generator.get_state`/`set_state` on CUDA (M2), `train_sec_per_step` on H100, and the about 1.5 GB `latest.pt` writes every 1,000 steps (I/O and disk on FIR). A pilot that is killed and resumed on a GPU node settles the first; the dev pilots' `train_sec_per_step` settles the second and shows whether B1's 2.1 s/step cliff matters.
- The tokenizer trainer still writes `latest.pt` only every `train.save_every` and has no data/RNG state (B1 and M2 were not applied there: both `tok_*_10k` are DONE). Any future tokenizer rerun, such as a from-scratch JEPA ablation, needs the same change first.
- T4.5 is syntax- and fixture-tested only: `chain.sh`/`launch_phase.sh` have never submitted a real FIR job, no pilot has been measured on an H100, and every chunk count / budget number above comes from fixtures. The launch kit also assumes `squeue` reports job ids from `sbatch` stdout in the standard `Submitted batch job <id>` form.
- T4.4 `scripts/gate_report.py` is fixture-tested only: it has never run on real eval aggregates (no cluster results yet), so no G0/G1/G2 verdict above is a real result.
- T0.2 and T1.2 are verified on this host (see VERIFIED); they are not yet repeated on FIR.
- T3.5: verified only at CPU smoke scale and on the 600–1200 step memorization runs above. Real generator runs (H=768, 12 layers, batch 1024, 10k steps, about 1.3B positions, SPEC 5.7) are untested; Gen PPL in T3.5 used a fake scorer (the real GPT-2 path is verified separately by T1.2). The `regress` memorization arm is structurally blocked (Q8).
- T3.1–T3.4 were verified on eval-mode modules; T3.5 is the first end-to-end (tokenizer -> generator -> decode) check, but only on 8 fixed sequences.
- G0a/G0/G1/G2: human cluster gates, not run.
- T5.3/T5.3b/T5.4 are CPU-verified at smoke scale only; no real JEPA joint run (dev-scale noisy tokenizer, H=768, batch 1024) has been made, no G2 comparison exists, and nothing was run on FIR. `jepa.sbatch` is syntax-checked only and its remote dataset/tokenizer paths are unproven.
- If any F2 test passed before its fix: `F2[noisy]` passed. The vq crash was real (`index_add_` dtype); the fp32 quantizer change was kept for both variants.
- T4.2 is verified on smoke-scale checkpoints and the throwaway CLI run above; a real 256-sample x 5-seed eval on a trained generator has not run yet (the real GPT-2 scoring path itself is covered by the T1.2 gate and the batched probe).

## Known issues / deviations
1. `GPT2Scorer.score_texts` scores in right-padded batches (default 8 texts); `score_sample` stays as the one-at-a-time reference. A real GPT-2 probe agrees to 4.4e-7 mean NLL (T4.2 VERIFIED).
2. Resume restores model/optimizer but not the data iterator or RNG; reproducibility across resumes assumes the same batch size and seed. Acceptable for dev, noted here.
3. Weight decay now excludes embeddings, LayerNorm parameters and biases (F9); this changes optimizer parameter groups, so pre-fix checkpoints cannot be resumed (model-only loading still works).
4. `train_tokenizer` logs `train_loss` as the last batch loss, not an interval mean. SPEC does not define this field.
5. `git_sha.txt` now ends in `-dirty` when the working tree has uncommitted changes.
6. T0.2 and T1.2 were ticked 2026-11-07 after the real-data runs above; the earlier "not run" entries are superseded.
7. `tok.sbatch` resumes from `results/<run_id>/ckpt/latest.pt`, which the trainer rewrites every `train.save_every` steps (5000 in the dev config), so a preempted job loses at most that many steps. Resume already does not restore the data iterator or RNG (item 2).
8. The slow acceptance takes about 2 minutes on CPU (mostly the flow arm), so `pytest -x -q` now takes about 2.5 minutes. `pytest -m "not slow"` is the fast loop.
9. FIR `~/jncm` is not a git repository (`git log` -> `fatal: not a git repository`); the tree was extracted from the tar, so cluster runs cannot write a real `git_sha.txt`. `RUNBOOK.md` §1.1 replaces it with a clone of the private remote; until then, compare cluster results against the local commit recorded here. The current remote `scripts/slurm/tok.sbatch` is also the pre-h100 copy (`--gres=gpu:1`), so pass `--gres=gpu:h100:1` on the command line.
10. RESOLVED 2026-11-07: `scripts/g0a_report.py` takes `--runs` (default `tok_vq_10k tok_noisy_10k`) and keys the thresholds on each run's quantizer variant (0.999 vq / 0.995 noisy), read from `results/<run_id>/config.yaml` (`tokenizer.quantizer`) with a run-id fallback; an unknown variant fails loudly. The 10k gate pair can now be judged directly.
11. With `jepa.vicreg` on and `head=flow`, the VICReg c-hat term integrates the flow (`flow_steps` network calls per level) inside the loss (Q11), so a JEPA flow step costs more than a Phase-1 flow step; `regress` has no such overhead. The EMA copy also holds the tokenizer decoder, which the target chain never uses (Q10).
12. RESOLVED 2026-11-08 (T4.4): `scripts/slurm/eval.sbatch` no longer requires `TOKENIZER`. It passes `--tokenizer` only when the variable is set, so a joint JEPA run (`run_eval.py` reads the tokenizer from the joint checkpoint) works without it; a staged run still needs `TOKENIZER=<path>` (RUNBOOK §3A.3).
13. FIR `~/jncm` (the tar tree) predates T4.5: `grep -c write_done jncm/utils.py` -> `0`, so runs there write no `results/<run_id>/DONE` and completion has to be read from `sacct` plus the last metrics record. The `~/jncm_git` clone has the T4.5 writer and the two finished tokenizer runs were backfilled through it on 2026-11-11. FIR still has no `models/` directory at all and `GPT2_LARGE_PATH` is unset, so `RUNBOOK.md` §1.2 must be repeated before any generator or eval job. Two launch-kit bugs seen only on a Linux checkout: `scripts/slurm/chain.sh` is committed `100644` but is exec'd by `launch_phase.sh:244` (needs `git update-index --chmod=+x`), and `.gitignore`'s `data/`/`models/`/`results/`/`.venv/` patterns do not match symlinks, so a docked checkout is never `git status`-clean without `.git/info/exclude`.

## ASSUMPTIONs (spec ambiguities resolved; details in docs/QUESTIONS.md)
- Synthetic data is allowed only with `allow_synthetic=True` / `--allow_synthetic`; real runs fail loudly if `data/tinystories/{train,val}.bin` are missing.
- VQ `codes` remain straight-through vectors so decoder gradients reach the encoder; `targets` remain integer indices.
- Learned Q/K/V and output projections are part of the SPEC's pre-LN transformer block.
- SPEC v1.2 resolves the schedule tie-breaking ambiguity: exact half-up integer arithmetic.
- SPEC 5.1 says "64 log-spaced frequencies" without a range: SinPos uses 64 geometric frequencies from 1 to 1e-4 cycles per normalized position, applies 2*pi*p_i*freq, concatenates sin and cos (128 features), and projects linearly to H.
- SPEC 5.2's "dropout 0.1" is applied as residual dropout (after the attention output projection and after the MLP), as in the tokenizer blocks; attention-internal dropout is 0.
- SPEC 5.3 "3 residual blocks (Linear 512 -> 512, SiLU)" is implemented as `x + SiLU(Linear(x))`; the time embedding is 64 log-spaced frequencies from 1 to 1e-4 cycles over `2*pi*t` for t in [0, 1], sin and cos concatenated to 128 dims (same convention as the backbone positions).
- SPEC 5.3's regress "option `cosine`" is the config flag `generator.regress_cosine` (default false = MSE). `FlowHead` widths follow SPEC (hidden 512, 3 blocks, 128 time dims) and are constructor arguments only so tests can shrink them.
- Head `loss` returns per-level means plus their sum; SPEC 5.5 weighting is `weighted_total(losses, schedule, alpha)` = `sum_k w_k * l_k * mean_k`, exact for the SPEC formula with per-position losses.
- SPEC 5.4 corruption: `corrupt.mode="noise"` requires `quantizer="noisy"` in `corrupt_codes` (missing/other variant raises, so the vq baseline cannot silently get channel noise).
- SPEC 5.4 swap: a level with fewer than two (batch, position) slots (e.g. B=1, l_1=1) has no "other" position to draw from and is left unchanged. `linear_schedule(K=1)` returns `[first]`, since the single level is the first level.
- SPEC 5.6 does not say whether the sampler switches modules to `eval()`. `sample_tokens` leaves train/eval modes untouched (the caller decides) and runs under `torch.no_grad()`; all draws take one explicit `torch.Generator`, drawn per level in order, so a seed fixes the whole sample.
- Corruption is out-of-place and takes one explicit `torch.Generator` for all levels (draw order: per level, mask then replacement index), so a run is reproducible.
- Overfit tests use batch size 1: level 1 is built from `zhat_0 = 0`, so every row of a level-1 batch sees the identical input and a per-position head can emit only one level-1 prediction (same for the regression floor). A perfect fit is only meaningful with one sequence; this is a property of SPEC 5.1/5.3, not a test shortcut.

- T3.5 acceptance metric: `sample_tokens` draws independently per row, so with 8 fixed sequences each of the 8 samples is compared with its **best-matching** training sequence (token accuracy), and the test also requires >= 3 distinct matches so that a collapse onto one sequence cannot pass. This is the reading that makes the TASKS wording meaningful for a stochastic sampler; the 90/90/80 thresholds are unchanged (Q8 covers the regress blocker).
- SPEC 5.7's generator defaults live in `generator.train` (lr 3e-4 -> 3e-5, wd 0.1, warmup 500, batch 1024, 10k steps), separate from the tokenizer's SPEC 4.4 `train` block. `smoke.yaml` overrides them to batch 16 / 200 steps so CPU smoke runs stay short; tests override them further.
- `train_generator` reads `tok_positions` from the tokenizer checkpoint as the curve offset for SPEC 6.2; `gen_positions = step * batch * L` and `total_positions` is their sum. A resume keeps the checkpoint's `tok_positions` and recomputes `gen_positions` from the step (same batch size assumed, as in the tokenizer trainer).
- `train_generator` requires the head/variant pairing of SPEC 5.3 (`ce` with vq, `regress`/`flow` with noisy) and fails loudly otherwise; the config does not guess.
- `configs/jepa_smoke.yaml` uses `head: regress` rather than flow because a flow JEPA step integrates the flow inside VICReg (Q11) and is slower on CPU; the file states that both heads are allowed.
- On the `total_positions_conservative` plot a staged record keeps its own `total_positions` (it has no conservative count), so RUNBOOK §3A.4 compares only overlapping x values and excludes the JEPA 100% point (2.62B), which is past every Phase-1 run (1.97B).
- Q9: T4.1 asks every template for the three `HF_*_OFFLINE` variables, but `prep.sbatch` exists to download TinyStories. `tok.sbatch`/`gen.sbatch`/`eval.sbatch` set all three; `prep.sbatch` sets none and says it needs Hub access. Smallest reversible choice; the submit forms and placeholder list above are the contract.
- T4.2 eval file naming: `results/<run_id>/eval/step_<n>.json`, `<n>` = completed training step zero-padded to six digits (same convention as `ckpt/step_<n>.pt` and `samples/step_<n>.txt`). Per-seed shards are `step_<n>.seed_<i>.json`; `--merge` folds them into the aggregate.
- T4.2 milestone checkpoints: `ckpt/milestone_<pct>.pt` at `round(fraction * max_steps)` clamped to `1..max_steps`, for fractions `[0.10, 0.25, 0.50, 1.0]`, behind `generator.train.save_milestones` (default true). Two fractions rounding to one step share it (first keeps it); 100% coincides with the last periodic checkpoint.
- T4.2 eval rebuilds the generator from the checkpoint's saved `config`; `--config` must still agree on L, V, head, quantizer and K. This is what lets a smoke run trained with overridden H/n_layers be evaluated from `configs/smoke.yaml`.
- T4.2 plot input: TASKS says `results/**/metrics.jsonl`, the task deliverable says `results/**/eval/*.json`; the script reads aggregate eval JSON and skips per-seed shards. The entropy panel plots dataset-token H solid and GPT-2-token H dashed; the diversity panel plots `pairwise_agree` solid and `unique_frac` dashed; x is `total_positions`, falling back to `gen_positions + tok_positions` (SPEC 6.2 offset).
- T4.2 `eval.sbatch`: one array task per seed (`--seeds $SLURM_ARRAY_TASK_ID`) writing shards; the header documents the `--merge` step. `#SBATCH --array=0-4` is a placeholder kept in sync with `N_SEEDS=5`.
- JEPA's E-bar is a full deep copy of the tokenizer, decoder included (unused); see Q10. `encode_clean` repeats the SPEC 4.1 chain with `c_k = LayerNorm(Linear_k(u))`, so `ema_clean` targets never see the channel noise (SPEC 8 step 2).
- `jepa.target_chain` defaults to `online` (SPEC v1.4 section 8 step 2): `t_k = LayerNorm(Ebar.proj_k(down(Ebar.encode_trunk(x) - detach(zhat_{k-1}), l_k)))` with the online `zhat` from `tokenizer.encode` before corruption. `ema_clean` keeps the v1.3 `encode_clean` chain. Joint runs do not charge the tokenizer offset: `joint_positions = steps*B*L`, `total_positions = joint_positions`, `total_positions_conservative = 2*joint_positions` (SPEC 6.2).
- SPEC 8's VICReg "on c-hat_k" uses the deterministic prediction: the raw `regress` output (no renorm, no channel noise) or `flow`'s noiseless Euler integration (Q11 states the exact flow definition). Sampling-time renorm/noise would make the variance term trivially satisfied.
- The SPEC 8 collapse alarm watches the online `c_k` mean std per level (below 0.1 for 200 consecutive steps). Target and prediction stds are logged too, but the alarm counts c_k only (Q12).
- JEPA has no separate train block: it reuses `generator.train` (lr, warmup, steps, batch, clip) and `cfg.tokenizer.p_drop` for nested dropout. A joint step is counted once as `joint_positions = steps*B*L` and twice as `total_positions_conservative` (SPEC 6.2); the tokenizer budget survives only as `tok_offset` bookkeeping.
- `jepa.enabled` is not consulted by `jncm.train_jepa`; the entry point itself selects joint mode (Q13). The four SPEC 8 ablation flags (`jepa.ema`, `jepa.vicreg`, `jepa.decoder_through_E`, `generator.head`) plus `jepa.target_chain` are all consumed and tested.

## Human must fill / do
- **FIR state and next steps (FIR is back since 2026-11-20; the agent drove it through the user's `newFir` zellij pane, every write announced):**
  1. Done 2026-11-20: `git push` (`dfd95a9`, then `5c20748`).
  2. Done 2026-11-20 on FIR: `git checkout -- scripts/slurm/gen.sbatch scripts/slurm/eval.sbatch scripts/slurm/jepa.sbatch` (the diff was read first: only the three `--time=00:20:00` patches), then `git pull --ff-only` -> `~/jncm_git` HEAD `5c20748`; gen, jepa and eval each carry exactly one `#SBATCH --mem` line. The untracked `slurm-*.out` logs were left in place (no `rm`).
  3. Obsolete, no action: the `pilot_gen_vq_ce` collision check (no `results/pilot_gen_vq_ce` exists; the OOM jobs died before writing anything), the `~/jncm_git/models` symlink (the pulled `eval.sbatch` honours `GPT2_LARGE_PATH`, which the launch command sets inline), and the cancel and resubmit of the four G1 evals (all 16 jobs had already failed).
  4. Done on FIR so far (each announced): `git checkout --` of the three `--time` patches and a fast-forward pull to `884a0db`, `scancel 4740493` (canary 1's dangling eval), `scancel 4748974` (probe 2, which had already finished), canaries 1 to 3 and probes 1 and 2 (see "FIR jobs"). The failed run's `results/pilot_gen_vq_ce/` held only `config.yaml`, `git_sha.txt` and empty `ckpt/`, `samples/` before canary 3 (no `latest.pt`), so the run id was reused on purpose.
  5. Done: eval 4754470 (see "FIR jobs"). Not done: `cd ~/jncm_git && python scripts/run_eval.py --run_id pilot_gen_vq_ce --merge` (it imports torch, 93 s cold; a one-off on the login node is what the RUNBOOK prescribes, but run it only if the shards need merging, since the aggregates already exist) and `python scripts/gate_report.py` on the pilot.
  6. On FIR now: the E1 pilots (see "FIR jobs"). When they end: `sacct -j 4763744,4763745,4763746,4763747,4763748,4763749 --format=JobID,State,Elapsed,MaxRSS,ExitCode`, the last metrics record and `DONE` of each run, and `ls results/<run>/eval`; SPEC 10.2 decides pass or fail. E3 (the corruption sweep) is already submitted (jobs 4775911 to 4775916, see "FIR jobs"; it needs only the vq + ce path that canary 3 verified). For reference, its training wall (RUNBOOK section 0.4, `--wall=03:13:22` = ceil(1.3 x 4.462225 x 2000) s from the canary's `train_sec_per_step`) is next and may run on the current code. After T4.7: E2 (`mv results/pilot_gen_vq_ce results/pilot_gen_vq_ce_unpacked` first, never `rm`). Never leave `SETS`, `GEN_PPL`, `N_SAMPLES` or `N_SEEDS` exported (`launch_phase.sh` refuses).
- **Pilot I/O risk after the pull (REVIEW advisor note):** with milestones at 10%, a 300-step pilot writes `latest.pt` (about 1.5 GB at dev size, with Adam state) plus a milestone file every 30 steps inside its `00:20:00` limit. After each pilot, compare `sacct` Elapsed with the limit; a `TIMEOUT` voids that pilot under `RUNBOOK.md` section 0.1, and the pilot then needs a longer `--time` (or `generator.train.milestones` reduced through `SETS`).
- **F16 sweep route (settled 2026-11-20):** it is submitted with `scripts/slurm/chain.sh scripts/slurm/gen.sbatch 1 ... --export=ALL,...,SETS=generator.corrupt.p_first=0.8` (exact commands: `RUNBOOK.md` section 0.4), not through `launch_phase.sh`, so the m6 refusal does not touch it; the dry-run form parses (checked locally), a Slurm `--export` value containing `=` is UNVERIFIED until the first submission. The old example keys in the `gen.sbatch` header (`generator.corruption.p`, `tokenizer.vq.codebook_size`) did not exist and would have raised; fixed, with a test.
- **SPEC edit (researcher only):** Q14 in `docs/QUESTIONS.md` has the exact replacement sentences for SPEC 6.2 and 8 (joint runs warm-start from the staged tokenizer and are charged `tok_offset`). The code and the paper draft already follow it.
- Leftover from the OpenCode worktree: the directory `../jncm-wt-t57c` and `.git/worktrees/jncm-wt-t57c` could not be deleted (`Permission denied`; the `opencode serve --service` process holds it). Delete both after `opencode service restart`, or leave them; the branch is merged and removed.
- **F16 sweep (2026-11-10):** GPT-2-large was copied to FIR by the human (2026-11-11) and `GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large` works. The old timing job 3234940 died of OOM, so no `sec_per_step` exists yet; the three `--time` values follow `ceil(1.3 * sec_per_step * 2000)` once the canary reports `train_sec_per_step`. Micro-batch decided: dev config plus `generator.train.grad_accum=16` in `SETS` (effective batch stays 1024). Exact commands: `RUNBOOK.md` section 0.4.
- T0.2 and T1.2 are complete locally (2026-11-07); no action. `models/gpt2-large` and `data/tinystories` are present on this host.
- G0a/G0 sync status (2026-11-20): `~/jncm_git` at `5c20748` (= local HEAD), tracked tree clean (only untracked `slurm-*.out` logs), `data`, `results`, `.venv` present; `~/jncm_git/models` does not exist (GPT-2 is reached through `GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large`, `model.safetensors` 3,096,165,928 bytes).
- Resolved 2026-11-11: the model was rsynced by the human; the old blocker text is kept only for the record: `rsync -avhP --partial models/gpt2-large/ fir:~/jncm/models/gpt2-large/` and `export GPT2_LARGE_PATH=$HOME/jncm/models/gpt2-large`. `~/jncm_git/models` is still not symlinked, but nothing needs it while `GPT2_LARGE_PATH` is exported.
- Gate G0a is done for both tokenizers: 2880034 `tok_vq_10k` (`02:21:20`, `acc_full 0.99998`) and 2880035 `tok_noisy_10k` (`02:16:49`, `acc_full 0.99994`) both COMPLETED -> PASS; `DONE` backfilled 2026-11-11. Record the numbers with `RUNBOOK.md` §4.2 if a written verdict is still wanted.
- **T4.1 placeholders — every one of them (nothing is guessed in the scripts):**
  - Account: pass `--account="$SBATCH_ACCOUNT"` on the sbatch command line (or let your submit wrapper read `$SLURM_ACCOUNT`); no account appears in any of the three files.
  - Partition: pass `--partition="$SBATCH_PARTITION"` on the sbatch command line; no partition appears in any file.
  - Wall clock: resolved 2026-11-19. `prep`, `gen`, `eval` and `jepa.sbatch` carry a valid default `#SBATCH --time=01:00:00` (`tok.sbatch` keeps `12:00:00`); `chain.sh` always passes `--time=<wall>`, and a manual sbatch overrides on the command line. `tests/test_sbatch_time.py` fails on any invalid or placeholder header.
  - GPU type: `tok.sbatch` (`#SBATCH --gres=gpu:1`, line 5, TODO at line 8) and `gen.sbatch` (same, line 5, TODO at line 8) ask for one generic GPU. Add a type only if FIR requires one. `prep.sbatch` requests no GPU (CPU-only numpy/tokenizers) — that is not a missing placeholder.
  - Modules: no `module load` line exists (TODO at `prep.sbatch:7`, `tok.sbatch:10`, `gen.sbatch:10`). Add your site's.
  - Venv: all three source `.venv/bin/activate` at the repo root and exit 1 if it is absent (TODO at `prep.sbatch:9`, `tok.sbatch:12`, `gen.sbatch:12`); create it per "Environment" below or edit the path.
  - `prep.sbatch` must reach the Hugging Face Hub (internet-enabled partition or a login node): it deliberately does not set `HF_*_OFFLINE` (Q9). `tok.sbatch` and `gen.sbatch` set all three offline variables.
  - Submit forms: `RUNBOOK.md` §2–§3 carries the exact, current ones (account, `--gres=gpu:h100:1`, `--time=<WALL_CLOCK>`, `HEAD`/`RUN_ID`/`TOKENIZER` inside `--export=ALL,...`). `prep.sbatch` is only needed if the data are ever rebuilt; the local `data/tinystories` is already prepared.
  - **T4.2 `eval.sbatch` placeholders:** wall clock (default `01:00:00` since 2026-11-19), array range (`--array=0-4`, TODO, keep equal to `N_SEEDS`), GPU type (`--gres=gpu:1`, TODO), modules (TODO), venv (sources `.venv/bin/activate`, TODO). Exact submit form plus the `--merge` and `plot_curves.py` steps: `RUNBOOK.md` §2.2 (Phase 1: §3).

## Environment
- **Local repo location (2026-11-24): `D:\code\jncm`**, on an external USB SSD formatted exFAT, moved at the researcher's request from the old OneDrive folder. `robocopy /E /COPY:DAT /DCOPY:DAT /XJ` copied 39,568 files (5.685 GB, 0 failed); a second pass `robocopy /L /FFT` listed 0 differences; `git status` clean, `git fsck` exit 0. exFAT records no ownership, so git needs `git config --global --add safe.directory D:/code/jncm` (added). The copied venv was re-pointed with `.venv\Scripts\python.exe -m pip install --no-index --no-deps --no-build-isolation -e .` (`import jncm` -> `D:\code\jncm\jncm\__init__.py`) and its activate scripts rewritten; the console `.exe` launchers still embed the old path, so run tools as `.venv\Scripts\python.exe -m pytest` (or `-m pip`). Full suite from `D:\code\jncm`: `487 passed, 1 skipped, 5 deselected in 571.03s`. FIR results come over with `scripts/pull_results.sh` (RUNBOOK 1.3; WSL Ubuntu has rsync 3.4.1).
- Python / torch: Python 3.13.7, torch 2.15.0+cpu (CUDA unavailable on this host); numpy 2.5.4, pytest 9.2.0.
- FIR venv (2026-11-10): torch/numpy/pyyaml as before, plus `transformers 5.22.1+computecanada`, `tokenizers 0.23.2+computecanada`, `safetensors 0.8.0+computecanada` and `huggingface-hub 1.33.0+computecanada` installed with `pip install --no-index` from `/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/`. Needed for `--gen_ppl` and `run_eval.py`.
- Recreate on Windows (PowerShell), from repo root:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  .\.venv\Scripts\python.exe -m pip install -e .
  .\.venv\Scripts\pytest.exe -q
  ```
- Recreate on Linux/cluster:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  pip install -e .
  pytest -q
  ```
- Data path: `data/tinystories/`   Models path: `models/`

## Next 3 steps
1. All four main chains are submitted (G0 5086493 to 5086497; G1 flow 5088616 to 5088619 + 5088624; G1 regress 5088620 to 5088623 + 5088625; G2 5088655 to 5088659). Poll (guarded) for failures and the first chunk boundaries.
2. Keep a poller running at all times (a usage-limit pause left four failed chunks unseen for a day). When a chunk fails on a new GPU-less node: exclude the node on every pending jncm job and append one chunk per lost chunk (as on 11-24). GPU resume is VERIFIED bitwise (11-24).
3. After the training chains: the 256 x 5 evals (in the chains), the 1024 x 5 final evals (`CKPT`, T4.8), merges (T4.9 or locally), `gate_report.py`; the first chunk boundary checked by hand (resume on the GPU).

## Results log (human-run gates; paste numbers from results/ only)
- T1.2 protocol gate (local, 2026-11-07): Gen PPL 10.80 +/- 0.26; H 4.03 +/- 0.01 (GPT-2 tokens) and 4.03 +/- 0.01 (dataset tokens); 256 samples x 5 seeds; verdict PASS; `results/t1_2_protocol_gate.json`.
- G0a: vq `tok_vq_10k` (2880034) COMPLETED 2026-11-09 — final step 10000, `acc_full 0.9999847`, `acc_cum` 0.078 -> 1.000, PASS vs 0.999. Noisy `tok_noisy_10k` (2880035) COMPLETED 2026-11-09 (`02:16:49`) — final step 10000, `acc_full 0.9999390`, 21 records, PASS vs 0.999. Both `DONE` files backfilled 2026-11-11.
- G0: not run
- G1: not run
- G2: not run
