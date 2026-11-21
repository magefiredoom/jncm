# Direction from 2026-11-20, night (after the E1 pilots)

Written by the agent at the researcher's request ("draft a plan for the future direction"). It follows `docs/PLAN_2026-11-20.md` and SPEC v1.6 section 10, which fix *what* runs; this plan says in what order, what the pilots changed, what it costs, where the decision points are and what the researcher decides. Facts carry their source (`docs/STATE.md` VERIFIED blocks); anything labelled *estimate* is arithmetic on them and is replaced by measurements as they come in. `docs/SPEC.md` is unchanged.

## 1. What tonight's runs changed

- **All three E1 pilots pass SPEC 10.2** (noisy + flow, noisy + regress, JEPA base; `COMPLETED` inside the 40-minute wall, `DONE`, finite Gen PPL at every milestone). Every arm of the main study has now run end to end on FIR. Each pilot used 32 samples and 1 seed, so the numbers below are signals, not results.
- **noisy + flow learns slower early:** Gen PPL 2951.8 at step 300 against 770 for vq + ce at step 300 (canary 3), falling steadily, H 4.53.
- **noisy + regress collapses as expected for a deterministic head:** Gen PPL alternates between about 36 and 390, H 2.06 to 3.17 (data row 4.03), `pairwise_agree` up to 0.42. A low Gen PPL here means repetitive text, which is why SPEC reports H and `pairwise_agree` next to Gen PPL.
- **JEPA is 1.7x dearer per step than the earlier plan assumed** (6.83 s against 3.99 s for flow, both unpacked; the old plan assumed 1.1x). This changes the ablation budget (section 4).
- **The E3 sweep runs on the unpacked code** (`corr_p080`, `corr_p050` running since 21:46 and 21:50, step 1500 at 23:45, about 4.05 s/step, their `p_first` confirmed in the configs; `corr_p030` pending on `ReqNodeNotAvail`). So the E2 packed re-pilot is needed after all, and T4.7 is pulled once `corr_p030` has started.
- **A merge needs Python, which the login node forbids.** Every eval array (one seed per task) needs `run_eval.py --merge` before the gate report; today that means an rsync to the local machine (MFA) or a compute job.

## 2. Directions chosen, and why

1. **Run the main comparison as one block.** Submit G0, G1 flow, G1 regress and G2 base together, as soon as E2 passes and E3 has fixed `p_first`, instead of G2 after G0/G1. SPEC 10.4 allows it ("E5 can start with E4"). Queue time sets the calendar (the E1 pilots waited over two hours), and the four chains then wait in parallel instead of in series.
2. **Measure, then decide on more speed work.** E2 measures the packed dev step. Further work (micro-batch, `torch.compile`) is opened only if the packed vq + ce step is above 3.0 s, and then only after one profiled step on the GPU. The walls come from the measured step (`ceil(1.3 x train_sec_per_step x steps)`, RUNBOOK 0.4), never from an estimate.
3. **New small task T4.9: a merge job.** Add `scripts/slurm/merge.sbatch` (CPU only, 1 CPU, 2 GB, 15 minutes; it runs `run_eval.py --run_id $RUN_ID --merge`) and have `chain.sh` chain it `afterok` on each eval array. That removes the rsync (an MFA step) from each of the roughly 15 evals ahead. Static tests like `tests/test_eval_sbatch.py`, written first. It touches only `scripts/slurm/`, so no plan mode is needed. The sweep evals finish tonight, so T4.9 is the first use; otherwise, rsync `results/corr_p0*` to local.
4. **Keep regress as SPEC says, and use it.** The full run is the deterministic baseline: it shows why a generative head is needed. The pilot already shows its failure pattern, so it makes a cheap, clear panel next to Table 1. If the allocation runs short, it is the first arm to cut, which is a SPEC change for the researcher.
5. **Write down the flow fallback now, decide it later.** If G1 (flow) fails its gate, the researcher chooses between reporting it and the HANDOFF section 10 option: a larger `d_k` or a smaller `sigma`. That option needs E8 (tokenizer resume parity) first, a new tokenizer (about 2.3 H100-h) and a new 10,000-step flow run. The verdict comes from the full run's 256 x 5 evals, not from the pilot's 32 samples.
6. **Ablations follow SPEC 10.6 order, gated by the allocation.** The first three (the 2x2 of section 8) come first. Items 4 to 6, the 10.7 training seed and the from-scratch JEPA (Q14) run only while budget remains. The repo does not record the remaining allocation (section 6).
7. **Paper alongside the runs.** Table 1 and Figure A (curves under both accountings, `plot_curves.py`) are filled the day each gate verdict exists; Figure B (per-level diagnostics) after G2. The regress collapse becomes the short paragraph that justifies reporting H with Gen PPL.

## 3. Ordered steps

| # | Step | Starts when | Requested H100-h (*estimate*) | Done when |
| --- | --- | --- | ---: | --- |
| 1 | Pull `c979361` (T4.7) on FIR | `corr_p030` has started (poller) | 0 | `git rev-parse` shows the new head; recorded in STATE |
| 2 | T4.9 merge job (tests first) | now, local | 0 | tests red then green; full suite passes |
| 3 | Merge the sweep evals (T4.9 job, or rsync) and apply SPEC 10.3; write `p_first` into the dev configs | the three eval arrays end | about 0.1 | one DECISIONS line in STATE; QUESTIONS entry if no setting qualifies |
| 4 | E2: packed vq + ce re-pilot (`mv results/pilot_gen_vq_ce results/pilot_gen_vq_ce_unpacked`, then `launch_phase.sh g0 pilot --wall=00:30:00`) | after step 1 (can run while the sweep finishes) | about 0.6 | `train_sec_per_step` measured; Gen PPL trajectory in line with canary 3 |
| 5 | E4 + E5 together: four 10,000-step chains and their evals | after steps 3 and 4 | 40 to 50 | four `DONE`s, merged evals, `gate_report.py` verdicts for G0, G1, G2 |
| 6 | First chunk boundary checked by hand (CUDA RNG and data stream on resume, still UNVERIFIED on GPU) | first chunk of step 5 ends | 0 | loss right after resume continues the record before it |
| 7 | E6 ablations 1 to 3 | after the G2 verdict | 45 to 50 | each evaluated like E5 |
| 8 | E6 ablations 4 to 6, then SPEC 10.7 seed 43 (G0, then the G1 arm nearest G0) | budget remains | 45 to 50, and 15 | as above; the list of arms that did not run is written down |
| 9 | E7 figures and tables; P1 results text; P2 references and novelty search; P3 template | as verdicts land; novelty search in the last January week | 0 | no `[TBD]` left in `paper/draft.md` |

## 4. Budget (*estimate*, H100-hours used; requested hours run 10 to 20% higher)

Packed step *estimates* until E2 measures them: vq + ce, flow and regress 2.2 to 2.6 s (the unpacked 3.96 to 4.07 s scaled by the 0.514 FLOP ratio on the 95% backbone share, plus overhead). JEPA 5.0 to 5.5 s (the unpacked 6.83 s minus the same backbone saving; its tokenizer and target passes do not shrink). Evals are about 1 to 2 H100-h per run (from the pilot evals, about 1:51 for 10 milestones x 32 samples including startup, scaled to 256 x 5 per milestone plus 1024 x 5 at the end).

| Item | H100-h |
| --- | ---: |
| G0 + G1 (3 x 10,000 steps) | 18 to 22 |
| G2 base (10,000 steps) | 14 to 15 |
| Evals for the four runs | 4 to 8 |
| E2 and the sweep merges | under 1 |
| **Main study** | **37 to 46** |
| Each JEPA ablation with its eval | 15 to 17 |
| Seed 43 (two runs) | 13 to 16 |

**Calendar.** With the four chains in parallel and waits like tonight's, the main study could finish within about a week of submission; fair-share decides. The binding limit is the allocation, not the calendar.

| When | What |
| --- | --- |
| Nov 21–22 | Steps 1 to 4 |
| Nov 22 | Submit step 5 |
| By about Dec 1 | Gate verdicts |
| December | Ablations |
| January | Paper |
| Last January week | Novelty search |
| Feb 1 | Submission |

## 5. Risks

- **Queue and fair-share.** Four parallel chains lower our priority further. If the queue stalls, G0 and G1 flow go first, then G2, then regress.
- **Code drift under pending jobs.** A job runs whatever `~/jncm_git` holds when it starts. The rule: pull only when no pending job belongs to a comparison already in flight; each run records its `git_sha.txt`.
- **First GPU resume.** CUDA RNG restore at a chunk boundary has not run on a GPU; step 6 checks it at the first boundary.
- **A misleading Gen PPL.** The regress pilot shows the failure mode. Every reported Gen PPL comes with H and `pairwise_agree`, and the G1 regress flag (> 0.5) is applied as SPEC 10.5 states.
- **SSH drops.** Every send to the pane is gated on the FIR prompt (memory note); a drop costs a reconnect, nothing else.

## 6. Needs the researcher

1. Review SPEC v1.6 (commit `d56c638`).
2. Direction 1: submit E4 + E5 together, about 37 to 46 H100-h (*estimate*), as soon as E2 passes and E3 has decided. This falls under the standing "submit once the probes pass" instruction, unless you want to approve that submission separately.
3. The remaining allocation (H100-h or RAP target). It decides how many ablations run (section 4).
4. Direction 3: T4.9 (the merge job), or an rsync of `results/corr_p0*` for the sweep: only the eval shards are needed (the merge reads `eval/step_*.seed_*.json`, not the checkpoints, which are about 6 GB per run). From the local repo root, for each of `corr_p080`, `corr_p050`, `corr_p030`: `rsync -avhP --partial aefad@fir.alliancecan.ca:~/jncm_git/results/corr_p080/eval/ results/corr_p080/eval/` (MFA), in the form RUNBOOK section 0 uses. The agent then merges locally and reports. The merged files stay local unless they are copied back.
5. Later: the flow fallback if G1 fails (direction 5), the ablation cut-off, the workshop (after the Nov 29 acceptances) and the title (after G2).

## 7. Storage (added 2026-11-21, after the sweep failures)

Both running sweep jobs died at 00:06 on a short `torch.save` write: the `/home` quota was full (`diskusage_report`: 48 GiB of 48 GiB; STATE, FIR jobs). `~/jncm_git/results` links to `/home/aefad/jncm/results`. A 10,000-step dev run writes about 9.4 GB, so the main study and the ablations need about 100 GB, and results so far are about 33 GB (*estimate*). Before step 5 of section 3:

1. The researcher, while no jncm job runs or is about to start: `scontrol hold` the pending jncm jobs, `mkdir -p /scratch/aefad/jncm`, `mv /home/aefad/jncm/results /scratch/aefad/jncm/results`, `ln -s /scratch/aefad/jncm/results /home/aefad/jncm/results`, `scontrol release`. `/scratch` has 19 TiB; no `/project` is listed. Scratch is purged after about 60 days (check Fir's policy), so the small records (`metrics.jsonl`, `eval/`, `config.yaml`, `git_sha.txt`, `DONE`) are copied to the local repo with each results rsync.
2. Optional, to shrink the footprint: `latest.pt` and the periodic `step_*.pt` files serve only for resuming (milestones carry no Adam state). After `DONE`, the periodic files of a finished run could go; only the researcher deletes files. Milestones stay: the evals need them.
3. `scripts/budget.py` gains a disk line (GB per run from the config) so a chain that will not fit is refused before submission (a candidate task, after the move).

## 8. Proposed SPEC v1.7 text (drafted 2026-11-21 for the researcher; not applied)

The researcher asked to "keep planning our next specs & tasks". SPEC stays untouched until the researcher approves these items; each is short and can be accepted alone.

1. **Section 7, storage (new bullet).** "Run outputs live on the cluster's scratch filesystem: `results` resolves there through a symlink, and a submission refuses a results root under `/home` (T4.10). After a run ends, its small records (`config.yaml`, `git_sha.txt`, `metrics.jsonl`, `DONE`, `eval/`, `samples/`) are copied to the local repository (T4.11); checkpoints that later runs depend on (the staged tokenizers) also have a second copy outside scratch (T4.12)."
2. **Section 6.1, merging (new sentence).** "Per-seed eval shards are merged by a CPU job chained `afterok` on the eval array (T4.9); no Python runs on a login node."
3. **Section 10.2 and 10.3, when a run counts as trained (clarification).** "A run counts as trained once its final milestone exists at the expected size; `DONE` is needed for chaining, resume bookkeeping and pilots. A sweep arm whose final resume save failed after that milestone is evaluated as normal, and the failure is recorded." (This covers `corr_p080` and `corr_p050`.)
4. **Section 10.3, a lost eval task (new sentence).** "An eval task cancelled by the system is rerun alone with the same seed (`--array=<i>`); the other seeds are kept."
5. **Section 10.4, submission order (new sentence).** "G0, G1 (flow and regress) and G2 base are submitted together once the packed re-pilot (10.2) has passed and 10.3 has fixed `p_first`; if the queue stalls, the order of priority is G0, G1 flow, G2, G1 regress."

## 9. A G0 risk seen in the sweep (added 2026-11-21)

At the end of 2,000 steps, vq + ce sits at Gen PPL about 96 to 103 with H 4.01 to 4.04 (`corr_p080`, `corr_p050`; 64 samples x 2 to 3 seeds). The G0 gate wants Gen PPL <= 20 at 10,000 steps. A log-log line through canary 3 (770 at step 300) and the sweep (98 at step 2000) reaches about 17 at step 10,000. That is an *estimate* only: the two runs have different learning-rate schedules and two points do not make a curve. G0 may therefore land close to its threshold. If it fails, SPEC 10.5 still lets G1 and G2 be judged against G0 at equal `total_positions`, and the paper reports the G0 number as measured. Changing the threshold or the step count is the researcher's call, and it should be made before the E4 results are seen, not after.
