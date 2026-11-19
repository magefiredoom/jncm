---
description: Read-only status of the FIR jobs and results (needs the FIR channel)
---
Read-only. Follow CLAUDE.md including the FIR rules, and use the channel recorded in docs/STATE.md "FIR operation notes". Print at most 30 lines per output.
1. Prove the channel (hostname plus an echoed token). If it fails, stop and tell me.
2. `squeue -u $USER -o "%.10i %.22j %.8T %.10M %.12l %.10r"`, then `sacct -S today --format=JobID,JobName%22,State,Elapsed,Timelimit,ExitCode`.
3. For each run listed in docs/STATE.md "FIR jobs": does results/<run_id>/DONE exist, and what is the last metrics record (step, acc_full or Gen PPL, sec_per_step)?
4. Report a table (run id, job id, state, elapsed, key metric) and anything that looks wrong. Do not submit, cancel or edit anything.
$ARGUMENTS
