---
description: Recover after another agent was cut off mid-task
---
The previous agent was cut off mid-task. Follow CLAUDE.md. Do not redo or overwrite anything yet.
1. Run `git status --short`, `git diff --stat`, `git log --oneline -8`. Read docs/STATE.md.
2. List what looks half-done (files changed but not committed, tasks ticked without evidence, STATE.md claims you cannot confirm) and what is unrecorded. If FIR is in scope and I say so: prove the channel, then run `squeue -u $USER` and `sacct -S today --format=JobID,JobName%22,State,Elapsed,Timelimit` and reconcile with STATE.md "FIR jobs".
3. Verify only what you need with targeted tests. Do not run the full suite. Do not discard or revert anything without asking me.
4. Write the reconciliation into docs/STATE.md and commit it. Report in at most 10 lines, then wait for my instruction.
$ARGUMENTS
