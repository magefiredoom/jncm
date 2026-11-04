# PROMPTS — Gemini bootstrap, then OpenCode/DeepSeek

Design: the prompts are thin and the repo files are thick. The spec, task list and state log live in files, so switching models loses nothing. Each prompt forces the agent to read those files, work on one task, test it, and write down what it did.

## Setup (once)

1. Make a repo folder. Put `AGENTS.md` and `extract_files.py` in the root. Put `SPEC.md`, `TASKS.md`, `STATE.md` in `docs/`. Create an empty `docs/QUESTIONS.md`. Run `git init && git add -A && git commit -m "kit"`.
2. Build the paste blob for Gemini (one file you paste after prompt P1):

```bash
for f in AGENTS.md docs/SPEC.md docs/TASKS.md docs/STATE.md; do
  printf '=== FILE: %s ===\n' "$f"; cat "$f"; printf '\n'
done > /tmp/context_blob.txt
```

3. After each Gemini turn: copy the whole reply into `reply.md`, run `git commit -am wip` (so you can undo), then `python extract_files.py reply.md` to write the files. It overwrites existing files, including `docs/STATE.md`, which is intended. If it reports no file blocks, the copy lost the `### FILE:` lines or the code fences; copy the reply again.
4. Optional: OpenCode loads `AGENTS.md` from the project root automatically. If you also want `docs/STATE.md` loaded in every session, add it to the `instructions` list in `opencode.json` (check the OpenCode docs for the current key name).

## Flow

| Stage | Where | Prompts | Output |
|---|---|---|---|
| Check the spec against the paper (optional) | AI Studio, grounding on | P1a | discrepancy table; you edit SPEC.md by hand |
| Draft M0 to M2 | AI Studio | P1, then P2 | files + STATE.md |
| Check torch-free logic | AI Studio, code execution on | P2b | SANDBOX-CHECKED list |
| Handoff | OpenCode + DeepSeek | P3 | tests pass, deviations listed, STATE.md VERIFIED |
| M3 onward | OpenCode + DeepSeek | P4, one fresh session per task | one committed task per session |
| Audit after each milestone | the *other* model | P5 | `docs/REVIEW_<date>.md` |
| Something breaks | either | P6 | diagnosis first, no blind tuning |
| After a gate run on FIR | either | P7 | table, pass/fail, next plots |

Why hand off after M2: AI Studio's code-execution sandbox has no PyTorch (next section), so Gemini cannot run this repo's tests. The riskiest code (generator, heads, JEPA) should be written where tests can run. If you want more speed, run P2 once more for M3, accepting more debugging at takeover.

Gate thresholds in TASKS.md are my starting values from the NCM paper's TinyStories numbers. Change them in TASKS.md if your first results say they are miscalibrated, but do it yourself, not through an agent.

## Using AI Studio for the Gemini turns

What the Google documentation I could find says about its code-execution tool (some pages predate your model, so check the Tools panel):
- Python only. Each run is capped at 30 seconds and the model gets only a handful of runs per prompt. No internet, no installing packages.
- The library list (numpy, pandas, scikit-learn, matplotlib, sympy and similar) has **no PyTorch**, so it cannot run this repo's tests. It can check torch-free logic, which is what P2b does.
- Google notes that enabling it can degrade other kinds of output. Keep it off while Gemini writes files and use it in a separate check turn.
- Grounding with Google Search is best spent on the spec audit (P1a). I would keep grounding and code execution apart: grounding for the audit, code execution for the check turn, neither for file generation.

| Setting | P1a (spec audit) | P1, P2 (write files) | P2b (check turn) |
|---|---|---|---|
| System instructions | AGENTS.md | AGENTS.md | AGENTS.md |
| Thinking level | high | high | high |
| Output length | default | maximum allowed | default |
| Code execution | off | off | on |
| Grounding with Google Search | on | off | off |

Put `AGENTS.md` in the System instructions box so it stays in force for the whole chat. Paste the context blob into the first message (it repeats AGENTS.md, which is harmless). Leave temperature at its default unless the model's page says otherwise. Set the output length to the maximum, otherwise long file blocks get cut off mid-file; if a reply is cut off, answer "continue from the last complete file" and re-extract.

---

## P0 — Stop and hand off (use before switching models mid-task)

```text
Stop coding. Update docs/STATE.md now using its template: what you finished, what you changed since the last commit, what is half-done (with file names), the exact next command to run, ASSUMPTIONs, and anything you were unsure about. Then commit with message "WIP: <task id>".
```

## P1a — Spec-versus-paper audit (optional; grounding on, code execution off)

Run this first, in a fresh chat, with the context blob pasted after it. It catches places where my reading of the NCM paper was wrong. A grounded model can still misquote, so spot-check two or three rows against the paper yourself, then fix `docs/SPEC.md` by hand.

```text
Audit task. Use Google Search grounding or URL context to open arXiv 2610.14162 (Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation). Read the method section, the architecture and training appendices, and the ablation appendices.
Compare the paper with docs/SPEC.md sections 1, 2, 4.1, 4.2, 4.4, 5.1 to 5.7 and 6.1. Produce ONE table with columns: item | what the paper says (section or table number) | what SPEC says | match / mismatch / paper silent.
Rules: report only. Do not edit the spec, do not propose redesigns, do not write code. Items tagged [DECISION] are expected to differ; list them only if the paper contradicts them. If you cannot open the paper, say so instead of answering from memory.

[paste /tmp/context_blob.txt here]
```

## P1 — Gemini, turn 1 (paste the context blob after this)

`````text
ROLE
You are a senior ML engineer implementing a research codebase from a written spec. I am a student researcher; I run the long jobs myself on a Slurm cluster. We work over several turns. Everything you need is pasted below as four files, each introduced by "=== FILE: <path> ===": AGENTS.md, docs/SPEC.md, docs/TASKS.md, docs/STATE.md. Read all four before answering.

LIMITS OF THIS CHAT
- You cannot run this repo's tests here: any Python sandbox you have lacks PyTorch and internet. Never write that tests pass. Mark everything UNVERIFIED in docs/STATE.md.
- Write code that is easy to check on CPU: pure PyTorch, tiny smoke config, deterministic seeds. Keep logic that does not need torch (level schedule, window packing, entropy and perplexity math, bits formula) in small functions whose source can be copied into a plain-Python sandbox and checked there.
- If the spec is ambiguous: use the SPEC default if it gives one; otherwise do not guess silently. Record the question in docs/QUESTIONS.md and mark your choice ASSUMPTION in STATE.md.
- Do not change anything marked [DECISION] in the spec.

THIS TURN
Do only tasks T0.1, T0.2, T0.3 and T1.1 from docs/TASKS.md. Do not start M2.

REPLY FORMAT (follow exactly, in this order)
1. UNDERSTANDING: at most 12 bullets restating in your own words how the tokenizer and the generator fit together (what each level contains, what the generator conditions on at level k, what its targets are) and what differs between the vq and noisy variants. Then AMBIGUITIES: a list, or "none".
2. PLAN: the files you will write, one line each.
3. FILES: every file in full, each as
### FILE: relative/path
````language
<entire file content>
````
Use a four-backtick fence for every file block. No diffs, no "rest unchanged", no ellipses inside code, no placeholders.
4. SELF-REVIEW: for each code file, trace tensor shapes by hand for configs/smoke.yaml and list anything you are unsure about.
5. docs/STATE.md and docs/QUESTIONS.md as full updated files (same ### FILE format).
6. NEXT TURN: the tasks you would do next.

STYLE: type hints, shape docstrings, files under 400 lines, only the dependencies allowed in AGENTS.md rule 7.

[paste /tmp/context_blob.txt here]
`````

## P2 — Gemini, later turns (same chat)

```text
Continue in this same chat with the same rules and reply format. I extracted your files into the repo; nothing has been run, so everything is still UNVERIFIED.

THIS TURN: {T2.1, T2.2, T2.3}   <- replace with the tasks for this turn; at most one milestone per turn

Changes since your last turn: {none | files I edited by hand}

For this turn:
- Reuse the interfaces you already wrote (config, utils, data). If a task depends on a file you can no longer see, ask me to paste it instead of reconstructing it.
- UNDERSTANDING: only concepts that are new this turn, at most 6 bullets.
- For every test the task lists in docs/TASKS.md, write that test. Do not weaken thresholds.
- End with the full updated docs/STATE.md and docs/QUESTIONS.md.
```

If you start a new Gemini chat, paste the context blob again plus the files the next tasks depend on (for M2: `jncm/config.py`, `jncm/utils.py`, `jncm/data.py`).

## P2b — Sandbox check turn (same chat; switch code execution on first)

```text
Sandbox check turn. Code execution is on. It has no PyTorch and no internet, so check only torch-free logic. For each item, copy the function's source from your last reply into a code cell (remove or stub any torch imports), run it, and paste the real output:
1. Level schedule: L=128, K=12 must give [1, 13, 24, 36, 47, 59, 70, 82, 93, 105, 116, 128] (sum 774). L=32, K=4 must give [1, 11, 22, 32]. Both strictly increasing, first = 1, last = L.
2. Window packing: stories joined by an end token, cut into windows of exactly L tokens, remainder dropped, no overlap. Test on a small synthetic example and check the counts by hand.
3. Unigram entropy: uniform over n distinct tokens equals ln n; a constant sequence gives 0.
4. Perplexity from per-token NLLs: a constant NLL c gives exp(c).
5. Bits per position: noisy with d_k=8 and sigma=0.4 gives about 11.43 bits; vq with 4096 codes gives 12.
If a check fails, fix the function and output the corrected file block (### FILE format, four-backtick fences). Change nothing else. Finish with the full updated docs/STATE.md, listing these checks as SANDBOX-CHECKED with their outputs and everything else still UNVERIFIED.
```

The next session (P3) re-runs the same checks as real tests.

## P3 — OpenCode + DeepSeek, takeover session

```text
Read AGENTS.md, then docs/STATE.md, then docs/TASKS.md. The code in this repo was drafted by another model in a chat that could not run code. Treat all of it as UNVERIFIED.

Do these steps in order and report briefly after each one:
1. If this is not a git repo, run `git init`. Commit everything as "import Gemini draft (unverified)".
2. Set up the Python environment as described in STATE.md "Environment" (if it is empty, create a venv, install requirements.txt, and write the exact commands into STATE.md). Print the python and torch versions.
3. Run `pytest -x -q`. For each failure: find the root cause, fix the CODE, rerun. If a test contradicts docs/SPEC.md, do not edit the test: add the conflict to docs/QUESTIONS.md and move on. If something needs a download or network access, mark that test with a clear skip reason.
4. Conformance pass. For each completed task, compare the code with the SPEC section it implements and fix clear bugs. Check explicitly:
   a. the generator at level k sees only levels < k (zhat~_{k-1}), and level 1 sees zeros;
   b. nested dropout is used only in tokenizer training, never when producing generator targets;
   c. generator targets are clean codes (noisy variant) or indices (vq); the chain codes carry the channel noise;
   d. the level schedule and down/up operators match SPEC section 2 (l_1 = 1, l_K = L, identity at l = L);
   e. position accounting: tok_positions + gen_positions = total_positions in every metrics record;
   f. tokenizer and generator checkpoints round-trip (save, load, identical outputs).
   List every deviation you find in STATE.md, fixed or not.
5. Update docs/STATE.md (VERIFIED needs the command and the last lines of output). Commit.
Then stop. Do not start new tasks. Finish with a report of at most 15 lines: what passes, what you fixed, what is still open, questions for me.
```

## P4 — OpenCode + DeepSeek, one task per fresh session

Tip: run it first in Plan mode (Tab) to check the plan, then switch to Build.

```text
Task: {TASK_ID}. Follow AGENTS.md.
1. Read docs/STATE.md, the task in docs/TASKS.md, and the SPEC sections it refers to ({SPEC_SECTIONS}).
2. Before writing code, reply with a plan: files you will create or change, the tests you will write, any ambiguity. Then proceed, unless the plan exposes a spec conflict (then log it in docs/QUESTIONS.md and stop).
3. Write the tests listed in TASKS.md first or together with the code. Run them. Fix the code until they pass. Do not change thresholds.
4. Run the full suite `pytest -x -q` to check for regressions.
5. Update docs/STATE.md (VERIFIED items with command and last lines of output), tick the task in docs/TASKS.md, commit "{TASK_ID}: <summary>".
Stop after this task. Report in at most 10 lines.
```

Examples: `{TASK_ID}` = T3.1, `{SPEC_SECTIONS}` = 5.1, 5.2.

## P5 — Cross-model review (use the model that did NOT write the code)

```text
You are reviewing code written by a different model. Do not edit any code. Read AGENTS.md and docs/SPEC.md, then review `git diff {BASE_COMMIT}..HEAD`.
Write docs/REVIEW_{DATE}.md with findings sorted by severity (BLOCKER, MAJOR, MINOR). For each finding give: file and line, what is wrong, why it matters for the experiment, the fix, and a unit test that would have caught it.
Check specifically:
1. information leak: anything at level k that depends on level >= k;
2. off-by-one in the cumulative reconstruction (zhat_{k-1} versus zhat_k);
3. nested dropout or channel noise applied in the wrong place or mode;
4. EMA: update formula, parameter handling, target in eval mode with no grad;
5. losses: normalization by number of positions, level weights w_k, masking of padded positions;
6. detach / stop-gradient placement (decoder_through_E, JEPA targets);
7. evaluation protocol: re-tokenization, no truncation, entropy definition, seeds;
8. token accounting identical for vq and noisy runs;
9. dtype and device bugs (bf16 in LayerNorm or softmax, CPU/GPU mismatches), unseeded randomness;
10. anything that makes the vq-versus-noisy comparison unfair.
No style nits. If a category has no findings, say so in one line.
```

## P6 — Debug protocol (diagnose before tuning)

```text
Problem: {e.g. "noisy+flow Gen PPL is 3x worse than vq", "loss is NaN at step N", "per-level std collapsed"}.
Do NOT change hyperparameters or the spec yet. Diagnose first:
1. Reproduce with the smallest config that shows it. State the exact command and seed.
2. Collect and paste: per-level losses; per-level std and effective rank of the codes; acc_cum per level; for the generator, per-level R2 (1 - MSE/Var) on held-out data; gradient norms per module; the fraction of NaN/Inf; 8 decoded samples.
3. Give at most 3 hypotheses, ranked, each with the one experiment that would confirm or kill it and the outcome you expect.
4. Run the cheapest experiment for hypothesis 1 and report. Stop.
Write the findings to docs/STATE.md under "Known issues". Propose changes as a list for me to approve; do not apply them.
```

## P7 — Gate report (after you run a gate on FIR)

```text
I ran {GATE} on the cluster. Results are in results/{RUN_IDS}. Read the metrics.jsonl files. Use only numbers that appear in those files or that I gave you.
Produce: (1) a table of Gen PPL, entropy, acc_full and total_positions per run (mean and std over seeds); (2) pass or fail against the gate criteria in docs/TASKS.md, with the margin; (3) the three most informative plots to make with scripts/plot_curves.py; (4) anything surprising (non-monotonic curves, entropy collapse, per-level anomalies); (5) update the "Results log" in docs/STATE.md.
Do not propose or apply code changes.
```
