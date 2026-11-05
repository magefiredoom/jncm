### Sandbox Checks (Torch-Free Logic)

#### 1. Level Schedule
Source from `jncm/levels.py`:
```python
def get_level_schedule(L: int, K: int) -> list[int]:
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}")
    if K == 1:
        return [L]

    schedule: list[int] = []
    for k in range(1, K + 1):
        l_k = int(round(1.0 + (k - 1) * (L - 1) / (K - 1)))
        schedule.append(l_k)

    if schedule[0] != 1:
        raise ValueError(f"Level schedule must start at 1, got {schedule[0]}")
    if schedule[-1] != L:
        raise ValueError(f"Level schedule must end at L={L}, got {schedule[-1]}")
    for i in range(len(schedule) - 1):
        if schedule[i] >= schedule[i + 1]:
            raise ValueError(
                f"Level schedule must be strictly increasing, got {schedule}"
            )

    return schedule
```
Execution and Output:
```text
128, 12: [1, 13, 24, 36, 47, 59, 70, 82, 93, 105, 116, 128] sum: 774
32, 4: [1, 11, 22, 32] sum: 66
128, 12 strictly increasing, first=1, last=L: True
32, 4 strictly increasing, first=1, last=L: True
```

---

#### 2. Window Packing
Source from `scripts/prep_tinystories.py`:
```python
import numpy as np

def pack_stories(
    story_token_lists: list[list[int]],
    eos_id: int,
    seq_len: int,
) -> np.ndarray:
    stream: list[int] = []
    for story in story_token_lists:
        if not story:
            continue
        stream.extend(story)
        if stream[-1] != eos_id:
            stream.append(eos_id)

    total_tokens = len(stream)
    num_windows = total_tokens // seq_len
    if num_windows == 0:
        return np.empty((0, seq_len), dtype=np.uint16)

    usable = num_windows * seq_len
    flat = np.array(stream[:usable], dtype=np.uint16)
    return flat.reshape(num_windows, seq_len)
```
Synthetic example:
- Story 1: `[10, 11]` $\to$ with `<eos>` (1): `[10, 11, 1]`
- Story 2: `[20, 21, 22]` $\to$ with `<eos>` (1): `[20, 21, 22, 1]`
- Story 3: `[30]` $\to$ with `<eos>` (1): `[30, 1]`
- Stream: `[10, 11, 1, 20, 21, 22, 1, 30, 1]` (9 tokens)
- With $L = 4$: $9 // 4 = 2$ windows, remainder `[1]` dropped.
- Window 0: `[10, 11, 1, 20]`
- Window 1: `[21, 22, 1, 30]`

Execution and Output:
```text
Windows shape: (2, 4)
Windows dtype: uint16
Windows content:
 [[10 11  1 20]
 [21 22  1 30]]
Exact match with hand calculation: True
```

---

#### 3. Unigram Entropy
Source from `jncm/eval/genppl.py`:
```python
import math

def unigram_entropy(token_id_lists: list[list[int]]) -> float:
    if not token_id_lists:
        return 0.0

    sample_entropies: list[float] = []
    for seq in token_id_lists:
        if not seq:
            sample_entropies.append(0.0)
            continue
        counts: dict[int, int] = {}
        for tok in seq:
            counts[tok] = counts.get(tok, 0) + 1
        n = float(len(seq))
        h = 0.0
        for cnt in counts.values():
            p = cnt / n
            h -= p * math.log(p)
        sample_entropies.append(h)

    return float(sum(sample_entropies) / len(sample_entropies))
```
Execution and Output:
```text
n=2: entropy=0.693147, ln(n)=0.693147, diff=0.0e+00
n=4: entropy=1.386294, ln(n)=1.386294, diff=0.0e+00
n=8: entropy=2.079442, ln(n)=2.079442, diff=0.0e+00
n=16: entropy=2.772589, ln(n)=2.772589, diff=0.0e+00
n=64: entropy=4.158883, ln(n)=4.158883, diff=5.3e-15
constant sequence: entropy=0.000000, expected=0.0
```

---

#### 4. Perplexity from Per-Token NLLs
Source from `jncm/eval/genppl.py`:
```python
from typing import Any
import math

def gen_ppl(texts: list[str], scorer: Any) -> float:
    if not texts:
        return float("nan")

    if hasattr(scorer, "score_texts"):
        tot_nll, tot_tokens = scorer.score_texts(texts)
    elif hasattr(scorer, "score_sample"):
        tot_nll = 0.0
        tot_tokens = 0
        for t in texts:
            nll, cnt = scorer.score_sample(t)
            tot_nll += nll
            tot_tokens += cnt
    elif callable(scorer):
        tot_nll, tot_tokens = scorer(texts)
    else:
        raise TypeError(f"Unsupported scorer: {type(scorer)}")

    if tot_tokens == 0:
        return float("nan")

    mean_nll = tot_nll / tot_tokens
    return float(math.exp(mean_nll))
```
Execution and Output:
```text
c=0.5: ppl=1.648721, exp(c)=1.648721, diff=0.0e+00
c=1.0: ppl=2.718282, exp(c)=2.718282, diff=0.0e+00
c=2.27: ppl=9.679401, exp(c)=9.679401, diff=0.0e+00
c=3.5: ppl=33.115452, exp(c)=33.115452, diff=0.0e+00
```

---

#### 5. Bits Per Position
Source from `jncm/utils.py`:
```python
import math
from typing import Any

def bits_per_position(
    cfg: Any = None,
    *,
    quantizer: str | None = None,
    d_k: int | None = None,
    sigma_k: float | None = None,
    V_k: int | None = None,
) -> float:
    if cfg is not None:
        tok_cfg = getattr(cfg, "tokenizer", cfg)
        if quantizer is None:
            quantizer = getattr(tok_cfg, "quantizer", None)
            if quantizer is None and isinstance(tok_cfg, dict):
                quantizer = tok_cfg.get("quantizer")
        if d_k is None:
            d_k = getattr(tok_cfg, "d_k", None)
            if d_k is None and isinstance(tok_cfg, dict):
                d_k = tok_cfg.get("d_k")
        if sigma_k is None:
            sigma_k = getattr(tok_cfg, "sigma_k", None)
            if sigma_k is None and isinstance(tok_cfg, dict):
                sigma_k = tok_cfg.get("sigma_k")
        if V_k is None:
            V_k = getattr(tok_cfg, "V_k", None)
            if V_k is None and isinstance(tok_cfg, dict):
                V_k = tok_cfg.get("V_k")

    if quantizer == "vq":
        if V_k is None or V_k <= 0:
            raise ValueError(f"V_k must be positive for vq quantizer, got {V_k}")
        return float(math.log2(V_k))

    if quantizer == "noisy":
        if d_k is None or d_k <= 0:
            raise ValueError(f"d_k must be positive for noisy quantizer, got {d_k}")
        if sigma_k is None or sigma_k <= 0:
            raise ValueError(f"sigma_k must be positive for noisy quantizer, got {sigma_k}")
        snr = 1.0 / (sigma_k**2)
        return float((d_k / 2.0) * math.log2(1.0 + snr))

    raise ValueError(f"Unknown quantizer: {quantizer} (expected 'vq' or 'noisy')")
```
Execution and Output:
```text
noisy (d_k=8, sigma=0.4): 11.4319 bits (spec reference: 11.43)
vq (V_k=4096): 12.0000 bits (spec reference: 12.0)
```

All 5 sandbox checks passed without errors or discrepancy.

---

### FILE: docs/STATE.md
````markdown
# STATE — jncm

Last updated: 2026-11-04 (Sandbox check turn: 5 torch-free functions verified in sandbox)
Rule: newest information wins. Rewrite sections; do not append history.

## Current position
- Milestone / task in progress: M2 complete (T2.1, T2.2, T2.3 implemented; Gate G0a is next human cluster gate)
- Last completed task: T2.3 (Tokenizer training loop, checkpointing, diagnostics, and tests)

## VERIFIED (command + last lines of output)
- `pytest -q`: 41 passed on CPU in turn 2 session.

## UNVERIFIED (written but not run; list SANDBOX-CHECKED items here with their output)
- SANDBOX-CHECKED:
  1. `get_level_schedule` (`jncm/levels.py`):
     - L=128, K=12 -> `[1, 13, 24, 36, 47, 59, 70, 82, 93, 105, 116, 128]` (sum: 774)
     - L=32, K=4 -> `[1, 11, 22, 32]` (sum: 66)
     - Output: strictly increasing, first = 1, last = L.
  2. `pack_stories` (`scripts/prep_tinystories.py`):
     - Output: windows shape `(2, 4)`, dtype `uint16`, contiguous windows `[[10, 11, 1, 20], [21, 22, 1, 30]]`, dropped remainder `[1]`, exact match with hand calculation.
  3. `unigram_entropy` (`jncm/eval/genppl.py`):
     - Output: uniform over n distinct tokens matches `ln(n)` to machine epsilon ($n=2, 4, 8, 16, 64$), constant sequence gives `0.0`.
  4. `gen_ppl` (`jncm/eval/genppl.py`):
     - Output: constant token NLL $c \in \{0.5, 1.0, 2.27, 3.5\}$ gives exactly $\exp(c)$ (diff `0.0e+00`).
  5. `bits_per_position` (`jncm/utils.py`):
     - Output: noisy ($d_k=8, \sigma_k=0.4$) gives `11.4319` bits (spec: 11.43); vq ($V_k=4096$) gives `12.0000` bits (spec: 12.0).
- UNVERIFIED:
  - T2.2 fixes in `jncm/tokenizer/vq.py` (added optional `threshold` keyword arg to `reinit_dead_codes`) and `jncm/tokenizer/noisy.py` (`eps=1e-8` for LayerNorm unit variance)
  - T2.3 `jncm/train_tokenizer.py` and `tests/test_train_tokenizer.py`

## Known issues / bugs
- none

## ASSUMPTIONs (spec ambiguities I resolved; each also in docs/QUESTIONS.md)
- none

## Human must fill / do
- Run `pytest -q` on CPU to confirm all 43 tests pass with T2.2 fixes and T2.3 tests.
- Run acceptance command: `python -m jncm.train_tokenizer --config configs/smoke.yaml --steps 50` on CPU.
- Run `python scripts/fetch_models.py` on a login node (internet access) to download `gpt2-large` into `models/gpt2-large`.
- Run `python scripts/prep_tinystories.py` on a login node to prepare `data/tinystories/`.
- Run Gate G0a on cluster GPU: train dev tokenizers (`vq` and `noisy`) and record exact-token accuracy on validation.
- Fill Slurm account and environment details.

## Environment
- Python / torch versions: PyTorch 2.x, Python 3.13.7 (Windows test host detected)
- How to activate the venv: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux)
- Data path: `data/tinystories/`   Models path: `models/`

## Next 3 steps
1. Gate G0a (Human runs dev tokenizers on GPU cluster: validation accuracy >= 99.9% for VQ, >= 99.5% for noisy)
2. T3.1 (`jncm/generator/backbone.py`: level-block transformer with no-leak guarantee)
3. T3.2 (`jncm/generator/heads.py`: ce, regress, and flow matching heads)

## Results log (human-run gates; paste numbers from results/ only)
- G0a: not run
- G0: not run
- G1: not run
- G2: not run
````