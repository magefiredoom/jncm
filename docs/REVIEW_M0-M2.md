# REVIEW — Gemini's M0–M2 draft (2026-11-05)

Scope: every project file in `jncm.zip` (the bundled `.venv` excluded) read against SPEC v1. Method: line-by-line read, `py_compile` on all files (clean), plus two torch-free checks I could run (below). **I could not run anything that imports torch**, so runtime claims are from reading and each has a test that will confirm or refute it.

Evidence that the draft runs: the zip contains `results/smoke_vq_run`, `results/smoke_noisy_run` and a resume run with metrics in the SPEC §4.5 schema. All smoke runs use 10 steps at lr ≈ 5e-5, so their accuracy is chance level and says nothing about learning.

## Verdict

M0–M2 is close to the spec and well structured. Four problems would waste the first GPU runs (F1–F4). Fix those first, with the tests in `tests/test_review_fixes.py`. Everything else is cleanup.

## Findings

### F1 BLOCKER — attention has no learned projections
`jncm/tokenizer/trunk.py:49-50`. `TransformerBlock` splits the normalized input into heads and calls `scaled_dot_product_attention(qkv, qkv, qkv)`. There is no Q/K/V projection and no output projection, so attention is parameter-free (q = k = v = the input). The decoder (and any encoder block) cannot learn to route information between positions. The full-hierarchy accuracy can still look fine because the finest level carries per-position codes, but coarse-level reconstructions (`acc_cum[k]` for small k, the whole point of the hierarchy) will be capped, and the overfit tests do not notice.
Fix:
```python
class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        assert d % n_heads == 0
        self.d, self.n_heads, self.dropout = d, n_heads, dropout
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp_ratio)), nn.GELU(), nn.Linear(int(d * mlp_ratio), d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))
```
Test: a standard block has `12*d*d + 13*d` parameters (the current one has `8*d*d + 9*d`).

### F2 BLOCKER — VQ will crash on GPU under bf16 autocast
`jncm/tokenizer/vq.py:74-96` with `jncm/train_tokenizer.py:212-232`. The trainer wraps the whole forward in bf16 autocast on CUDA. Inside the quantizer, `u_prime = self.proj(u)` comes out bf16, while `sums` is created with `u.dtype` (fp32), so `sums.index_add_(0, indices, flat_u)` mixes dtypes and PyTorch rejects that. Separately, the distance cross-term `matmul(flat_u, embed_w.t())` runs in bf16, which degrades the argmin assignments. CPU tests never see this because the trainer only enables autocast on CUDA.
Fix: run the whole quantizer in fp32.
```python
def forward(self, u):
    with torch.autocast(device_type=u.device.type, enabled=False):
        u = u.float()
        ...  # existing body unchanged (projection, distances, EMA update, loss, straight-through)
    return targets, codes, loss
```
Test: forward and backward under `torch.autocast("cpu", dtype=torch.bfloat16)` for both variants. If that test already passes before the fix, my crash theory is wrong; keep the fp32 change anyway for the argmin precision.

### F3 BLOCKER — data prep holds the whole dataset in Python lists
`scripts/prep_tinystories.py:27-31, 127-134`. `train_texts`, `train_tokens` and `stream` are full-dataset Python lists. I measured 37 bytes per token for a list of ints, so full TinyStories (about 0.5B tokens) needs on the order of 20 GB. The `--max_stories 2000` acceptance run cannot show this. Fix: stream in chunks. This packer is equivalent to `pack_stories` (I checked it on 300 random inputs with random chunking):
```python
class WindowPacker:
    """Streaming equivalent of pack_stories(): memory is O(chunk), not O(dataset)."""
    def __init__(self, eos_id: int, seq_len: int) -> None:
        self.eos_id, self.seq_len = eos_id, seq_len
        self._tail = np.empty(0, dtype=np.uint16)

    def add(self, stories) -> np.ndarray:
        parts = [self._tail]
        for story in stories:
            if len(story) == 0:
                continue
            arr = np.asarray(story, dtype=np.uint16)
            parts.append(arr)
            if arr[-1] != self.eos_id:
                parts.append(np.array([self.eos_id], dtype=np.uint16))
        flat = np.concatenate(parts)
        n = len(flat) // self.seq_len
        self._tail = flat[n * self.seq_len:].copy()
        return flat[: n * self.seq_len].reshape(n, self.seq_len)

    @property
    def dropped_tokens(self) -> int:
        return int(len(self._tail))
```
Use it per split: loop over chunks of about 20,000 stories (`split[i:i+chunk]["text"]`), `tokenizer.encode_batch`, `packer.add([e.ids for e in encodings])`, then `windows.tofile(open_file)` and count windows. Keep `pack_stories` for its existing tests. Download the dataset on a login node, but run tokenization as a job if the login node limits CPU time or memory.

### F4 BLOCKER — silent fallback to random data
`jncm/train_tokenizer.py:48-80`, especially the bare `except Exception: pass` at line 66. If the real `train.bin` is missing, or its length does not match `cfg.L`, the trainer silently trains on random synthetic tokens. On the cluster, a wrong path would produce a meaningless "G0a result". Fix:
- Real files present: load them with no try/except. If `meta.json` exists, raise `ValueError` when its `L` or `V` differ from the config.
- Real files absent: raise `FileNotFoundError` unless `allow_synthetic=True`. Add a `--allow_synthetic` CLI flag and a `train(..., allow_synthetic=False)` parameter. Tests and the smoke command pass it.
- Record `data_source: real|synthetic` in `config.yaml` or the first `metrics.jsonl` record.

### F5 MAJOR — misplaced files; scripts cannot import `jncm`
- `pytest.ini` and `requirements.txt` are in `configs/`, not the repo root. So `testpaths`, the `slow` marker and any `pythonpath` are not applied, and bare `pytest` cannot import `jncm` (consistent with the five `lastfailed` entries in `tests/.pytest_cache`).
- `python scripts/eval_data_row.py` (the documented T1.2 command) will fail with `ModuleNotFoundError: jncm`, because the script's directory, not the repo root, is on `sys.path`.

Fix: move both files to the root; `pytest.ini` gets `pythonpath = .`; add a minimal `pyproject.toml` and install with `pip install -e .`:
```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "jncm"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
include = ["jncm*"]
```
Acceptance: bare `pytest -q` passes from the repo root, and `python scripts/eval_data_row.py --help` works.

### F6 MAJOR — config typos are silently ignored
`jncm/config.py`, `_dict_to_dataclass` (skips keys it does not know) and `apply_overrides` (creates arbitrary keys). `--set tokenizer.sigma_K=0.3` would run the default `sigma_k` and nobody would know. Fix: raise `ValueError` listing unknown keys in `_dict_to_dataclass`. This covers YAML and `--set`.

### F7 MAJOR — VQ `Encoded.codes` are vectors; SPEC said indices
`jncm/tokenizer/vq.py:57-110` returns `targets` (indices) and `codes` (straight-through float vectors). That is the right design for the gradient path, and SPEC v1.1 now says so. Danger for later: the generator must build its conditioning from `targets` (indices), never from `codes`. Test: `tok.reconstruct(enc.targets)` must reproduce `enc.zhat` for the vq variant. The current tests only check `reconstruct(enc.codes)`.

### F8 MINOR — tests write into the real `results/` directory
`train_tokenizer.py:170` hardcodes `Path("results")`. Add a `results_dir` argument (default `"results"`) and have tests pass `tmp_path`.

### F9 MINOR — smaller items
- `prep_tinystories.py:61`: `BpeTrainer` without `initial_alphabet=ByteLevel.alphabet()` can silently drop rare bytes at encode time.
- Repo has one commit (`kit`); all Gemini code is uncommitted, so every `git_sha.txt` in `results/` names the wrong code. Commit before any real run and record a dirty-tree flag.
- `STATE.md` says "41 passed" in one place and "43 tests" in another, and no `TASKS.md` box is ticked. Re-verify in OpenCode and tick only what passes.
- `GPT2Scorer.score_sample` scores one text at a time; batch it before the generator evals (T4.2).
- Resume restores model and optimizer but not the data iterator or RNG; acceptable for dev, worth knowing for repeatability.
- Weight decay is applied to embeddings and LayerNorm parameters (0.01, low impact).
- Sharing tip: the upload was 42k files because `.venv`, `results/` and `data/` were zipped. Exclude them.

### F10 MINOR — level schedule uses banker's rounding
`jncm/levels.py:27` uses `int(round(...))`, which rounds exact ties to the nearest even number. (L=128, K=11) therefore gives l_6 = 64 where half-up gives 65, and (4, 3) gives [1, 2, 4] instead of [1, 3, 4]. The dev (128, 12) and smoke (32, 4) schedules are unaffected. SPEC v1.2 defines it exactly: `l_k = 1 + (2*(k-1)*(L-1) + (K-1)) // (2*(K-1))`. I checked this against float half-up on 10,920 (L, K) pairs: identical, always strictly increasing, l_1 = 1, l_K = L. Test: `get_level_schedule(4, 3) == [1, 3, 4]` and `get_level_schedule(128, 11)[5] == 65`.

## What checked out
Level schedule and `down`/`up` identities; chain identity and `reconstruct` for both variants; nested dropout semantics (per batch, never at generator-target time); noisy bottleneck matches SPEC §4.3 (LayerNorm unit power, channel noise in train and eval, clean targets); VQ EMA update, dead-code reinit and straight-through; commitment loss levels ≤ cut; LR schedule; token accounting (`tok_positions + gen_positions = total_positions`); Gen PPL and entropy math; the data-row protocol script's logic; checkpoint step arithmetic on resume; the metrics schema.

## Do next
1. **T2.4 in OpenCode** (prompt below): F1–F10, tests first. Copy `SPEC.md` (v1.1), `TASKS.md`, `AGENTS.md`, this file into `docs/` (AGENTS.md at the root) and `test_review_fixes.py` into `tests/` before you start.
2. On a login node: `python scripts/fetch_models.py`, then the fixed `python scripts/prep_tinystories.py`. Then run T1.2 (`scripts/eval_data_row.py`): real data should give Gen PPL ≈ 9.67 and H ≈ 3.96 (SPEC 6.1).
3. **Gate G0a** on FIR: `python -m jncm.train_tokenizer --config configs/tinystories_dev.yaml --set tokenizer.quantizer=vq --run_id tok_vq_dev`, and the same with `noisy`.
4. While those run, start M3 (T3.1–T3.3) with prompt P4.

## Prompt for OpenCode (paste into a fresh session)
```text
Read AGENTS.md, docs/STATE.md, docs/TASKS.md, then docs/REVIEW_M0-M2.md. Task: T2.4 (findings F1 to F10). Follow AGENTS.md.
1. Commit the current tree as "import Gemini M0-M2 draft" before editing anything.
2. F5 first: move pytest.ini and requirements.txt to the repo root, add pythonpath, add pyproject.toml, run `pip install -e .`. Run bare `pytest -q` and write the baseline result into docs/STATE.md.
3. Then F1 to F10 in order. For each: make sure its test in tests/test_review_fixes.py fails for the reason the review gives, apply the fix, make it pass, run the full suite, commit "F<n>: <summary>". The tests were drafted without a torch runtime; if one is wrong for a reason unrelated to its finding, fix the test and record it in docs/QUESTIONS.md.
4. If any test in F2 passes before its fix, say so in STATE.md (it means the crash theory was wrong) and keep the fp32 change.
5. Do not edit docs/SPEC.md. If a finding conflicts with it, log it in docs/QUESTIONS.md and move on.
6. At the end: update docs/STATE.md (VERIFIED needs command and output tail), tick T0.1 to T2.4 in docs/TASKS.md only where the acceptance commands passed, commit. Report in at most 15 lines.
```