# SPEC — jncm (v1.2, 2026-11-05)

Source of truth for all implementation work. Items tagged **[DECISION]** are design choices that are NOT taken from the NCM paper. They belong to the researcher; do not change them, and record any problem you find in docs/QUESTIONS.md.

**v1.1 changes** (from docs/REVIEW_M0-M2.md): learned attention projections made explicit (4.1); vq `codes` are straight-through vectors and the generator conditions from `targets` (4.1); vq arithmetic in fp32 (4.2); bounded-memory data prep and no silent synthetic fallback (3); unknown config keys raise (7).

**v1.2 change:** level-schedule ties round half up, defined with exact integer arithmetic (2).

## 1. Purpose and scope

Compare **discrete** (vector-quantized) and **continuous** multiscale residual hierarchies for coarse-to-fine text generation, then add JEPA-style joint training. All variants share one backbone so comparisons are controlled.

Baseline: NCM, Next-Concept Modeling (Kumbong et al., arXiv 2610.14162). A multiscale residual VQ-VAE turns a length-L token sequence into K levels of discrete codes (level k has l_k codes, l_1 < ... < l_K = L). A transformer generates the levels coarse to fine, all codes of one level in parallel, conditioned on the cumulative reconstruction of the earlier levels. The VQ-VAE decoder maps the final cumulative reconstruction to tokens.

- Dev dataset: TinyStories, L = 128. LM1B / Sudoku / XSum come later (TASKS M6).
- Out of scope: generators above ~100M parameters, distributed training, any UI.

Reference numbers (TinyStories; GPT-2-large generative perplexity / mean unigram entropy H):
- real validation text: 9.67 / 3.96
- NCM, 12 sampling steps: 15.41 / 3.98
- NCM tokenizer exact-token reconstruction: 99.98 %

Use these to validate the evaluation protocol and sanity-check runs. Never tune toward them.

## 2. Notation and defaults

| Symbol | Meaning | Dev default | Smoke default |
|---|---|---|---|
| B | batch size | 512 (tok), 1024 (gen) | 16 |
| L | sequence length | 128 | 32 |
| V | text vocab size | 8192 | 64 |
| d | tokenizer width | 384 | 64 |
| K | number of levels | 12 | 4 |
| l_k | codes at level k | schedule below | schedule below |
| V_k | codebook size (vq) | 4096 | 64 |
| d_code | VQ code width | 64 | 16 |
| d_k | continuous code width | 8 | 8 |
| sigma_k | channel noise std (noisy) | 0.4 | 0.4 |
| H | generator width | 768 | 128 |

Level schedule: `l_k = 1 + (2*(k-1)*(L-1) + (K-1)) // (2*(K-1))` for k = 1..K. This is `round(1 + (k-1)*(L-1)/(K-1))` with exact ties rounded **up** (not Python's banker's rounding), computed in integers.
- L=128, K=12 gives [1, 13, 24, 36, 47, 59, 70, 82, 93, 105, 116, 128] (sum 774).
- L=32, K=4 gives [1, 11, 22, 32] (sum 66).
- Tie examples: L=128, K=11 gives 65 at k=6 (64.5 rounds up); L=4, K=3 gives [1, 3, 4].
- Must satisfy l_1 = 1, l_K = L, strictly increasing (unit test).

Tensor layout: pooling and interpolation operate on [B, C, T]. Our modules use [B, T, C]; transpose explicitly and unit-test with l = 1 and l = L (`down(x, L)` and `up(x, L)` must be the identity).

## 3. Data

- Dataset: HF `roneneldan/TinyStories`, train and validation splits.
- Tokenizer: BPE, vocab 8192, trained with HF `tokenizers` on a subset of the train text; specials `<pad>`, `<eos>`.
- Packing: concatenate stories separated by `<eos>`, cut into non-overlapping windows of exactly L tokens (drop the remainder). Store uint16 memmaps `data/tinystories/{train,val}.bin` plus `meta.json` (counts, V, L) and `tokenizer.json`.
- Download and tokenization happen only in `scripts/prep_tinystories.py` (login node). Training and eval read memmaps only. Prep runs in bounded memory: tokenize and pack in chunks (about 20,000 stories), write windows incrementally, carry the unfinished tail to the next chunk. Never hold the full dataset in Python lists (about 37 bytes per token).
- Unit tests use synthetic random sequences (seeded). They never touch real data or the network.
- No silent fallback to synthetic data. If real `train.bin` / `val.bin` are missing, training raises `FileNotFoundError`; if `meta.json` exists and its `L` or `V` differ from the config, it raises `ValueError`. Synthetic data is allowed only with an explicit `--allow_synthetic` flag (smoke tests) and is recorded as `data_source` in the run's config or metrics.

## 4. Multiscale residual tokenizer (`jncm/tokenizer/`)

### 4.1 Shared trunk

```
z      = Emb(x) + Pos                    # [B,L,d]; Emb = nn.Embedding(V,d); Pos = learned [L,d]
z      = EncBlocks(z)                    # config encoder_blocks, default 0
zhat_0 = 0
for k in 1..K:
    r      = z - zhat_{k-1}                          # [B,L,d]   residual
    u      = down(r, l_k)                            # [B,l_k,d] adaptive_avg_pool1d  (Down_k)
    code_k = Quantizer_k(u)                          # variant-specific (4.2 / 4.3)
    zhat_k = zhat_{k-1} + R_k(code_k)                # [B,L,d]
```

- `R_k` (reconstruction operator): Linear(code width -> d) applied to the code vectors, then linear interpolation from l_k to L positions (`F.interpolate(mode="linear", align_corners=False)`; for l_k = 1 this is a broadcast), then depthwise Conv1d(kernel 3, padding 1). **[DECISION]** (NCM does not specify the parameterization.)
- Decoder: `decoder_blocks` bidirectional pre-LN transformer blocks (default 1; d, 6 heads, learned positions added to the input) followed by Linear(d -> V). Weight tying with Emb is allowed. Each block is a standard pre-LN block: multi-head self-attention with learned q, k, v and output projections (`scaled_dot_product_attention`), a GELU MLP with ratio 4, and two LayerNorms (12d^2 + 13d parameters). Attention without projections is a bug.
- `decode(zhat_k)` returns logits [B,L,V].

API (all variants):

```python
@dataclass
class Encoded:
    targets: list[Tensor]   # per level. vq: LongTensor [B,l_k]. noisy: CLEAN code c_k [B,l_k,d_k]
    codes:   list[Tensor]   # per level, the codes actually used in the residual chain
                            #   vq: straight-through code vectors [B,l_k,d_code] (value = codebook entry).
                            #   noisy: c_k + sigma_k * eps
    zhat:    list[Tensor]   # K+1 tensors [B,L,d]; zhat[0] = 0; zhat[k] = cumulative recon after level k
    quant_loss: Tensor      # scalar (0 for noisy)

class MultiscaleTokenizer(nn.Module):
    def encode(self, x, cutoff: int | None = None) -> Encoded   # cutoff=None: all K levels
    def contrib(self, k: int, code: Tensor) -> Tensor           # R_k applied to a code tensor; [B,L,d]; no grad needed
    def reconstruct(self, codes: list[Tensor]) -> list[Tensor]  # zhat list from a list of codes (levels 1..len)
    def decode(self, zhat_k: Tensor) -> Tensor                  # logits [B,L,V]
```

`encode` followed by `reconstruct(enc.codes)` must reproduce `enc.zhat` exactly (unit test). For vq, `reconstruct(enc.targets)` (indices) must also reproduce `enc.zhat`: the generator builds its conditioning from `targets`, never from `codes`.

### 4.2 Variant `vq` (discrete baseline, NCM-style)

- u' = Linear_k(u): [B,l_k,d_code]. Codebook E_k in R^{V_k x d_code}. idx = argmin_j ||u' - E_k[j]||^2. e = E_k[idx]. Straight-through: e_st = u' + (e - u').detach(). `contrib` uses R_k(e_st) in training and R_k(E_k[idx]) when called from codes.
- Loss: per level `beta * MSE(u', sg(e))` with beta = 0.25, summed over levels up to the cutoff. **[DECISION]** (per-level commitment; NCM's Eq. 9 uses a cumulative form.)
- All vq arithmetic (projection, distances, EMA update, loss) runs in fp32 even under autocast; bf16 breaks the EMA update and degrades the argmin.
- Codebooks update by EMA, decay 0.99, no gradient. Dead-code reinit: every 200 steps replace codes whose EMA usage is below 1e-4 of the mean usage with random current u' vectors.

### 4.3 Variant `noisy` (continuous) **[DECISION]**

Why: a continuous code without a capacity limit can carry unbounded information, so the hierarchy degenerates (everything in level 1) and generating level 1 becomes as hard as the whole task. VQ limits capacity by construction. Here we limit it with a power constraint plus Gaussian channel noise.

- u' = Linear_k(u): [B,l_k,d_k]
- c_k = LayerNorm(u', elementwise_affine=False). This is the **clean code** (unit power per dimension).
- code_k = c_k + sigma_k * eps, eps ~ N(0,I). Applied in training AND eval (it is a channel, not a regularizer). Pass an explicit `torch.Generator` for reproducibility.
- contrib = R_k(code_k), with Linear(d_k -> d).
- Capacity bound per position: `(d_k/2) * log2(1 + 1/sigma_k^2)` bits. Defaults d_k = 8, sigma_k = 0.4 give 11.43 bits, versus log2(4096) = 12 bits for VQ: matched information budget per level. Implement `utils.bits_per_position(cfg)` for both variants and log it.
- No KL term. quant_loss = 0.
- Note: LayerNorm bounds power per position but does not prevent all positions from sharing one vector. The decoder CE prevents that in tokenizer training. In JEPA mode (section 8) VICReg does.

### 4.4 Training and nested dropout

Loss = CE(decode(zhat_cut), x) + quant_loss(levels <= cut).
Nested dropout (NCM): with probability p_drop (default 0.5) pick cut ~ Uniform{1..K-1}, else cut = K. One cut per batch. The decoder input is `zhat[cut]`.

Defaults (dev): AdamW, lr 5e-4 cosine to 5e-5, betas (0.9, 0.95), weight decay 0.01, warmup 1000, clip 1.0, bf16 autocast on CUDA, 40k steps, batch 512.

### 4.5 Diagnostics (logged every `eval_every` steps, on a fixed held-out batch, full hierarchy, no nested dropout)

- `acc_cum[k]` for k = 1..K: exact-token accuracy of `decode(zhat[k])`. Log only; a healthy hierarchy rises gradually (not a step at level 1, not zero until level K).
- `acc_full`: exact-token accuracy at k = K.
- vq: per-level codebook usage fraction and code perplexity.
- noisy: per-level std over the batch of every dimension of c_k (min and mean), effective rank of c_k (exp of entropy of the normalized covariance eigenvalues), nominal bits per position.

## 5. Generator (`jncm/generator/`)

### 5.1 Inputs

Given tokens x and the frozen tokenizer in eval mode (no grad):

1. `enc = tok.encode(x)`. Targets are `enc.targets`; chain codes are `enc.codes`.
2. Corrupt the chain codes (5.4) to get `codes~`. Corruption is applied once to the whole hierarchy; targets stay clean.
3. For each k: `zhat~_{k-1} = sum_{j<k} tok.contrib(j, codes~_j)`; `zhat~_0 = 0`.
4. Level-k input tokens: `s_k = down(zhat~_{k-1}, l_k)` -> [B,l_k,d]; `x_k = Linear(d->H)(s_k) + SinPos(p_i) + LevelEmb(k)`, where p_i = (i+0.5)/l_k in (0,1) and SinPos uses 64 log-spaced frequencies (sin and cos) projected to H. **[DECISION]** (normalized positions instead of RoPE, because levels have different resolutions.)

Level k must depend only on levels < k. This is the most likely place for a bug (off-by-one); it has a dedicated test (TASKS T3.1).

### 5.2 Backbone

Pre-LN transformer: 12 layers, H = 768, 12 heads, MLP 4H with GELU, dropout 0.1, bidirectional attention **within a level only** (block diagonal across levels). Implementation: pad each level to length L, run on [B*K, L, H] with a key-padding mask (use `scaled_dot_product_attention`), then gather valid positions. Training processes all K levels in one forward pass. Sampling processes one level per forward pass. Conditional generation (M6) prepends a prefix of P tokens to every level block (prefix attends to itself only; level tokens attend to prefix and own level).

### 5.3 Heads (config `head`)

- `ce` (for `vq`): per-level Linear(H -> V_k); cross-entropy on code indices.
- `regress` (for `noisy`): per-level Linear(H -> d_k) giving c-hat_k; loss MSE against clean c_k (option `cosine`). At sampling apply LayerNorm(no affine) to c-hat_k (`renorm_pred`, default true) because the regression mean has shrunken norm.
- `flow` (for `noisy`): per-level conditional flow-matching MLP on the clean code. **[DECISION]**
  - Input: x_t [N,d_k], time embedding (sinusoidal, 128 dims), hidden state h [N,H]. Network: Linear(d_k + 128 + H -> 512), SiLU, 3 residual blocks (Linear 512 -> 512, SiLU), Linear(512 -> d_k).
  - Train: for each valid position draw `n_rep` = 4 pairs (t ~ U(0,1), eps ~ N(0,I)); x_t = (1-t)*eps + t*c_k; target velocity v = c_k - eps; loss MSE(net(x_t, t, h), v).
  - Sample: Euler integration from t = 0 to 1 with `flow_steps` = 8 steps, starting from eps ~ N(0,I); output x_1. Positions are independent given h (same factorization as NCM).
- For `regress` and `flow`: after producing c-hat_k, sampling sets `code_k = c-hat_k + sigma_k * eps'` with fresh noise (`add_channel_noise`, default true), because the decoder was trained on noisy codes.

### 5.4 Conditioning corruption **[DECISION]** (principle from NCM Appendix F.3; schedule values are guesses to sweep)

`corrupt.mode` is one of `none | swap | noise`.
- `swap`: for level j with rate p_j, replace the code at (b,i) with the code at a uniformly random other (batch, position) of the same level. p_j decreases linearly from `p_first` = 0.8 to `p_last` = 0.2 (mean 0.5; NCM found moderate mean rates best).
- `noise`: add N(0, s_j^2) to chain codes (noisy variant only), s_j linearly decreasing from 0.5 to 0.1.

### 5.5 Loss

`L_gen = sum_k w_k * sum_i loss_{k,i}` with `w_k = l_k^(-alpha) / sum_j l_j^(1-alpha)`, alpha = 0 by default (equal weight per position across all levels; NCM Eq. 10).

### 5.6 Sampling

```
zhat = 0
for k in 1..K:
    s_k  = down(zhat, l_k);  h = backbone(level k only)
    code = head.sample(h)            # ce: categorical at temperature 1; regress / flow: see 5.3
    zhat = zhat + tok.contrib(k, code)
tokens = argmax(tok.decode(zhat))
```

Exactly K backbone forward passes per batch (unit test counts them with a forward hook).

### 5.7 Defaults (dev)

AdamW, lr 3e-4 cosine to 3e-5, betas (0.9, 0.95), weight decay 0.1, warmup 500, clip 1.0, bf16, batch 1024, 10k steps (the budget NCM used for its TinyStories ablations, about 1.3B positions), generator about 90M parameters.

## 6. Evaluation (`jncm/eval/`, `scripts/`)

### 6.1 Generative perplexity and entropy (protocol follows the diffusion-LM literature that NCM uses)

- Draw `n_samples` = 256 sequences, `n_seeds` = 5 (final results: 1024 samples).
- Decode ids to text with the dataset tokenizer, re-tokenize with the GPT-2 tokenizer, score with GPT-2-large loaded from a local path (`$GPT2_LARGE_PATH`, offline). Gen PPL = exp(mean token NLL over all scored tokens). Score the full re-tokenized sample (no truncation). Report mean and std over seeds.
- H = mean over samples of the entropy (nats) of the empirical unigram distribution of that sample's tokens. Compute it on GPT-2 tokens; also compute it on the dataset's own tokens and record which one matches the real-data reference.
- **Protocol gate (T1.2):** the same pipeline applied to real validation sequences must give about 9.67 PPL and about 3.96 H (tolerance 15% and 0.15). If not, debug the protocol before any model comparison. Do not adjust the metric to hit the numbers; write the diagnosis in docs/QUESTIONS.md.

### 6.2 Token accounting (for the sample-efficiency claim)

Every metrics record carries `tok_positions` (tokenizer steps x batch x L), `gen_positions` (generator steps x batch x L) and `total_positions` (sum). For comparisons, curves are plotted against `total_positions`, with the tokenizer budget added as an offset for generator curves. Both discrete and continuous runs use identical accounting.

### 6.3 Samples

Each evaluation writes 32 decoded samples to `results/<run_id>/samples/step_<n>.txt`.

## 7. Configs, logging, reproducibility

- YAML configs load into dataclasses (no hydra). CLI override: `--set a.b=value`. Unknown keys (YAML or `--set`) raise `ValueError`: a typo must never run silently. The results root is configurable (default `results`); tests use temporary directories.
- `results/<run_id>/` contains `config.yaml`, `git_sha.txt`, `metrics.jsonl` (one JSON object per eval or log step), `ckpt/`, `samples/`.
- Seed everything (python, numpy, torch, DataLoader generators). Checkpoints are resumable.
- No wandb. Logging is JSONL plus `scripts/plot_curves.py`.
- Slurm scripts read the account from `$SLURM_ACCOUNT`. Never hard-code or guess account, partition or GPU type.

## 8. JEPA joint mode (Phase 2) **[DECISION]**: the researcher will revise this after Phase-1 results

Requires `quantizer: noisy`. Modules: online encoder E (the tokenizer encode path), EMA encoder E-bar (deep copy; requires_grad False; eval mode; channel noise off), decoder D, generator G.

Per training step on batch x:

1. Online encode: clean codes c_k, chain codes, `zhat`.
2. EMA encode under no_grad with noise off: targets t_k, computed with E-bar's own residual chain; shape [B,l_k,d_k].
3. Corrupt the online chain codes (5.4); G predicts level k from `zhat~_{k-1}` (online; gradient reaches E through the conditioning).
4. `L_pred = sum_k w_k * dist(c-hat_k, sg(t_k))`. For `regress`: MSE (or cosine). For `flow`: the flow-matching target is t_k.
5. `L_dec = CE(D(zhat[cut]), x)` with nested dropout. Flag `decoder_through_E` (default true; false feeds `zhat[cut].detach()`).
6. VICReg per level (flag `vicreg`), on online c_k and on c-hat_k, over the B*l_k vectors of that level: `var = mean_d relu(gamma - std_d)` with gamma = 1; `cov = sum_{i != j} Cov_ij^2 / d_k`. `L_vic = sum_k (lambda_var * var + lambda_cov * cov)`, defaults lambda_var = 1.0, lambda_cov = 0.04 (to tune).
7. `L = L_dec + lambda_pred * L_pred + L_vic`, lambda_pred = 1.
8. After the optimizer step: `theta_bar <- m * theta_bar + (1 - m) * theta`, m cosine-scheduled from 0.996 to 1.0.

Ablation flags (all configurable): `ema` (false: targets come from the online encoder under stop-gradient), `vicreg`, `decoder_through_E`, `head` (`regress` | `flow`). The planned 2x2 is `decoder_through_E` x `vicreg`.

Diagnostics (log per level): min and mean std of c_k and t_k; effective rank; explained variance `R2_k = 1 - MSE_k / Var(t_k)`; collapse alarm if mean std < 0.1 for 200 consecutive steps.

## 9. Smoke configuration (`configs/smoke.yaml`)

L = 32, V = 64, K = 4 (l = [1, 11, 22, 32]), d = 64, d_code = 16, V_k = 64, d_k = 8, sigma = 0.4, generator H = 128 with 2 layers and 4 heads, batch 16. Every unit test runs on CPU with this config in under 60 seconds.