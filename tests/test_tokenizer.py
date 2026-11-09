"""
Comprehensive tests for MultiscaleTokenizer (SPEC Section 4, TASKS T2.2).
Tests shapes, chain identities, reconstruction, nested dropout, EMA updates,
dead code reinit, unit power, and batch overfitting on CPU.
"""

import math
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from jncm.config import Config, TokenizerConfig
from jncm.tokenizer import MultiscaleTokenizer, sample_cutoff
from jncm.utils import seed_everything


def test_tokenizer_shapes_vq() -> None:
    seed_everything(42)
    B, L, V = 4, 32, 64
    cfg = TokenizerConfig(d=64, K=4, quantizer="vq", V_k=64, d_code=16)
    tok = MultiscaleTokenizer(cfg, L=L, V=V)
    tok.eval()

    x = torch.randint(0, V, (B, L))
    enc = tok.encode(x)

    # Level lengths: [1, 11, 22, 32]
    expected_ls = [1, 11, 22, 32]
    assert len(enc.targets) == 4
    assert len(enc.codes) == 4
    assert len(enc.zhat) == 5

    for k, l_k in enumerate(expected_ls):
        assert enc.targets[k].shape == (B, l_k)
        assert enc.targets[k].dtype == torch.long
        assert enc.codes[k].shape == (B, l_k, 16)
        assert enc.zhat[k + 1].shape == (B, L, 64)

    logits = tok.decode(enc.zhat[-1])
    assert logits.shape == (B, L, V)


def test_tokenizer_shapes_noisy() -> None:
    seed_everything(42)
    B, L, V = 4, 32, 64
    cfg = TokenizerConfig(d=64, K=4, quantizer="noisy", d_k=8, sigma_k=0.4)
    tok = MultiscaleTokenizer(cfg, L=L, V=V)
    tok.eval()

    x = torch.randint(0, V, (B, L))
    enc = tok.encode(x)

    expected_ls = [1, 11, 22, 32]
    assert len(enc.targets) == 4
    assert len(enc.codes) == 4
    assert len(enc.zhat) == 5

    for k, l_k in enumerate(expected_ls):
        # Continuous clean code target
        assert enc.targets[k].shape == (B, l_k, 8)
        assert enc.codes[k].shape == (B, l_k, 8)
        assert enc.zhat[k + 1].shape == (B, L, 64)

    logits = tok.decode(enc.zhat[-1])
    assert logits.shape == (B, L, V)


def test_chain_identity() -> None:
    # Acceptance (b): zhat[k] - zhat[k-1] equals level-k contrib
    seed_everything(101)
    B, L, V = 4, 32, 64
    for q_type in ["vq", "noisy"]:
        cfg = TokenizerConfig(d=64, K=4, quantizer=q_type)
        tok = MultiscaleTokenizer(cfg, L=L, V=V)
        tok.eval()
        x = torch.randint(0, V, (B, L))
        enc = tok.encode(x)

        for k in range(1, 5):
            contrib_k = tok.contrib(k, enc.codes[k - 1])
            diff = enc.zhat[k] - enc.zhat[k - 1]
            assert torch.allclose(diff, contrib_k, atol=1e-5)


def test_reconstruct_reproduces_zhat_exactly() -> None:
    # Acceptance (c): reconstruct(enc.codes) reproduces enc.zhat exactly
    seed_everything(202)
    B, L, V = 4, 32, 64
    for q_type in ["vq", "noisy"]:
        cfg = TokenizerConfig(d=64, K=4, quantizer=q_type)
        tok = MultiscaleTokenizer(cfg, L=L, V=V)
        tok.eval()
        x = torch.randint(0, V, (B, L))
        enc = tok.encode(x)

        recon_zhat = tok.reconstruct(enc.codes)
        assert len(recon_zhat) == len(enc.zhat)
        for z_orig, z_rec in zip(enc.zhat, recon_zhat):
            assert torch.allclose(z_orig, z_rec, atol=1e-5)


def test_cutoff_decoder_input() -> None:
    # Acceptance (d): with cutoff k the decoder input is zhat[k]
    seed_everything(303)
    B, L, V = 4, 32, 64
    cfg = TokenizerConfig(d=64, K=4, quantizer="vq")
    tok = MultiscaleTokenizer(cfg, L=L, V=V)
    tok.eval()
    x = torch.randint(0, V, (B, L))

    for k in [1, 2, 3, 4]:
        logits, enc, cut = tok(x, cutoff=k)
        assert cut == k
        assert len(enc.zhat) == k + 1
        logits_direct = tok.decode(enc.zhat[k])
        assert torch.equal(logits, logits_direct)


def test_cutoff_out_of_range_raises() -> None:
    seed_everything(505)
    cfg = TokenizerConfig(d=64, K=4, quantizer="vq")
    tok = MultiscaleTokenizer(cfg, L=32, V=64)
    tok.eval()
    x = torch.randint(0, 64, (2, 32))
    with pytest.raises(ValueError):
        tok(x, cutoff=5)


def test_nested_dropout_frequencies() -> None:
    # Acceptance (e): nested-dropout sampler empirical cutoff frequencies match p_drop
    seed_everything(404)
    K = 4
    p_drop = 0.5
    n_samples = 10000

    cuts = [sample_cutoff(K, p_drop=p_drop) for _ in range(n_samples)]
    k_count = sum(1 for c in cuts if c == K)
    sub_k_count = sum(1 for c in cuts if c < K)

    freq_k = k_count / n_samples
    freq_sub_k = sub_k_count / n_samples

    # With p_drop=0.5, cut == K with prob 0.5, cut < K with prob 0.5
    assert abs(freq_k - 0.5) < 0.02
    assert abs(freq_sub_k - 0.5) < 0.02

    # Each sub-level in 1..K-1 should have prob ~ 0.5 / 3 ~ 0.1667
    for sub_k in range(1, K):
        freq_sub = sum(1 for c in cuts if c == sub_k) / n_samples
        assert abs(freq_sub - (p_drop / (K - 1))) < 0.02


def test_sample_cutoff_accepts_a_cpu_generator() -> None:
    """T5.5a F12: the draws follow the generator's device (CPU case)."""
    gen = torch.Generator(device="cpu").manual_seed(7)
    cuts = [sample_cutoff(4, p_drop=1.0, generator=gen) for _ in range(32)]
    assert all(1 <= cut < 4 for cut in cuts)

    def replay() -> list[int]:
        generator = torch.Generator(device="cpu").manual_seed(3)
        return [sample_cutoff(4, p_drop=0.5, generator=generator) for _ in range(16)]

    assert replay() == replay()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sample_cutoff_accepts_a_cuda_generator() -> None:
    """T5.5a F12: a CUDA generator must not be used for CPU draws."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    cuts = [sample_cutoff(4, p_drop=1.0, generator=gen) for _ in range(32)]
    assert all(1 <= cut < 4 for cut in cuts)

    generator = torch.Generator(device="cuda").manual_seed(3)
    cuts = [sample_cutoff(4, p_drop=0.5, generator=generator) for _ in range(16)]
    assert all(1 <= cut <= 4 for cut in cuts)


def test_vq_ema_update_and_dead_code_reinit() -> None:
    # Acceptance (f): vq EMA update changes codebooks; dead-code reinit triggers
    seed_everything(505)
    B, L, V = 8, 32, 64
    cfg = TokenizerConfig(d=64, K=4, quantizer="vq", V_k=64, d_code=16)
    tok = MultiscaleTokenizer(cfg, L=L, V=V)
    tok.train()

    quantizer = tok.quantizers[0]
    initial_w = quantizer.embedding.weight.clone()

    x = torch.randint(0, V, (B, L))
    _ = tok(x)  # Forward triggers EMA update

    updated_w = quantizer.embedding.weight
    # EMA update must change codebook weights
    assert not torch.equal(initial_w, updated_w)

    # Engineer a dead code: set usage of code 5 to zero
    quantizer.cluster_size[5] = 0.0
    u_dummy = torch.randn(20, 16)
    n_reinit = quantizer.reinit_dead_codes(u_dummy, threshold=1e-4)

    assert n_reinit >= 1
    assert quantizer.cluster_size[5] > 0.0


def test_noisy_unit_power_and_empirical_noise_std() -> None:
    # Acceptance (g): clean code c_k has unit power (LayerNorm), codes-targets std matches sigma_k
    seed_everything(606)
    B, L, V = 128, 32, 64
    sigma_k = 0.4
    cfg = TokenizerConfig(d=64, K=4, quantizer="noisy", d_k=8, sigma_k=sigma_k)
    tok = MultiscaleTokenizer(cfg, L=L, V=V)
    tok.eval()

    x = torch.randint(0, V, (B, L))
    enc = tok.encode(x)

    for k in range(4):
        clean_c = enc.targets[k]  # [B, l_k, d_k]
        # Unit variance along the code dimension (LayerNorm with no affine)
        var_per_pos = torch.var(clean_c, dim=-1, unbiased=False)
        assert torch.allclose(var_per_pos, torch.ones_like(var_per_pos), atol=1e-4)

        # Difference is channel noise
        diff = enc.codes[k] - clean_c
        empirical_std = torch.std(diff).item()
        assert abs(empirical_std - sigma_k) < 0.03


def test_tokenizer_checkpoint_roundtrip(tmp_path: Path) -> None:
    # Acceptance (f): save/load a tokenizer checkpoint and get identical outputs
    seed_everything(909)
    B, L, V = 4, 32, 64
    x = torch.randint(0, V, (B, L))

    for q_type in ["vq", "noisy"]:
        cfg = TokenizerConfig(d=64, K=4, quantizer=q_type)
        tok = MultiscaleTokenizer(cfg, L=L, V=V)
        tok.eval()

        ckpt_path = tmp_path / f"{q_type}.pt"
        torch.save(tok.state_dict(), ckpt_path)

        tok_loaded = MultiscaleTokenizer(cfg, L=L, V=V)
        tok_loaded.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        tok_loaded.eval()

        gen_a = torch.Generator().manual_seed(123)
        gen_b = torch.Generator().manual_seed(123)
        logits_a = tok.decode(tok.encode(x, generator=gen_a).zhat[-1])
        logits_b = tok_loaded.decode(tok_loaded.encode(x, generator=gen_b).zhat[-1])
        assert torch.equal(logits_a, logits_b)


def test_overfit_batch_vq() -> None:
    # Acceptance (h): VQ reaches >= 99% token accuracy on 16 fixed sequences within 600 steps
    seed_everything(707)
    B, L, V = 16, 32, 64
    cfg = TokenizerConfig(
        d=64,
        K=4,
        quantizer="vq",
        V_k=64,
        d_code=16,
        encoder_blocks=0,
        decoder_blocks=1,
    )
    tok = MultiscaleTokenizer(cfg, L=L, V=V)
    tok.train()

    optimizer = torch.optim.AdamW(tok.parameters(), lr=3e-3, weight_decay=0.0)
    fixed_x = torch.randint(0, V, (B, L))

    final_acc = 0.0
    for step in range(600):
        optimizer.zero_grad()
        # Train without nested dropout to overfit full representation
        logits, enc, _ = tok(fixed_x, cutoff=4)
        ce_loss = F.cross_entropy(logits.view(-1, V), fixed_x.view(-1))
        loss = ce_loss + enc.quant_loss
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == fixed_x).float().mean().item()
            final_acc = acc
            if acc >= 0.99:
                break

    assert final_acc >= 0.99, f"VQ overfit reached {final_acc:.4f}, expected >= 0.99"


def test_overfit_batch_noisy() -> None:
    # Acceptance (h): Noisy reaches >= 99% token accuracy on 16 fixed sequences within 600 steps
    seed_everything(808)
    B, L, V = 16, 32, 64
    cfg = TokenizerConfig(
        d=64,
        K=4,
        quantizer="noisy",
        d_k=8,
        sigma_k=0.4,
        encoder_blocks=0,
        decoder_blocks=1,
    )
    tok = MultiscaleTokenizer(cfg, L=L, V=V)
    tok.train()

    optimizer = torch.optim.AdamW(tok.parameters(), lr=3e-3, weight_decay=0.0)
    fixed_x = torch.randint(0, V, (B, L))

    final_acc = 0.0
    for step in range(600):
        optimizer.zero_grad()
        logits, _, _ = tok(fixed_x, cutoff=4)
        loss = F.cross_entropy(logits.view(-1, V), fixed_x.view(-1))
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == fixed_x).float().mean().item()
            final_acc = acc
            if acc >= 0.99:
                break

    assert final_acc >= 0.99, f"Noisy overfit reached {final_acc:.4f}, expected >= 0.99"