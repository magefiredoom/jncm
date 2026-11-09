"""Tests for generator training (SPEC 5.5-5.7, TASKS T3.5).

Fast tests cover checkpoint loading (frozen tokenizer, position offset), the
SPEC 5.1/5.4/5.5 loss pipeline, one packed backbone pass per step, metrics and
position accounting, the sampling/Gen PPL hook, resume, and head/variant
validation.

Acceptance (``pytest -m slow``, CPU): overfit the tokenizer on 8 fixed
sequences, then the generator; sampled outputs match the training sequences at
>= 90% token accuracy for ``ce`` and >= 80% for ``flow``. ``regress`` is a
deterministic head: it memorises ONE fixed sequence at >= 90%, and with the
true level-1 code supplied it reproduces all 8 sequences when sampling levels
2..K (SPEC 5.3, Q8).

Level 1 is fed ``zhat_0 = 0`` for every sequence (SPEC 5.1), so the only
sequence-identifying information is the level-1 code the head samples. Each
``ce``/``flow`` sample is therefore compared against its best-matching training
sequence, and the test also requires that several distinct sequences are
matched (a collapse onto one sequence would otherwise pass).
"""

from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from jncm.config import Config, GeneratorTrainConfig, load_config
from jncm.generator import (
    GeneratorBackbone,
    LevelHead,
    corrupt_codes,
    weighted_total,
)
import jncm.train_generator as tg
from jncm.tokenizer import MultiscaleTokenizer
from jncm.train_generator import (
    build_generator,
    compute_generator_loss,
    diversity_metrics,
    load_tokenizer,
    sample_and_score,
    train,
)
from jncm.train_tokenizer import train as train_tokenizer
from jncm.utils import JSONLLogger, seed_everything

L = 32
K = 4
N_SEQ = 8
TOK_STEPS = 700


def _smoke_cfg(
    head: str = "ce",
    quantizer: str = "vq",
    H: int = 32,
    n_layers: int = 1,
    n_samples: int = 2,
) -> Config:
    """Smoke config with a small, dropout-free generator for CPU tests."""
    cfg = load_config(Path("configs/smoke.yaml"))
    cfg.tokenizer.quantizer = quantizer
    cfg.generator.head = head
    cfg.generator.H = H
    cfg.generator.n_layers = n_layers
    cfg.generator.n_heads = 2
    cfg.generator.dropout = 0.0
    cfg.eval.n_samples = n_samples
    cfg.device = "cpu"
    return cfg


def _save_tokenizer_ckpt(
    tmp_path: Path,
    cfg: Config,
    tok_positions: int = 1234,
    seed: int = 0,
) -> Path:
    """Write a tokenizer checkpoint in the format `train_tokenizer` produces."""
    seed_everything(seed)
    tok = MultiscaleTokenizer(cfg.tokenizer, L=cfg.L, V=cfg.V)
    path = tmp_path / "tok.pt"
    torch.save(
        {
            "step": 0,
            "model_state_dict": tok.state_dict(),
            "config": asdict(cfg),
            "tok_positions": tok_positions,
        },
        path,
    )
    return path


def _write_data(directory: Path, sequences: torch.Tensor) -> Path:
    """Write uint16 train.bin/val.bin memmaps (as prep_tinystories.py does)."""
    directory.mkdir(parents=True, exist_ok=True)
    flat = np.asarray(sequences, dtype=np.uint16).reshape(-1)
    flat.tofile(directory / "train.bin")
    flat.tofile(directory / "val.bin")
    return directory


def _load_generator(
    ckpt_path: Path, cfg: Config
) -> tuple[MultiscaleTokenizer, GeneratorBackbone, LevelHead]:
    """Frozen tokenizer plus generator modules with the run's weights loaded."""
    tokenizer, _ = load_tokenizer(ckpt_path, cfg)
    backbone, head = build_generator(cfg, tokenizer)
    return tokenizer, backbone, head


# --------------------------------------------------------------------------- #
# Fast tests
# --------------------------------------------------------------------------- #


def test_load_tokenizer_is_frozen_and_reads_positions(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg, tok_positions=4321)

    tok, tok_positions = load_tokenizer(ckpt, cfg)

    assert tok_positions == 4321
    assert tok.training is False
    assert all(not param.requires_grad for param in tok.parameters())
    assert tok.K == cfg.tokenizer.K and tok.L == cfg.L and tok.V == cfg.V


def test_load_tokenizer_rejects_mismatches(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)

    with pytest.raises(FileNotFoundError):
        load_tokenizer(tmp_path / "missing.pt", cfg)

    wrong_L = _smoke_cfg()
    wrong_L.L = 64
    with pytest.raises(ValueError):
        load_tokenizer(ckpt, wrong_L)

    wrong_variant = _smoke_cfg(quantizer="noisy")
    with pytest.raises(ValueError):
        load_tokenizer(ckpt, wrong_variant)


@pytest.mark.parametrize("mode", ["none", "swap"])
def test_loss_matches_manual_spec_pipeline(tmp_path: Path, mode: str) -> None:
    # Right-reason check: loss = weighted per-level head loss on one packed pass
    # over chain codes corrupted once, with clean targets.
    cfg = _smoke_cfg("ce", "vq")
    cfg.generator.corrupt.mode = mode
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    tok, backbone, head = _load_generator(ckpt, cfg)

    seed_everything(1)
    x = torch.randint(0, cfg.V, (N_SEQ, cfg.L))

    got, got_per_level = compute_generator_loss(
        tok, backbone, head, x, cfg, generator=torch.Generator().manual_seed(5)
    )

    gen = torch.Generator().manual_seed(5)
    with torch.no_grad():
        enc = tok.encode(x, generator=gen)
        codes = corrupt_codes(
            enc.codes,
            cfg.generator.corrupt,
            generator=gen,
            quantizer=cfg.tokenizer.quantizer,
        )
        if mode == "swap":
            assert any(not torch.equal(c, e) for c, e in zip(codes, enc.codes))
        else:
            assert all(torch.equal(c, e) for c, e in zip(codes, enc.codes))
        level_inputs = backbone.embed_levels(tok, codes)
    hidden = backbone.forward_levels(level_inputs)
    _, per_level = head.loss(hidden, enc.targets, generator=gen)

    schedule = backbone.schedule
    # SPEC 5.5 with alpha = 0: w_k = 1 / sum_j l_j, so w_k * l_k = l_k / sum(l).
    expected = sum(
        float(length) / float(sum(schedule)) * loss
        for length, loss in zip(schedule, per_level)
    )
    assert torch.allclose(got, expected, atol=1e-6)
    assert torch.allclose(torch.stack(got_per_level), torch.stack(per_level))
    assert torch.allclose(weighted_total(per_level, schedule, cfg.generator.alpha), got)


def test_one_packed_backbone_pass_per_loss(tmp_path: Path) -> None:
    cfg = _smoke_cfg("ce", "vq")
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    tok, backbone, head = _load_generator(ckpt, cfg)

    x = torch.randint(0, cfg.V, (N_SEQ, cfg.L))
    shapes: list[tuple[int, ...]] = []
    # One block call per layer per packed pass; the trunk's forward_levels is
    # called as a plain method, so the first block is what a hook observes.
    handle = backbone.trunk.blocks[0].register_forward_hook(
        lambda module, args, output: shapes.append(tuple(args[0].shape))
    )
    try:
        compute_generator_loss(
            tok, backbone, head, x, cfg, generator=torch.Generator().manual_seed(0)
        )
    finally:
        handle.remove()

    assert len(shapes) == 1, f"expected one packed pass, saw {len(shapes)}"
    assert shapes[0] == (N_SEQ * K, cfg.L, cfg.generator.H)


@pytest.mark.parametrize(("head_name", "quantizer"), [("ce", "vq"), ("flow", "noisy")])
def test_one_staged_step_gives_every_generator_parameter_a_gradient(
    tmp_path: Path, head_name: str, quantizer: str
) -> None:
    """T5.5a F11: ``input_embed`` must train, not stay at random init."""
    cfg = _smoke_cfg(head_name, quantizer)
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    tok, backbone, head_mod = _load_generator(ckpt, cfg)
    x = torch.randint(0, cfg.V, (4, cfg.L))

    loss, _ = compute_generator_loss(
        tok,
        backbone,
        head_mod,
        x,
        cfg,
        generator=torch.Generator().manual_seed(0),
    )
    loss.backward()

    missing = [
        name
        for name, param in (
            list(backbone.named_parameters()) + list(head_mod.named_parameters())
        )
        if param.grad is None
    ]
    assert not missing, f"parameters without a gradient: {missing}"


def test_grad_accum_defaults_to_one() -> None:
    """T5.5a F13: accumulation is opt-in."""
    assert GeneratorTrainConfig().grad_accum == 1


def test_grad_accum_matches_full_batch_gradients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T5.5a F13: 4 equal micro-batches average to the full-batch gradient."""
    cfg = _smoke_cfg("ce", "vq")
    cfg.generator.train.batch_size = 8
    cfg.generator.train.warmup_steps = 0
    cfg.generator.train.grad_clip = 0.0
    cfg.generator.train.eval_every = 100
    cfg.generator.train.save_every = 100
    cfg.generator.train.save_milestones = False
    cfg.eval.n_samples = 1
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    seed_everything(123)
    data_dir = _write_data(
        tmp_path / "data", torch.randint(0, cfg.V, (8, cfg.L))
    )

    calls = {"count": 0}
    real_loss_fn = tg.compute_generator_loss

    def counting_loss(*args, **kwargs):
        calls["count"] += 1
        return real_loss_fn(*args, **kwargs)

    monkeypatch.setattr(tg, "compute_generator_loss", counting_loss)
    real_step = torch.optim.AdamW.step
    captured: dict[int, list[torch.Tensor]] = {}
    micro_calls: dict[int, int] = {}

    def run(accum: int) -> None:
        grads: list[torch.Tensor] = []
        calls["count"] = 0

        def spy(optimizer, *args, **kwargs):
            if not grads:
                grads.extend(
                    param.grad.detach().clone()
                    for group in optimizer.param_groups
                    for param in group["params"]
                    if param.grad is not None
                )
            return real_step(optimizer, *args, **kwargs)

        monkeypatch.setattr(torch.optim.AdamW, "step", spy)
        cfg.generator.train.grad_accum = accum
        train(
            cfg,
            ckpt,
            max_steps=1,
            run_id=f"accum{accum}",
            results_dir=tmp_path / f"res{accum}",
            data_dir=data_dir,
        )
        captured[accum] = grads
        micro_calls[accum] = calls["count"]

    run(1)
    run(4)

    assert micro_calls == {1: 1, 4: 4}
    full, accum = captured[1], captured[4]
    assert len(full) == len(accum)
    max_diff = max(
        float((left - right).abs().max()) for left, right in zip(full, accum)
    )
    assert max_diff <= 1e-5, f"max gradient difference {max_diff}"


def test_train_rejects_unequal_micro_batches(tmp_path: Path) -> None:
    """T5.5a F13: equal micro-batches are required for loss averaging."""
    cfg = _smoke_cfg()
    cfg.generator.train.grad_accum = 3  # batch 16 is not divisible by 3
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    data_dir = _write_data(
        tmp_path / "data", torch.randint(0, cfg.V, (4, cfg.L))
    )
    with pytest.raises(ValueError, match="divisible"):
        train(
            cfg,
            ckpt,
            max_steps=1,
            run_id="bad_accum",
            results_dir=tmp_path / "results",
            data_dir=data_dir,
        )


def test_grad_accum_position_accounting_uses_full_batch(tmp_path: Path) -> None:
    """T5.5a F13: one optimizer step still charges the whole batch."""
    cfg = _smoke_cfg()
    cfg.generator.train.batch_size = 4
    cfg.generator.train.grad_accum = 2
    cfg.generator.train.warmup_steps = 0
    cfg.generator.train.eval_every = 1
    cfg.generator.train.save_every = 10
    cfg.generator.train.save_milestones = False
    cfg.eval.n_samples = 1
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    data_dir = _write_data(
        tmp_path / "data", torch.randint(0, cfg.V, (4, cfg.L))
    )

    run_dir = train(
        cfg,
        ckpt,
        max_steps=2,
        run_id="accum_accounting",
        results_dir=tmp_path / "results",
        data_dir=data_dir,
    )

    records = JSONLLogger(run_dir / "metrics.jsonl").read_all()
    assert [record["step"] for record in records] == [1, 2]
    assert [record["gen_positions"] for record in records] == [
        4 * cfg.L,
        2 * 4 * cfg.L,
    ]
    assert all(record["sec_per_step"] > 0.0 for record in records)


def test_train_writes_metrics_and_position_accounting(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    cfg.generator.train.batch_size = 4
    cfg.generator.train.eval_every = 2
    cfg.generator.train.save_every = 2
    cfg.generator.train.warmup_steps = 1
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg, tok_positions=777)
    data_dir = _write_data(
        tmp_path / "data", torch.randint(0, cfg.V, (4, cfg.L))
    )

    run_dir = train(
        cfg,
        ckpt,
        max_steps=4,
        run_id="gen_smoke",
        results_dir=tmp_path / "results",
        data_dir=data_dir,
    )

    assert run_dir == tmp_path / "results" / "gen_smoke"
    for name in ("config.yaml", "git_sha.txt", "metrics.jsonl"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "ckpt" / "latest.pt").exists()

    saved = load_config(run_dir / "config.yaml")
    assert saved.generator.train.max_steps == 4
    assert saved.generator.H == cfg.generator.H

    records = JSONLLogger(run_dir / "metrics.jsonl").read_all()
    assert [record["step"] for record in records] == [1, 2, 4]
    last = records[-1]
    assert math.isfinite(last["train_loss"]) and last["train_loss"] > 0.0
    assert len(last["per_level_loss"]) == K
    assert last["gen_positions"] == 4 * 4 * cfg.L
    assert last["tok_positions"] == 777
    assert last["total_positions"] == 777 + 4 * 4 * cfg.L
    assert last["sample_entropy"] >= 0.0
    assert "gen_ppl" not in last
    assert not list((run_dir / "samples").glob("step_*.txt"))


def test_train_eval_hook_writes_samples_and_gen_ppl(tmp_path: Path) -> None:
    cfg = _smoke_cfg(n_samples=3)
    cfg.generator.train.batch_size = 4
    cfg.generator.train.eval_every = 2
    cfg.generator.train.save_every = 2
    cfg.generator.train.warmup_steps = 1
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    data_dir = _write_data(
        tmp_path / "data", torch.randint(0, cfg.V, (4, cfg.L))
    )

    decoded: list[list[int]] = []

    def decode_fn(ids: list[int]) -> str:
        decoded.append(ids)
        return " ".join(str(i) for i in ids)

    def scorer(texts: list[str]) -> tuple[float, int]:
        total = sum(len(text) for text in texts)
        return float(total), total

    run_dir = train(
        cfg,
        ckpt,
        max_steps=2,
        run_id="gen_hook",
        results_dir=tmp_path / "results",
        data_dir=data_dir,
        decode_fn=decode_fn,
        scorer=scorer,
    )

    records = JSONLLogger(run_dir / "metrics.jsonl").read_all()
    last = records[-1]
    # One NLL per token over equal-length texts -> exp(1).
    assert abs(last["gen_ppl"] - math.e) < 1e-3
    assert len(decoded) == 3 * len(records)
    expected_diversity = diversity_metrics(decoded[-3:])
    assert last["unique_frac"] == pytest.approx(expected_diversity["unique_frac"])
    assert last["pairwise_agree"] == pytest.approx(
        expected_diversity["pairwise_agree"]
    )

    sample_files = sorted((run_dir / "samples").glob("step_*.txt"))
    assert len(sample_files) == len(records)
    lines = sample_files[-1].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert lines == [" ".join(str(i) for i in ids) for ids in decoded[-3:]]


def test_diversity_metrics_endpoints() -> None:
    # SPEC 6.1: identical samples are one distinct sample and agree everywhere.
    identical = [[1, 2, 3, 4]] * 4
    metrics = diversity_metrics(identical)
    assert metrics["unique_frac"] == pytest.approx(1 / 4)
    assert metrics["pairwise_agree"] == pytest.approx(1.0)

    # Fully different samples with disjoint tokens: all distinct, agree nowhere.
    different = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    metrics = diversity_metrics(different)
    assert metrics["unique_frac"] == pytest.approx(1.0)
    assert metrics["pairwise_agree"] == pytest.approx(0.0)

    # No pairs exist with a single sample.
    metrics = diversity_metrics([[1, 2, 3]])
    assert metrics["unique_frac"] == pytest.approx(1.0)
    assert metrics["pairwise_agree"] == pytest.approx(0.0)


def test_resume_continues_step_and_positions(tmp_path: Path) -> None:
    cfg = _smoke_cfg()
    cfg.generator.train.batch_size = 4
    cfg.generator.train.eval_every = 2
    cfg.generator.train.save_every = 2
    cfg.generator.train.warmup_steps = 1
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg, tok_positions=99)
    data_dir = _write_data(
        tmp_path / "data", torch.randint(0, cfg.V, (4, cfg.L))
    )
    results = tmp_path / "results"

    run_dir = train(
        cfg,
        ckpt,
        max_steps=2,
        run_id="gen_resume",
        results_dir=results,
        data_dir=data_dir,
    )
    latest = run_dir / "ckpt" / "latest.pt"
    assert latest.exists()

    train(
        cfg,
        ckpt,
        max_steps=4,
        run_id="gen_resume",
        resume_path=latest,
        results_dir=results,
        data_dir=data_dir,
    )

    records = JSONLLogger(run_dir / "metrics.jsonl").read_all()
    last = records[-1]
    assert last["step"] == 4
    assert last["gen_positions"] == 4 * 4 * cfg.L
    assert last["tok_positions"] == 99

    state = torch.load(latest, map_location="cpu")
    assert state["step"] == 3
    assert state["gen_positions"] == 4 * 4 * cfg.L


@pytest.mark.parametrize(
    "head,quantizer", [("regress", "vq"), ("ce", "noisy"), ("flow", "vq")]
)
def test_head_variant_mismatch_rejected(
    tmp_path: Path, head: str, quantizer: str
) -> None:
    cfg = _smoke_cfg(head, quantizer)
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    tok, _ = load_tokenizer(ckpt, cfg)
    with pytest.raises(ValueError):
        build_generator(cfg, tok)


def test_noise_corruption_needs_noisy_variant(tmp_path: Path) -> None:
    # corrupt.mode='noise' on a vq tokenizer must fail loudly (SPEC 5.4).
    cfg = _smoke_cfg("ce", "vq")
    cfg.generator.corrupt.mode = "noise"
    ckpt = _save_tokenizer_ckpt(tmp_path, cfg)
    tok, backbone, head = _load_generator(ckpt, cfg)
    with pytest.raises(ValueError):
        compute_generator_loss(
            tok,
            backbone,
            head,
            torch.randint(0, cfg.V, (2, cfg.L)),
            cfg,
            generator=torch.Generator().manual_seed(0),
        )


# --------------------------------------------------------------------------- #
# Acceptance (slow): memorise 8 fixed sequences
# --------------------------------------------------------------------------- #


def _fixed_sequences() -> torch.Tensor:
    """Eight distinct fixed token sequences [8, 32] over the smoke vocab."""
    gen = torch.Generator().manual_seed(2026)
    return torch.randint(0, 64, (N_SEQ, L), generator=gen)


@pytest.fixture(scope="module")
def memorisation_ckpts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Overfit one tokenizer per variant on the 8 fixed sequences."""
    root = tmp_path_factory.mktemp("t35_memorise")
    data_dir = _write_data(root / "data", _fixed_sequences())
    ckpts: dict[str, Path] = {}
    for quantizer in ("vq", "noisy"):
        cfg = _smoke_cfg("ce" if quantizer == "vq" else "regress", quantizer)
        cfg.train.lr = 5e-3
        cfg.train.min_lr = 5e-4
        cfg.train.warmup_steps = 20
        cfg.train.batch_size = N_SEQ
        cfg.train.eval_every = 10**9
        cfg.train.save_every = 10**9
        cfg.train.max_steps = TOK_STEPS
        run_dir = train_tokenizer(
            cfg=cfg,
            max_steps=TOK_STEPS,
            run_id=f"tok_{quantizer}",
            results_dir=root / "tok_results",
            data_dir=data_dir,
            allow_synthetic=False,
        )
        ckpts[quantizer] = run_dir / "ckpt" / "latest.pt"
    return ckpts


@pytest.mark.slow
@pytest.mark.parametrize(
    "quantizer,head_name,steps,lr,threshold",
    [
        pytest.param("vq", "ce", 600, 1e-3, 0.90, id="ce"),
        pytest.param("noisy", "flow", 1200, 2e-3, 0.80, id="flow"),
    ],
)
def test_generator_memorises_eight_sequences(
    tmp_path: Path,
    memorisation_ckpts: dict[str, Path],
    quantizer: str,
    head_name: str,
    steps: int,
    lr: float,
    threshold: float,
) -> None:
    sequences = _fixed_sequences()
    cfg = _smoke_cfg(head_name, quantizer, H=64, n_layers=2)
    cfg.generator.train.batch_size = N_SEQ
    cfg.generator.train.lr = lr
    cfg.generator.train.min_lr = lr / 10
    cfg.generator.train.warmup_steps = 10
    cfg.generator.train.weight_decay = 0.0
    cfg.generator.train.grad_clip = 1.0
    cfg.generator.train.eval_every = 10**9
    cfg.generator.train.save_every = 10**9
    # mode 'none': this is a memorisation test. At level 1 a swap would replace
    # the only sequence-identifying code, so the deeper levels learn to ignore
    # the prefix (measured: ce 1.000 -> 0.277).
    cfg.generator.corrupt.mode = "none"

    data_dir = _write_data(tmp_path / "data", sequences)
    ckpt_path = memorisation_ckpts[quantizer]
    run_dir = train(
        cfg,
        ckpt_path,
        max_steps=steps,
        run_id="gen_mem",
        results_dir=tmp_path / "results",
        data_dir=data_dir,
        allow_synthetic=False,
    )

    tokenizer, backbone, head = _load_generator(ckpt_path, cfg)
    state = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu")
    backbone.load_state_dict(state["backbone"])
    head.load_state_dict(state["head"])

    # The frozen tokenizer must round-trip the 8 sequences, else the generator
    # is being asked to reproduce something the tokenizer itself cannot.
    with torch.no_grad():
        enc = tokenizer.encode(
            sequences, generator=torch.Generator().manual_seed(0)
        )
        round_trip = (
            tokenizer.decode(enc.zhat[-1]).argmax(dim=-1) == sequences
        ).float().mean()
    assert float(round_trip) >= 0.95, f"tokenizer round trip {float(round_trip):.3f}"

    tokens, _, _ = sample_and_score(
        tokenizer,
        backbone,
        head,
        N_SEQ,
        generator=torch.Generator().manual_seed(0),
    )

    # [samples, training sequences] token accuracy, best match per sample.
    matches = (tokens[:, None, :] == sequences[None, :, :]).float().mean(dim=-1)
    best_acc, best_idx = matches.max(dim=1)
    mean_best = float(best_acc.mean())
    distinct = len(set(best_idx.tolist()))

    assert mean_best >= threshold, (
        f"{head_name}: best-match token accuracy {mean_best:.3f} "
        f"< {threshold:.2f} ({best_acc.tolist()})"
    )
    assert distinct >= 3, (
        f"{head_name}: samples match only {distinct} distinct training "
        f"sequences; the generator collapsed"
    )


def _train_regress_memoriser(
    tmp_path: Path,
    ckpt_path: Path,
    sequences: torch.Tensor,
    steps: int,
) -> tuple[MultiscaleTokenizer, GeneratorBackbone, LevelHead]:
    """Train the deterministic ``regress`` head on fixed sequences."""
    cfg = _smoke_cfg("regress", "noisy", H=64, n_layers=2)
    cfg.generator.train.batch_size = sequences.shape[0]
    cfg.generator.train.lr = 1e-3
    cfg.generator.train.min_lr = 1e-4
    cfg.generator.train.warmup_steps = 10
    cfg.generator.train.weight_decay = 0.0
    cfg.generator.train.grad_clip = 1.0
    cfg.generator.train.eval_every = 10**9
    cfg.generator.train.save_every = 10**9
    cfg.generator.corrupt.mode = "none"

    data_dir = _write_data(tmp_path / "data", sequences)
    run_dir = train(
        cfg,
        ckpt_path,
        max_steps=steps,
        run_id="gen_regress_mem",
        results_dir=tmp_path / "results",
        data_dir=data_dir,
        allow_synthetic=False,
    )

    tokenizer, backbone, head = _load_generator(ckpt_path, cfg)
    state = torch.load(run_dir / "ckpt" / "latest.pt", map_location="cpu")
    backbone.load_state_dict(state["backbone"])
    head.load_state_dict(state["head"])
    tokenizer.eval()
    backbone.eval()
    head.eval()
    return tokenizer, backbone, head


@pytest.mark.slow
def test_regress_memorises_one_sequence(
    tmp_path: Path, memorisation_ckpts: dict[str, Path]
) -> None:
    # A deterministic head at level 1 can only emit one code (SPEC 5.3, Q8),
    # so the unconditional acceptance is one fixed sequence.
    sequences = _fixed_sequences()[:1]
    tokenizer, backbone, head = _train_regress_memoriser(
        tmp_path, memorisation_ckpts["noisy"], sequences, steps=1000
    )

    tokens, _, _ = sample_and_score(
        tokenizer,
        backbone,
        head,
        1,
        generator=torch.Generator().manual_seed(0),
    )
    accuracy = float((tokens[0] == sequences[0]).float().mean())
    assert accuracy >= 0.90, (
        f"regress one-sequence token accuracy {accuracy:.3f} < 0.90"
    )


@pytest.mark.slow
def test_regress_teacher_forced_level_one_reproduces_eight(
    tmp_path: Path, memorisation_ckpts: dict[str, Path]
) -> None:
    # Supply the true level-1 code, then sample levels 2..K from the regress
    # head; all 8 sequences must be reproduced (SPEC 5.3, Q8).
    sequences = _fixed_sequences()
    tokenizer, backbone, head = _train_regress_memoriser(
        tmp_path, memorisation_ckpts["noisy"], sequences, steps=2500
    )

    with torch.no_grad():
        enc = tokenizer.encode(
            sequences, generator=torch.Generator().manual_seed(0)
        )
        zhat = tokenizer.contrib(1, enc.codes[0])  # true level-1 code
        generator = torch.Generator().manual_seed(0)
        for k in range(2, tokenizer.K + 1):
            level_input = backbone.embed_level(tokenizer, zhat, k)
            hidden = backbone(level_input)
            code = head.sample_level(hidden, k, generator=generator)
            zhat = zhat + tokenizer.contrib(k, code)
        tokens = tokenizer.decode(zhat).argmax(dim=-1)

    accuracy = (tokens == sequences).float().mean(dim=1)
    assert float(accuracy.min()) >= 0.90, (
        f"teacher-forced regress accuracy per sequence {accuracy.tolist()} "
        "has a sequence below 0.90"
    )
