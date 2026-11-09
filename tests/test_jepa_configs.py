"""Tests for configs/jepa_dev.yaml and configs/jepa_smoke.yaml (T5.4).

The dev config must be the Phase-2 base run: same tokenizer and generator sizes
as configs/tinystories_dev.yaml, head flow, online targets, EMA, VICReg and a
decoder-through-E loss, batch 1024, 10000 steps. The smoke config must be a
CPU-scale JEPA config that ``validate_jepa_config`` accepts.
"""

from __future__ import annotations

from pathlib import Path

from jncm.config import Config, load_config
from jncm.train_jepa import validate_jepa_config


def _field_equalities(cfg: Config, other: Config, fields: tuple[str, ...]) -> None:
    for name in fields:
        assert getattr(cfg, name) == getattr(other, name), name


TOKENIZER_SIZES = (
    "d",
    "K",
    "V_k",
    "d_code",
    "d_k",
    "sigma_k",
    "decoder_blocks",
    "decoder_heads",
)
GENERATOR_SIZES = ("H", "n_layers", "n_heads", "mlp_ratio", "dropout", "flow_steps")


def test_jepa_dev_is_the_phase2_base_run() -> None:
    cfg = load_config(Path("configs/jepa_dev.yaml"))
    dev = load_config(Path("configs/tinystories_dev.yaml"))

    validate_jepa_config(cfg)

    assert cfg.jepa.enabled is True
    assert cfg.tokenizer.quantizer == "noisy"
    assert cfg.generator.head == "flow"
    assert cfg.jepa.ema is True
    assert cfg.jepa.vicreg is True
    assert cfg.jepa.decoder_through_E is True
    assert cfg.jepa.target_chain == "online"
    assert cfg.generator.train.batch_size == 1024
    assert cfg.generator.train.max_steps == 10000

    assert cfg.L == dev.L and cfg.V == dev.V
    _field_equalities(cfg.tokenizer, dev.tokenizer, TOKENIZER_SIZES)
    _field_equalities(cfg.generator, dev.generator, GENERATOR_SIZES)


def test_jepa_smoke_is_cpu_scale_and_valid() -> None:
    cfg = load_config(Path("configs/jepa_smoke.yaml"))
    smoke = load_config(Path("configs/smoke.yaml"))

    validate_jepa_config(cfg)

    assert cfg.device == "cpu"
    assert cfg.jepa.enabled is True
    assert cfg.tokenizer.quantizer == "noisy"
    assert cfg.generator.head in ("regress", "flow")
    assert cfg.L == smoke.L and cfg.V == smoke.V
    _field_equalities(cfg.tokenizer, smoke.tokenizer, TOKENIZER_SIZES)
    _field_equalities(cfg.generator, smoke.generator, GENERATOR_SIZES)


def test_grad_accum_is_present_and_documents_per_micro_batch_vicreg() -> None:
    """T5.5a F13: default 1 in every config, VICReg caveat in the JEPA ones."""
    for path in (
        "configs/smoke.yaml",
        "configs/tinystories_dev.yaml",
        "configs/jepa_smoke.yaml",
        "configs/jepa_dev.yaml",
    ):
        cfg = load_config(Path(path))
        assert cfg.generator.train.grad_accum == 1, path
        assert "grad_accum: 1" in Path(path).read_text(encoding="utf-8"), path
    for path in ("configs/jepa_smoke.yaml", "configs/jepa_dev.yaml"):
        assert "per micro-batch" in Path(path).read_text(encoding="utf-8"), path


def test_both_configs_document_the_head_restriction() -> None:
    for path in ("configs/jepa_dev.yaml", "configs/jepa_smoke.yaml"):
        text = Path(path).read_text(encoding="utf-8")
        assert "generator.head must be" in text, path
        assert "regress" in text and "flow" in text, path
