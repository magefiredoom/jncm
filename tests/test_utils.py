"""Tests for seeding, JSONL logger, and bits_per_position calculations."""

import math
from pathlib import Path
import numpy as np
import pytest
import torch
from jncm.config import load_config
from jncm.utils import JSONLLogger, bits_per_position, seed_everything


def test_seed_everything_determinism() -> None:
    seed_everything(1234)
    t1 = torch.randn(10, 10)
    a1 = np.random.randn(10)

    seed_everything(1234)
    t2 = torch.randn(10, 10)
    a2 = np.random.randn(10)

    assert torch.equal(t1, t2)
    assert np.array_equal(a1, a2)


def test_jsonl_logger(tmp_path: Path) -> None:
    log_file = tmp_path / "metrics.jsonl"
    logger = JSONLLogger(log_file)

    entry1 = {
        "step": 1,
        "loss": torch.tensor(0.42),
        "lr": 0.001,
        "arr": np.array([1, 2, 3]),
    }
    entry2 = {
        "step": 2,
        "loss": 0.35,
        "acc": torch.tensor(0.99),
    }

    logger.log(entry1)
    logger.log(entry2)

    records = logger.read_all()
    assert len(records) == 2
    assert records[0]["step"] == 1
    assert math.isclose(records[0]["loss"], 0.42, rel_tol=1e-5)
    assert records[0]["arr"] == [1, 2, 3]
    assert records[1]["step"] == 2
    assert math.isclose(records[1]["acc"], 0.99, rel_tol=1e-5)


def test_bits_per_position_vq() -> None:
    # V_k = 4096 => log2(4096) = 12 bits
    bits = bits_per_position(quantizer="vq", V_k=4096)
    assert math.isclose(bits, 12.0, rel_tol=1e-6)

    # Smoke V_k = 64 => log2(64) = 6 bits
    bits_smoke = bits_per_position(quantizer="vq", V_k=64)
    assert math.isclose(bits_smoke, 6.0, rel_tol=1e-6)


def test_bits_per_position_noisy() -> None:
    # Dev: d_k = 8, sigma_k = 0.4 => (8/2) * log2(1 + 1/0.16) = 4 * log2(7.25) ~ 11.4319 bits
    bits = bits_per_position(quantizer="noisy", d_k=8, sigma_k=0.4)
    expected = 4.0 * math.log2(1.0 + 1.0 / 0.16)
    assert math.isclose(bits, expected, rel_tol=1e-6)
    assert math.isclose(bits, 11.43187, rel_tol=1e-4)


def test_bits_per_position_from_config() -> None:
    cfg = load_config(Path("configs/tinystories_dev.yaml"))
    assert math.isclose(bits_per_position(cfg), 12.0)

    cfg_noisy = load_config(
        Path("configs/tinystories_dev.yaml"),
        overrides=["tokenizer.quantizer=noisy"],
    )
    assert math.isclose(bits_per_position(cfg_noisy), 11.43187, rel_tol=1e-4)


def test_bits_per_position_invalid() -> None:
    with pytest.raises(ValueError):
        bits_per_position(quantizer="unknown")