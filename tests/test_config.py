"""Tests for configuration dataclasses, YAML loader, overrides, and round-tripping."""

from pathlib import Path
from jncm.config import (
    Config,
    apply_overrides,
    dump_config,
    load_config,
)


def test_config_defaults() -> None:
    cfg = Config()
    assert cfg.L == 32
    assert cfg.V == 64
    assert cfg.tokenizer.d == 64
    assert cfg.tokenizer.K == 4
    assert cfg.tokenizer.quantizer == "vq"
    assert cfg.generator.H == 128
    assert cfg.train.batch_size == 16


def test_config_smoke_yaml_loads() -> None:
    smoke_path = Path("configs/smoke.yaml")
    assert smoke_path.exists()
    cfg = load_config(smoke_path)
    assert cfg.L == 32
    assert cfg.V == 64
    assert cfg.tokenizer.K == 4
    assert cfg.tokenizer.quantizer == "vq"
    assert cfg.generator.H == 128


def test_config_tinystories_dev_yaml_loads() -> None:
    dev_path = Path("configs/tinystories_dev.yaml")
    assert dev_path.exists()
    cfg = load_config(dev_path)
    assert cfg.L == 128
    assert cfg.V == 8192
    assert cfg.tokenizer.K == 12
    assert cfg.tokenizer.d == 384
    assert cfg.generator.H == 768


def test_dev_configs_have_dense_milestones() -> None:
    """T5.5b F14: every 10% of the budget, for both dev configurations."""
    expected = [round(0.1 * index, 1) for index in range(1, 11)]
    for path in (
        Path("configs/tinystories_dev.yaml"),
        Path("configs/jepa_dev.yaml"),
    ):
        cfg = load_config(path)
        assert cfg.generator.train.milestones == expected, path


def test_config_yaml_round_trip(tmp_path: Path) -> None:
    cfg_orig = load_config(Path("configs/smoke.yaml"))
    yaml_text = dump_config(cfg_orig)

    tmp_file = tmp_path / "roundtrip.yaml"
    tmp_file.write_text(yaml_text, encoding="utf-8")

    cfg_loaded = load_config(tmp_file)
    assert cfg_loaded == cfg_orig


def test_config_overrides() -> None:
    overrides = [
        "tokenizer.quantizer=noisy",
        "train.lr=0.001",
        "train.batch_size=32",
        "generator.corrupt.mode=swap",
        "compile=true",
    ]
    cfg = load_config(Path("configs/smoke.yaml"), overrides=overrides)
    assert cfg.tokenizer.quantizer == "noisy"
    assert cfg.train.lr == 0.001
    assert cfg.train.batch_size == 32
    assert cfg.generator.corrupt.mode == "swap"
    assert cfg.compile is True


def test_apply_overrides_dict() -> None:
    d = {"a": {"b": 1, "c": "val"}, "flag": False}
    out = apply_overrides(d, ["a.b=42", "a.c=new_val", "flag=true", "a.sub.item=hello"])
    assert out["a"]["b"] == 42
    assert out["a"]["c"] == "new_val"
    assert out["flag"] is True
    assert out["a"]["sub"]["item"] == "hello"