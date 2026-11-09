"""Configuration dataclasses, YAML loader, serializer, and CLI override parser."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints
import yaml


@dataclass
class TokenizerConfig:
    d: int = 64
    K: int = 4
    quantizer: str = "vq"  # "vq" | "noisy"
    V_k: int = 64
    d_code: int = 16
    d_k: int = 8
    sigma_k: float = 0.4
    beta: float = 0.25
    ema_decay: float = 0.99
    dead_code_reinit_steps: int = 200
    dead_code_threshold: float = 1e-4
    encoder_blocks: int = 0
    decoder_blocks: int = 1
    decoder_heads: int = 4
    p_drop: float = 0.5


@dataclass
class CorruptionConfig:
    mode: str = "none"  # "none" | "swap" | "noise"
    p_first: float = 0.8
    p_last: float = 0.2
    s_first: float = 0.5
    s_last: float = 0.1


@dataclass
class GeneratorTrainConfig:
    """Generator training defaults (SPEC 5.7)."""

    lr: float = 3e-4
    min_lr: float = 3e-5
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000
    grad_clip: float = 1.0
    batch_size: int = 1024
    # T5.5a F13: split each batch into this many equal micro-batches, average
    # the losses, then take one optimizer step. VICReg statistics are computed
    # per micro-batch when this is > 1.
    grad_accum: int = 1
    eval_every: int = 500
    save_every: int = 5000
    save_milestones: bool = True
    milestones: list[float] = field(
        default_factory=lambda: [0.10, 0.25, 0.50, 1.0]
    )


@dataclass
class GeneratorConfig:
    H: int = 128
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    head: str = "ce"  # "ce" | "regress" | "flow"
    flow_steps: int = 8
    n_rep: int = 4
    renorm_pred: bool = True
    add_channel_noise: bool = True
    regress_cosine: bool = False
    alpha: float = 0.0
    corrupt: CorruptionConfig = field(default_factory=CorruptionConfig)
    train: GeneratorTrainConfig = field(default_factory=GeneratorTrainConfig)


@dataclass
class JEPAConfig:
    enabled: bool = False
    vicreg: bool = True
    decoder_through_E: bool = True
    ema: bool = True
    target_chain: str = "online"  # "online" | "ema_clean"
    lambda_var: float = 1.0
    lambda_cov: float = 0.04
    lambda_pred: float = 1.0
    gamma: float = 1.0
    m_start: float = 0.996
    m_end: float = 1.0


@dataclass
class TrainConfig:
    lr: float = 5e-4
    min_lr: float = 5e-5
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 1000
    grad_clip: float = 1.0
    batch_size: int = 16
    eval_every: int = 50
    save_every: int = 500
    allow_synthetic_data: bool = False


@dataclass
class EvalConfig:
    n_samples: int = 16
    n_seeds: int = 2
    gpt2_model_path: str = "models/gpt2-large"


@dataclass
class Config:
    L: int = 32
    V: int = 64
    seed: int = 42
    device: str = "cpu"
    compile: bool = False
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    jepa: JEPAConfig = field(default_factory=JEPAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _parse_override_value(val_str: str) -> Any:
    """Safely parse override string value into bool, int, float, list, or str."""
    if val_str.lower() == "true":
        return True
    if val_str.lower() == "false":
        return False
    if val_str == "null":
        return None
    try:
        return int(val_str)
    except ValueError:
        pass
    try:
        return float(val_str)
    except ValueError:
        pass
    if (val_str.startswith("[") and val_str.endswith("]")) or (
        val_str.startswith("{") and val_str.endswith("}")
    ):
        try:
            return yaml.safe_load(val_str)
        except Exception:
            pass
    return val_str


def apply_overrides(cfg_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply list of 'path.key=val' overrides to a config dictionary."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format (must be key=value): {item}")
        k_str, v_str = item.split("=", 1)
        keys = k_str.strip().split(".")
        val = _parse_override_value(v_str.strip())

        curr = cfg_dict
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], dict):
                curr[k] = {}
            curr = curr[k]
        curr[keys[-1]] = val
    return cfg_dict


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively instantiate dataclass from nested dictionary."""
    if not is_dataclass(cls):
        return data
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) for {cls.__name__}: {', '.join(unknown)}"
        )
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            val = data[f.name]
            f_type = hints.get(f.name, f.type)
            if is_dataclass(f_type) and isinstance(val, dict):
                kwargs[f.name] = _dict_to_dataclass(f_type, val)
            else:
                kwargs[f.name] = val
    return cls(**kwargs)


def config_from_dict(data: dict[str, Any]) -> Config:
    """Build a Config from a plain dictionary (for example a checkpoint's)."""
    return _dict_to_dataclass(Config, data)


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load Config from a YAML file with optional CLI overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if loaded:
                data = loaded

    if overrides:
        data = apply_overrides(data, overrides)

    return _dict_to_dataclass(Config, data)


def dump_config(cfg: Config) -> str:
    """Serialize Config dataclass to YAML string."""
    return yaml.safe_dump(asdict(cfg), sort_keys=False)


def save_config(cfg: Config, path: str | Path) -> None:
    """Save Config dataclass to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(dump_config(cfg))


def parse_cli_args(args: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    """Parse --config and --set arguments and load the resulting Config."""
    parser = argparse.ArgumentParser(description="jncm config parser")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Config overrides, e.g. --set tokenizer.quantizer=noisy",
    )
    parsed, _ = parser.parse_known_args(args)
    cfg = load_config(parsed.config, parsed.overrides)
    return cfg, parsed