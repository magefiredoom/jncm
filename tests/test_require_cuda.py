"""A Slurm GPU job must not train on the CPU silently (JNCM_REQUIRE_CUDA).

2026-11-20: the first canary pilot logged 'CUDA driver initialization failed', the
trainer fell back to the CPU without a word, grew to 32 GB and was OOM-killed four
minutes later on an H100 allocation. The CPU tests rely on the fallback (configs ask
for cuda, the host has none), so it stays the default; the Slurm templates opt in to
the strict behaviour with JNCM_REQUIRE_CUDA=1.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jncm.config import load_config
from jncm.utils import REQUIRE_CUDA_ENV, resolve_device

TEMPLATES = [
    "scripts/slurm/tok.sbatch",
    "scripts/slurm/gen.sbatch",
    "scripts/slurm/jepa.sbatch",
    "scripts/slurm/eval.sbatch",
]


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv(REQUIRE_CUDA_ENV, raising=False)


def test_cuda_request_falls_back_to_cpu_by_default() -> None:
    assert resolve_device("cuda") == torch.device("cpu")


def test_cuda_request_is_honoured_when_a_gpu_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv(REQUIRE_CUDA_ENV, "1")

    assert resolve_device("cuda").type == "cuda"


def test_required_cuda_that_is_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REQUIRE_CUDA_ENV, "1")

    with pytest.raises(RuntimeError, match=REQUIRE_CUDA_ENV):
        resolve_device("cuda")


def test_only_the_value_one_turns_the_requirement_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REQUIRE_CUDA_ENV, "0")

    assert resolve_device("cuda") == torch.device("cpu")


def test_an_explicit_cpu_request_is_never_rewritten(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(REQUIRE_CUDA_ENV, "1")

    assert resolve_device("cpu") == torch.device("cpu")


def _cuda_cfg(config: str = "configs/smoke.yaml"):
    return load_config(config, overrides=["device=cuda"])


def test_generator_trainer_stops_before_touching_any_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jncm import train_generator

    monkeypatch.setenv(REQUIRE_CUDA_ENV, "1")

    with pytest.raises(RuntimeError, match=REQUIRE_CUDA_ENV):
        train_generator.train(
            _cuda_cfg(),
            tokenizer_ckpt=tmp_path / "missing.pt",
            max_steps=1,
            results_dir=tmp_path,
            data_dir=tmp_path / "no_data",
        )

    assert not any(tmp_path.iterdir())


def test_jepa_trainer_stops_before_touching_any_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jncm import train_jepa

    monkeypatch.setenv(REQUIRE_CUDA_ENV, "1")

    with pytest.raises(RuntimeError, match=REQUIRE_CUDA_ENV):
        train_jepa.train(
            _cuda_cfg("configs/jepa_smoke.yaml"),
            tokenizer_ckpt=tmp_path / "missing.pt",
            max_steps=1,
            results_dir=tmp_path,
            data_dir=tmp_path / "no_data",
        )

    assert not any(tmp_path.iterdir())


def test_tokenizer_trainer_stops_before_touching_any_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jncm import train_tokenizer

    monkeypatch.setenv(REQUIRE_CUDA_ENV, "1")

    with pytest.raises(RuntimeError, match=REQUIRE_CUDA_ENV):
        train_tokenizer.train(
            _cuda_cfg(),
            max_steps=1,
            results_dir=tmp_path,
            data_dir=tmp_path / "no_data",
        )

    assert not any(tmp_path.iterdir())


def test_eval_cli_stops_before_loading_gpt2(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_eval

    monkeypatch.setenv(REQUIRE_CUDA_ENV, "1")

    with pytest.raises(RuntimeError, match=REQUIRE_CUDA_ENV):
        run_eval.main(["--run_id", "no_such_run", "--gpt2", "models/no_such_gpt2"])


@pytest.mark.parametrize("script", TEMPLATES)
def test_every_gpu_template_requires_cuda(script: str) -> None:
    text = Path(script).read_text(encoding="utf-8")

    assert f"export {REQUIRE_CUDA_ENV}=1" in text, script


@pytest.mark.parametrize("script", TEMPLATES)
def test_every_gpu_template_loads_python_before_the_venv(script: str) -> None:
    """tok.sbatch always did and its jobs ran on cuda; the other three did not, and
    the first canary that skipped it logged 'CUDA driver initialization failed'."""
    lines = Path(script).read_text(encoding="utf-8").splitlines()
    load = [i for i, line in enumerate(lines) if line.strip() == "module load python/3.12"]
    activate = [i for i, line in enumerate(lines) if line.strip() == "source .venv/bin/activate"]

    assert len(load) == 1 and len(activate) == 1, (script, load, activate)
    assert load[0] < activate[0], script
