"""The GPT-2 weights are found, or reported missing, before anything slow starts.

2026-11-20 (canary 4742789): `train_generator --gen_ppl` built its scorer from the config's
relative `models/gpt2-large` and ignored $GPT2_LARGE_PATH, which `launch_phase.sh` and the
RUNBOOK export. The scorer also imported `transformers` first (about nine minutes from the
Lustre venv) and checked the path afterwards, so the missing weights surfaced at step 0,
after the whole startup and the first training step, as FileNotFoundError.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from jncm.eval import genppl
from jncm.eval.genppl import require_gpt2_weights, resolve_gpt2_path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GPT2_LARGE_PATH", raising=False)


def test_the_environment_variable_wins_over_the_configured_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    weights = tmp_path / "gpt2"
    weights.mkdir()
    monkeypatch.setenv("GPT2_LARGE_PATH", str(weights))

    assert resolve_gpt2_path("models/gpt2-large") == weights


def test_the_configured_path_is_used_without_the_variable(tmp_path: Path) -> None:
    weights = tmp_path / "gpt2"
    weights.mkdir()

    assert resolve_gpt2_path(str(weights)) == weights


def test_missing_weights_raise_at_once_and_name_both_sources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GPT2_LARGE_PATH", str(tmp_path / "nope"))

    with pytest.raises(FileNotFoundError, match="GPT2_LARGE_PATH"):
        resolve_gpt2_path("models/gpt2-large")


def test_require_gpt2_weights_returns_an_existing_path(tmp_path: Path) -> None:
    assert require_gpt2_weights(tmp_path) == tmp_path
    with pytest.raises(FileNotFoundError):
        require_gpt2_weights(tmp_path / "missing")


def test_scorer_checks_the_path_before_it_imports_transformers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The check used to follow the import; with a slow filesystem that is minutes."""
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    imported: list[str] = []
    real_import = __import__

    def spy(name, *args, **kwargs):
        if name == "transformers":
            imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", spy)

    with pytest.raises(FileNotFoundError):
        genppl.GPT2Scorer(model_path=tmp_path / "missing", device="cpu")._load()

    assert imported == []


def test_train_generator_stops_before_training_when_the_weights_are_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jncm import train_generator

    monkeypatch.setenv("GPT2_LARGE_PATH", str(tmp_path / "nope"))
    monkeypatch.setattr(train_generator, "build_decode_fn", lambda path: (lambda ids: ""))
    monkeypatch.setattr(
        train_generator,
        "train",
        lambda **kwargs: pytest.fail("the trainer started although the weights are missing"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_generator",
            "--config",
            "configs/smoke.yaml",
            "--tokenizer",
            str(tmp_path / "tok.pt"),
            "--gen_ppl",
            "--steps",
            "1",
        ],
    )

    with pytest.raises(FileNotFoundError, match="GPT2_LARGE_PATH"):
        train_generator.main()


def test_train_generator_hands_the_environment_path_to_the_scorer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from jncm import train_generator

    weights = tmp_path / "gpt2"
    weights.mkdir()
    seen: dict[str, object] = {}

    class Recorder:
        def __init__(self, model_path=None, device="cpu") -> None:
            seen["path"] = Path(model_path)

    monkeypatch.setenv("GPT2_LARGE_PATH", str(weights))
    monkeypatch.setattr(genppl, "GPT2Scorer", Recorder)
    monkeypatch.setattr(train_generator, "build_decode_fn", lambda path: (lambda ids: ""))
    monkeypatch.setattr(train_generator, "train", lambda **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_generator",
            "--config",
            "configs/smoke.yaml",
            "--tokenizer",
            str(tmp_path / "tok.pt"),
            "--gen_ppl",
            "--steps",
            "1",
        ],
    )

    train_generator.main()

    assert seen["path"] == weights
