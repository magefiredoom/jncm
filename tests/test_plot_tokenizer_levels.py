"""T5.5c: scripts/plot_tokenizer_levels.py (Figure C) on synthetic metrics."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.plot_tokenizer_levels import last_record, main


def _write(run: Path, records: list[dict]) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )


def _vq(step: int = 10000, acc: tuple[float, ...] = (0.1, 0.5, 0.99)) -> dict:
    return {
        "step": step,
        "acc_full": acc[-1],
        "acc_cum": list(acc),
        "codebook_usage": [0.2, 0.7, 0.9],
        "data_source": "real",
    }


def _noisy(step: int = 10000, acc: tuple[float, ...] = (0.2, 0.6, 0.999)) -> dict:
    return {
        "step": step,
        "acc_full": acc[-1],
        "acc_cum": list(acc),
        "eff_rank": [6.3, 6.9, 7.0],
        "data_source": "real",
    }


def _fixture(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    _write(results / "tok_vq_10k", [_vq(5000, (0.0, 0.0, 0.0)), _vq()])
    _write(results / "tok_noisy_10k", [_noisy()])
    return results


def test_uses_the_last_record_that_has_acc_full(tmp_path: Path) -> None:
    vq = tmp_path / "tok_vq_10k" / "metrics.jsonl"
    vq.parent.mkdir()
    vq.write_text(
        json.dumps(_vq(step=10, acc=(0.1, 0.2, 0.3)))
        + "\n"
        + json.dumps(_vq(step=20, acc=(0.4, 0.5, 0.6)))
        + "\n",
        encoding="utf-8",
    )
    noisy = tmp_path / "tok_noisy_10k" / "metrics.jsonl"
    noisy.parent.mkdir()
    noisy.write_text(
        json.dumps(_noisy(step=30)) + "\n" + json.dumps({"step": 31}) + "\n",
        encoding="utf-8",
    )

    assert last_record(vq)["step"] == 20
    assert last_record(noisy)["step"] == 30


def test_writes_the_png(tmp_path: Path) -> None:
    results = _fixture(tmp_path)
    out = tmp_path / "figures" / "fig_c_tokenizers.png"

    code = main(["--results", str(results), "--out", str(out)])

    assert code == 0
    assert out.exists() and out.stat().st_size > 1000


def test_table_numbers_come_from_the_records(tmp_path: Path, capsys) -> None:
    results = _fixture(tmp_path)

    code = main(["--results", str(results), "--out", str(tmp_path / "fig.png")])

    assert code == 0
    printed = capsys.readouterr().out
    header = (
        "| level | vq acc_cum | noisy acc_cum | vq codebook_usage | "
        "noisy eff_rank | noisy eff_rank/7 |"
    )
    assert header in printed
    assert "| 1 | 0.1000 | 0.2000 | 0.2000 | 6.3000 | 0.9000 |" in printed
    assert "| 3 | 0.9900 | 0.9990 | 0.9000 | 7.0000 | 1.0000 |" in printed
    assert printed.count("| 2 |") == 1


def test_missing_metrics_file_is_a_clear_error(tmp_path: Path, capsys) -> None:
    results = tmp_path / "results"
    _write(results / "tok_vq_10k", [_vq()])
    out = tmp_path / "fig.png"

    code = main(["--results", str(results), "--out", str(out)])

    assert code == 1
    assert not out.exists()
    error = capsys.readouterr().err
    assert "missing" in error
    assert "tok_noisy_10k" in error


def test_record_without_acc_full_is_an_error(tmp_path: Path, capsys) -> None:
    results = tmp_path / "results"
    _write(results / "tok_vq_10k", [{"step": 1, "train_loss": 1.0}])
    _write(results / "tok_noisy_10k", [_noisy()])

    code = main(["--results", str(results), "--out", str(tmp_path / "fig.png")])

    assert code == 1
    assert "no record with 'acc_full'" in capsys.readouterr().err


def test_mismatched_level_counts_are_an_error(tmp_path: Path, capsys) -> None:
    results = tmp_path / "results"
    _write(results / "tok_vq_10k", [_vq(acc=(0.1, 0.5, 0.99))])
    _write(results / "tok_noisy_10k", [_noisy(acc=(0.2, 0.6))])

    code = main(["--results", str(results), "--out", str(tmp_path / "fig.png")])

    assert code == 1
    assert "level counts differ" in capsys.readouterr().err
