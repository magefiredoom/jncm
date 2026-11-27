"""T4.11: scripts/pack_records.sh and scripts/unpack_records.py (SPEC 7, v1.7 storage).

The pack holds the small records of each run (config.yaml, git_sha.txt, metrics.jsonl, DONE,
eval/, samples/), never ckpt/ and never a half-written *.tmp; the unpack checks the md5, extracts
into results/ and refuses, writing nothing, when an existing file differs or a member is unsafe.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "scripts" / "pack_records.sh"
UNPACK = ROOT / "scripts" / "unpack_records.py"


def _working_bash() -> str | None:
    candidates = [shutil.which("bash"), shutil.which("bash.exe")]
    if os.name == "nt":
        candidates += [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
        ]
    for candidate in candidates:
        if not candidate:
            continue
        probe = subprocess.run([candidate, "-c", "echo ok"], capture_output=True, text=True)
        if probe.returncode == 0 and "ok" in probe.stdout:
            return candidate
    return None


BASH = _working_bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no working bash on this host")


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _make_run(
    results: Path, run: str, *, done: bool = True, evals: bool = True, samples: bool = True
) -> dict[str, bytes]:
    """Write a fake run under results/<run>; return the record files (path relative to results/)."""
    files: dict[str, bytes] = {
        "config.yaml": b"run_id: " + run.encode() + b"\r\nseed: 0\n",  # CRLF kept as is
        "git_sha.txt": b"0123abcd\n",
        "metrics.jsonl": b'{"step": 1, "loss": 3.5}\n{"step": 2, "loss": 3.1}\n',
    }
    if done:
        files["DONE"] = b'{"step": 2, "sec_per_step": 1.0}\n'
    if evals:
        files["eval/step_000002.json"] = b'{"gen_ppl": 12.5}\n'
        files["eval/step_000002.seed_0.json"] = b'{"gen_ppl": 12.4}\n'
    if samples:
        files["samples/step_000002.txt"] = bytes(range(256))  # not UTF-8
    for rel, data in files.items():
        path = results / run / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    junk = {"ckpt/latest.pt": b"\x00" * 64, "ckpt/milestone_100.pt": b"\x01" * 64}
    if evals:
        junk["eval/step_000003.seed_0.json.tmp"] = b"half"
    for rel, data in junk.items():
        path = results / run / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {f"{run}/{rel}": data for rel, data in files.items()}


def _pack(cwd: Path, *runs: str) -> subprocess.CompletedProcess:
    assert BASH is not None
    return subprocess.run(
        [BASH, str(PACK), *runs], cwd=cwd, capture_output=True, text=True
    )


def _unpack(tgz: Path, results: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(UNPACK), str(tgz), "--results", str(results)],
        capture_output=True,
        text=True,
    )


def _archives(results: Path) -> list[Path]:
    return sorted((results / "_records").glob("*.tgz")) if (results / "_records").is_dir() else []


def _files_under(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _packed(tmp_path: Path, *runs: str) -> tuple[Path, dict[str, bytes]]:
    """Pack fake runs in tmp_path/src; return the archive and the expected records."""
    src = tmp_path / "src"
    expected: dict[str, bytes] = {}
    for run in runs:
        expected.update(_make_run(src / "results", run))
    proc = _pack(src, *runs)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (tgz,) = _archives(src / "results")
    return tgz, expected


def _write_archive(path: Path, members: list[tarfile.TarInfo], data: bytes = b"x\n") -> None:
    """Write a .tgz from hand-made members plus a correct .md5 next to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tar:
        for info in members:
            if info.isfile():
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.addfile(info)
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".md5").write_text(f"{digest}  {path.name}\n", encoding="ascii")


# ---------------------------------------------------------------- pack_records.sh


@needs_bash
def test_pack_writes_a_dated_archive_with_md5_and_never_ckpt(tmp_path: Path) -> None:
    before = _utc_today()
    tgz, expected = _packed(tmp_path, "gen_a", "tok_b")
    after = _utc_today()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}\.tgz", tgz.name)
    assert tgz.stem in {before, after}
    md5_line = tgz.with_name(tgz.name + ".md5").read_text(encoding="ascii")
    # md5sum line: text mode ("  ") on Linux, binary mode (" *") under Git Bash; both are valid.
    digest = hashlib.md5(tgz.read_bytes()).hexdigest()
    assert md5_line in (f"{digest}  {tgz.name}\n", f"{digest} *{tgz.name}\n")
    with tarfile.open(tgz, "r:gz") as tar:
        members = tar.getmembers()
    names = {m.name for m in members if m.isfile()}
    assert names == set(expected)
    assert not any("ckpt" in m.name or m.name.endswith(".tmp") for m in members)
    # No owner name in the archive (the FIR username must not leak into pasted listings).
    assert all(m.uname in ("", "root") and m.uid == 0 and m.gid == 0 for m in members)
    assert not (tgz.parent / (tgz.name + ".tmp")).exists()


@needs_bash
def test_pack_reports_absent_optional_parts(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _make_run(results, "smoke_x", done=False, evals=False, samples=False)
    proc = _pack(tmp_path, "smoke_x")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if "smoke_x" in ln and "absent" in ln)
    assert "DONE" in line and "eval/" in line and "samples/" in line
    (tgz,) = _archives(results)
    with tarfile.open(tgz, "r:gz") as tar:
        assert sorted(tar.getnames()) == [
            "smoke_x/config.yaml",
            "smoke_x/git_sha.txt",
            "smoke_x/metrics.jsonl",
        ]


@needs_bash
@pytest.mark.parametrize(
    "runs, message",
    [
        ([], "usage"),
        (["../gen_a"], "bad run id"),
        (["_records"], "bad run id"),
        (["no_such_run"], "results/no_such_run"),
        (["gen_a", "gen_a"], "twice"),
        (["no_config"], "config.yaml"),
    ],
)
def test_pack_refuses_bad_input_and_writes_nothing(
    tmp_path: Path, runs: list[str], message: str
) -> None:
    results = tmp_path / "results"
    _make_run(results, "gen_a")
    _make_run(results, "no_config")
    (results / "no_config" / "config.yaml").unlink()
    proc = _pack(tmp_path, *runs)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert message in proc.stderr
    assert _archives(results) == []


@needs_bash
def test_pack_refuses_outside_the_repo_root(tmp_path: Path) -> None:
    proc = _pack(tmp_path, "gen_a")
    assert proc.returncode == 2
    assert "results/" in proc.stderr


@needs_bash
def test_pack_never_overwrites_an_archive_of_the_same_day(tmp_path: Path) -> None:
    tgz, _ = _packed(tmp_path, "gen_a")
    first = tgz.read_bytes()
    md5_first = tgz.with_name(tgz.name + ".md5").read_bytes()
    proc = _pack(tmp_path / "src", "gen_a")
    assert proc.returncode == 2
    assert "exists" in proc.stderr
    assert tgz.read_bytes() == first
    assert tgz.with_name(tgz.name + ".md5").read_bytes() == md5_first


# ---------------------------------------------------------------- unpack_records.py


@needs_bash
def test_round_trip_reproduces_the_records_byte_for_byte(tmp_path: Path) -> None:
    tgz, expected = _packed(tmp_path, "gen_a", "tok_b")
    dest = tmp_path / "dest" / "results"
    dest.mkdir(parents=True)
    proc = _unpack(tgz, dest)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _files_under(dest) == expected  # byte for byte; no ckpt/, no *.tmp
    assert re.search(rf"\b{len(expected)} new\b", proc.stdout)
    src_results = tmp_path / "src" / "results"
    for rel in expected:
        assert abs((dest / rel).stat().st_mtime - (src_results / rel).stat().st_mtime) < 2.0


@needs_bash
def test_unpacking_twice_changes_nothing(tmp_path: Path) -> None:
    tgz, expected = _packed(tmp_path, "gen_a")
    dest = tmp_path / "dest"
    dest.mkdir()
    assert _unpack(tgz, dest).returncode == 0
    proc = _unpack(tgz, dest)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert re.search(r"\b0 new\b", proc.stdout)
    assert re.search(rf"\b{len(expected)} unchanged\b", proc.stdout)
    assert _files_under(dest) == expected


@needs_bash
def test_unpack_refuses_a_differing_file_and_writes_nothing(tmp_path: Path) -> None:
    tgz, _ = _packed(tmp_path, "gen_a")
    dest = tmp_path / "dest"
    planted = dest / "gen_a" / "metrics.jsonl"
    planted.parent.mkdir(parents=True)
    planted.write_bytes(b'{"step": 1, "loss": 3.5}\n')  # an older, shorter copy
    proc = _unpack(tgz, dest)
    assert proc.returncode == 2
    assert "gen_a/metrics.jsonl" in proc.stdout + proc.stderr
    assert "nothing was written" in (proc.stdout + proc.stderr).lower()
    assert _files_under(dest) == {"gen_a/metrics.jsonl": b'{"step": 1, "loss": 3.5}\n'}


@needs_bash
def test_unpack_refuses_a_damaged_archive(tmp_path: Path) -> None:
    tgz, _ = _packed(tmp_path, "gen_a")
    data = bytearray(tgz.read_bytes())
    data[len(data) // 2] ^= 0xFF
    tgz.write_bytes(bytes(data))
    dest = tmp_path / "dest"
    dest.mkdir()
    proc = _unpack(tgz, dest)
    assert proc.returncode == 2
    assert "md5" in proc.stderr.lower()
    assert _files_under(dest) == {}


@needs_bash
def test_unpack_refuses_without_its_md5_file(tmp_path: Path) -> None:
    tgz, _ = _packed(tmp_path, "gen_a")
    tgz.with_name(tgz.name + ".md5").unlink()
    dest = tmp_path / "dest"
    dest.mkdir()
    proc = _unpack(tgz, dest)
    assert proc.returncode == 2
    assert ".md5" in proc.stderr
    assert _files_under(dest) == {}


def _file(name: str) -> tarfile.TarInfo:
    return tarfile.TarInfo(name)


def _link(name: str, target: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info


@pytest.mark.parametrize(
    "bad",
    [
        _file("../evil.txt"),
        _file("/abs/evil.txt"),
        _file("gen_a/../../evil.txt"),
        _file("gen_a/ckpt/latest.pt"),
        _file("gen_a/eval/step_000003.seed_0.json.tmp"),
        _file("gen_a/notes.txt"),
        _file("_records/2026-01-01.tgz"),
        _file("gen_a/config.yaml"),  # a second copy of a member
        _link("gen_a/eval/step_000002.json", "../../../evil.txt"),
    ],
    ids=["parent", "absolute", "inner-parent", "ckpt", "tmp", "not-a-record", "records-dir",
         "duplicate", "symlink"],
)
def test_unpack_refuses_unsafe_or_non_record_members(tmp_path: Path, bad: tarfile.TarInfo) -> None:
    tgz = tmp_path / "in" / "2026-01-01.tgz"
    _write_archive(tgz, [_file("gen_a/config.yaml"), _file("gen_a/metrics.jsonl"), bad])
    dest = tmp_path / "dest" / "results"
    dest.mkdir(parents=True)
    proc = _unpack(tgz, dest)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert bad.name in proc.stderr
    assert _files_under(tmp_path / "dest") == {}


def test_unpack_refuses_a_missing_results_dir(tmp_path: Path) -> None:
    tgz = tmp_path / "2026-01-01.tgz"
    _write_archive(tgz, [_file("gen_a/config.yaml")])
    proc = _unpack(tgz, tmp_path / "no_results_here")
    assert proc.returncode == 2
    assert "no_results_here" in proc.stderr
    assert not (tmp_path / "no_results_here").exists()
