"""Unpack a records archive written by scripts/pack_records.sh into results/ (T4.11, SPEC 7).

Usage (local, repo root, after `bash scripts/pull_results.sh _records`):
    python scripts/unpack_records.py results/_records/<UTC date>.tgz [--results DIR]

Order of checks, all before anything is written:
1. the `<archive>.md5` next to the archive exists and names it;
2. the archive's md5 matches;
3. every member is a record: `<run_id>/{config.yaml, git_sha.txt, metrics.jsonl, DONE}` or a regular
   file under `<run_id>/eval/` or `<run_id>/samples/`; no absolute path, `..`, link, device,
   `ckpt/`, `*.tmp` or repeated member (tarfile's `data_filter` runs on top of these checks);
4. no target in results/ differs from the archive.
A file that already exists with the same bytes is left alone, so unpacking twice changes nothing.
If any target differs, the differing files are listed and nothing is written (exit 2). New files
keep the archive's modification time, so a later `pull_results.sh` rsync sees them as up to date.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MD5_LINE = re.compile(r"^([0-9a-f]{32}) [ *](.+)$")
TOP_FILES = frozenset({"config.yaml", "git_sha.txt", "metrics.jsonl", "DONE"})
TOP_DIRS = frozenset({"eval", "samples"})
MAX_LISTED = 20


class RecordsError(Exception):
    """An archive that must not be unpacked (the message says why)."""


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_md5(archive: Path) -> str:
    """The md5 recorded next to the archive by pack_records.sh (`<hex>  <name>`)."""
    md5_path = archive.with_name(archive.name + ".md5")
    if not md5_path.is_file():
        raise RecordsError(
            f"{md5_path} is missing; pack_records.sh writes it next to the archive, copy both"
        )
    lines = md5_path.read_text(encoding="ascii").splitlines()
    match = MD5_LINE.match(lines[0]) if len(lines) == 1 else None
    if match is None:
        raise RecordsError(f"{md5_path} is not one md5sum line ('<md5>  {archive.name}')")
    if match.group(2) != archive.name:
        raise RecordsError(
            f"{md5_path} is for {match.group(2)!r}, not {archive.name!r} (was the archive renamed?)"
        )
    return match.group(1)


def record_path(member: tarfile.TarInfo, results: Path) -> PurePosixPath | None:
    """Check one member; return its path relative to results/ for a file, None for a directory.

    Raises RecordsError for anything that is not a record of a run.
    """
    name = member.name
    raw = name.rstrip("/").split("/")
    if name.startswith("/") or "\\" in name or any(part in ("", ".", "..") for part in raw):
        raise RecordsError(f"unsafe member path {name!r}")
    if not RUN_ID.match(raw[0]):
        raise RecordsError(f"member {name!r}: {raw[0]!r} is not a run id")
    if member.isdir():
        if len(raw) == 1 or raw[1] in TOP_DIRS:
            return None
        raise RecordsError(f"member {name!r} is a directory that holds no records")
    if not member.isfile():
        raise RecordsError(f"member {name!r} is not a regular file (links and devices are refused)")
    is_record = (len(raw) == 2 and raw[1] in TOP_FILES) or (len(raw) >= 3 and raw[1] in TOP_DIRS)
    if not is_record:
        raise RecordsError(
            f"member {name!r} is not a record (only config.yaml, git_sha.txt, metrics.jsonl, DONE, "
            "eval/ and samples/; never ckpt/)"
        )
    if name.endswith(".tmp"):
        raise RecordsError(f"member {name!r} is a half-written *.tmp file")
    try:
        tarfile.data_filter(member, str(results))
    except tarfile.FilterError as error:
        raise RecordsError(f"member {name!r} refused by tarfile's data filter: {error}") from error
    return PurePosixPath(*raw)


def blocking_parent(target: Path, results: Path) -> Path | None:
    """A path between results/ and target that exists but is not a directory, if any."""
    for parent in reversed(target.relative_to(results).parents):
        path = results / parent
        if path != results and path.exists() and not path.is_dir():
            return path
    return None


def plan(
    tar: tarfile.TarFile, results: Path
) -> tuple[list[tuple[tarfile.TarInfo, Path]], list[str], list[str]]:
    """Return (new files to write, unchanged paths, conflicts) without writing anything."""
    new: list[tuple[tarfile.TarInfo, Path]] = []
    unchanged: list[str] = []
    conflicts: list[str] = []
    seen: set[str] = set()
    for member in tar.getmembers():
        rel = record_path(member, results)
        if rel is None:
            continue
        if str(rel) in seen:
            raise RecordsError(f"member {member.name!r} appears twice in the archive")
        seen.add(str(rel))
        target = results.joinpath(*rel.parts)
        handle = tar.extractfile(member)
        if handle is None:
            raise RecordsError(f"member {member.name!r} has no data")
        data = handle.read()
        blocker = blocking_parent(target, results)
        if blocker is not None:
            conflicts.append(f"{rel} ({blocker} is a file, not a directory)")
        elif target.is_symlink() or (target.exists() and not target.is_file()):
            conflicts.append(f"{rel} (exists and is not a regular file)")
        elif target.exists():
            if target.read_bytes() == data:
                unchanged.append(str(rel))
            else:
                conflicts.append(str(rel))
        else:
            new.append((member, target))
    return new, unchanged, conflicts


def write_new(tar: tarfile.TarFile, new: list[tuple[tarfile.TarInfo, Path]]) -> None:
    """Write each new file atomically (temp file, then rename) with the archive's mtime."""
    for member, target in new:
        handle = tar.extractfile(member)
        assert handle is not None  # checked in plan()
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".unpack.tmp")
        tmp.write_bytes(handle.read())
        os.replace(tmp, target)
        os.utime(target, (member.mtime, member.mtime))


def unpack(archive: Path, results: Path) -> int:
    """Unpack one archive into results/; return the exit code (0 done, 2 refused)."""
    if not results.is_dir():
        raise RecordsError(f"results directory {results} does not exist")
    if not archive.is_file():
        raise RecordsError(f"archive {archive} does not exist")
    want = expected_md5(archive)
    got = file_md5(archive)
    if got != want:
        raise RecordsError(
            f"md5 mismatch for {archive.name}: the file has {got}, its .md5 says {want}; "
            "the copy is damaged, copy it again"
        )
    with tarfile.open(archive, "r:gz") as tar:
        new, unchanged, conflicts = plan(tar, results)
        if conflicts:
            print(
                f"REFUSED: {len(conflicts)} file(s) in {results} differ from {archive.name}; "
                "nothing was written:",
                file=sys.stderr,
            )
            for line in conflicts[:MAX_LISTED]:
                print(f"  {line}", file=sys.stderr)
            if len(conflicts) > MAX_LISTED:
                print(f"  ... and {len(conflicts) - MAX_LISTED} more", file=sys.stderr)
            print(
                "Compare them, move the local copies aside, then run this again "
                "(a pack made while a run was still training differs in metrics.jsonl).",
                file=sys.stderr,
            )
            return 2
        write_new(tar, new)
    runs = sorted({path.split("/")[0] for path in unchanged} | {
        target.relative_to(results).parts[0] for _, target in new
    })
    print(f"md5 ok: {got}  {archive.name}")
    print(f"unpacked into {results}: {len(new)} new, {len(unchanged)} unchanged")
    print(f"runs ({len(runs)}): {' '.join(runs)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archive", type=Path, help="results/_records/<UTC date>.tgz")
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results", help="target results directory"
    )
    args = parser.parse_args(argv)
    try:
        return unpack(args.archive, args.results)
    except (RecordsError, tarfile.TarError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
