#!/usr/bin/env python3
"""Write every file block found in a saved model reply (markdown) to disk.

Expected block format (the Gemini prompt asks for exactly this; looser headers such as
'FILE: path' or '**FILE: path**' are also accepted):

    ### FILE: jncm/levels.py
    ````python
    <file content>
    ````

The fence may be 3+ backticks; a 4-backtick fence lets the content contain
3-backtick fences. Absolute paths and '..' are refused.

Usage:
    python extract_files.py reply.md            # write into current dir
    python extract_files.py reply.md --out repo # write into ./repo
    python extract_files.py reply.md --dry-run  # list only
Commit before running: existing files are overwritten (git is your undo).
"""
import argparse
import pathlib
import re
import sys

# Tolerant header: "### FILE: x", "FILE: x", "**FILE: x**", "FILE: `x`"; blank lines before the fence are OK.
PATTERN = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?FILE:[ \t]*(?P<path>[^\n]+?)[ \t]*(?:\*\*)?[ \t]*\n"
    r"(?:[ \t]*\n)*"
    r"(?P<fence>`{3,})[^\n]*\n"
    r"(?P<body>.*?)\n"
    r"(?P=fence)[ \t]*$",
    re.S | re.M,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reply", help="markdown file containing the model reply")
    ap.add_argument("--out", default=".", help="output root directory")
    ap.add_argument("--dry-run", action="store_true", help="list files without writing")
    args = ap.parse_args()

    text = pathlib.Path(args.reply).read_text(encoding="utf-8").replace("\r\n", "\n")
    out = pathlib.Path(args.out).resolve()
    count = 0
    for m in PATTERN.finditer(text):
        raw = m.group("path").strip().strip("`*'\" ")
        rel = pathlib.PurePosixPath(raw)
        if rel.is_absolute() or ".." in rel.parts or not rel.parts:
            print(f"SKIP unsafe path: {raw}")
            continue
        dest = out.joinpath(*rel.parts)
        body = m.group("body")
        verb = "would write" if args.dry_run else "write"
        print(f"{verb}: {dest.relative_to(out)}  ({len(body.splitlines())} lines)")
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(body + "\n", encoding="utf-8")
        count += 1
    print(f"{count} file(s)")
    if count == 0:
        print("No '### FILE: path' blocks found. Check the reply format.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
