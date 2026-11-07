#!/usr/bin/env python3
"""Copy a marked recipe pipe table into the receipt README; --check never writes."""

import argparse
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

START = "<!-- lfm3-evidence:start -->"
END = "<!-- lfm3-evidence:end -->"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = ROOT / "docs/recipes/lfm3-native.md"
DEFAULT_README = ROOT / "docs/receipts/lfm3/README.md"


def bounds(text):
    """Require one exact, standalone, ordered marker pair and retain all surrounding bytes as text."""
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("expected exactly one start and one end marker")
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == START]
    ends = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == END]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError("markers must be standalone lines in start/end order")
    return lines, starts[0], ends[0]


def cells(row):
    """Parse outer-pipe rows; only unescaped pipes separate cells, including inside Markdown code spans."""
    if not row.startswith("|") or not row.endswith("|"):
        raise ValueError("table rows require leading and trailing pipes without indentation")
    separators = []
    backslashes = 0
    for i, char in enumerate(row):
        if char == "|" and backslashes % 2 == 0:
            separators.append(i)
        backslashes = backslashes + 1 if char == "\\" else 0
    if len(separators) < 2 or separators[-1] != len(row) - 1:
        raise ValueError("table rows require an unescaped trailing pipe")
    return [row[left + 1:right].strip() for left, right in zip(separators, separators[1:])]


def table(source):
    """Accept one header, a :?---+:? delimiter row, and data rows of equal width; no multiline cells."""
    lines, start, end = bounds(source)
    body = "".join(lines[start + 1:end]).replace("\r\n", "\n").strip("\n")
    if "\r" in body:
        raise ValueError("bare carriage returns are unsupported in tables")
    rows = body.split("\n")
    if len(rows) < 3:
        raise ValueError("source block requires a header, delimiter and at least one data row")
    parsed = [cells(row) for row in rows]
    width = len(parsed[0])
    if any(len(row) != width for row in parsed):
        raise ValueError("table rows must have consistent column counts")
    if any(re.fullmatch(r":?-{3,}:?", cell) is None for cell in parsed[1]):
        raise ValueError("table delimiter cells require at least three hyphens and optional edge colons")
    return body


def render(source, target):
    """Replace only the target marker interior, using the start marker's LF or CRLF line ending."""
    body = table(source)
    lines, start, end = bounds(target)
    newline = "\r\n" if lines[start].endswith("\r\n") else "\n"
    return "".join(lines[:start + 1]) + body.replace("\n", newline) + newline + "".join(lines[end:])


def destination_mode(recipe, readme):
    """Refuse destination symlinks, nonregular inputs and source aliases, including hardlinks."""
    target_stat = readme.lstat()
    if stat.S_ISLNK(target_stat.st_mode):
        raise ValueError("destination must not be a symlink")
    if not stat.S_ISREG(target_stat.st_mode) or not stat.S_ISREG(recipe.stat().st_mode):
        raise ValueError("source and destination must be regular files")
    if os.path.samefile(recipe, readme):
        raise ValueError("source and destination must be different files")
    return stat.S_IMODE(target_stat.st_mode)


def atomic_write(path, content, mode):
    """Write a mode-preserving sibling, then replace atomically; remove the sibling on every failure."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--check", action="store_true", help="exit 1 if stale without writing")
    args = parser.parse_args(argv)
    try:
        mode = destination_mode(args.recipe, args.readme)
        source = args.recipe.read_bytes().decode("utf-8")
        old = args.readme.read_bytes().decode("utf-8")
        new = render(source, old)
        if new == old:
            return 0
        if args.check:
            print("generated evidence table is stale", file=sys.stderr)
            return 1
        atomic_write(args.readme, new, mode)
        return 0
    except (OSError, ValueError) as exc:
        print(f"evidence renderer: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
