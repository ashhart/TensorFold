#!/usr/bin/env python3
"""Scan TensorFold public files for local/private release leaks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Iterable


PUBLIC_ROOT_FILES = {"MANIFEST.in", "README.md", "pyproject.toml"}
PUBLIC_DIRS = ("src/smarttensor", "src/tensorfold", "docs", "examples", "tools")
PUBLIC_SUFFIXES = {".md", ".py", ".toml", ".json", ".sh", ".txt", ".yaml", ".yml"}
EXCLUDED_FILES = {
    "HANDOVER.md",
    "findings.md",
    "tools/check_public_scrub.py",
}
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "logs"}
EXCLUDED_PREFIXES = (
    "docs/superpowers/",
    "tests/SmartTensor-",
)


@dataclass(frozen=True)
class Leak:
    path: Path
    line: int
    label: str
    match: str


def find_public_leaks(root: Path, private_terms: Iterable[str] = ()) -> list[Leak]:
    root = root.resolve()
    leaks: list[Leak] = []
    patterns = _patterns(private_terms)
    for path in _iter_public_files(root):
        rel = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in patterns:
                match = pattern.search(line)
                if match:
                    leaks.append(
                        Leak(
                            path=rel,
                            line=line_number,
                            label=label,
                            match=match.group(0),
                        )
                    )
    return leaks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check TensorFold public files for local/private leakage.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--private-term",
        action="append",
        default=[],
        help=(
            "Additional private token to reject in public files. May be passed "
            "multiple times. TENSORFOLD_PRIVATE_SCRUB_TERMS also accepts a "
            "comma-separated list for private CI."
        ),
    )
    args = parser.parse_args(argv)

    leaks = find_public_leaks(args.root, private_terms=_configured_private_terms(args.private_term))
    if not leaks:
        print("public scrub: ok")
        return 0

    for leak in leaks:
        print(f"{leak.path}:{leak.line}: {leak.label}: {leak.match}", file=sys.stderr)
    return 1


def _iter_public_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for filename in PUBLIC_ROOT_FILES:
        path = root / filename
        if path.is_file():
            candidates.append(path)

    for dirname in PUBLIC_DIRS:
        base = root / dirname
        if not base.exists():
            continue
        if base.is_file():
            candidates.append(base)
            continue
        for path in base.rglob("*"):
            if path.is_file() and _is_public_path(root, path):
                candidates.append(path)

    return sorted(set(candidates))


def _is_public_path(root: Path, path: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if rel in EXCLUDED_FILES:
        return False
    if any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts):
        return False
    if any(rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False
    return path.suffix in PUBLIC_SUFFIXES


def _configured_private_terms(cli_terms: Iterable[str]) -> tuple[str, ...]:
    terms: list[str] = []
    env_terms = os.environ.get("TENSORFOLD_PRIVATE_SCRUB_TERMS", "")
    for raw in [*cli_terms, *env_terms.split(",")]:
        term = raw.strip()
        if term:
            terms.append(term)
    return tuple(dict.fromkeys(terms))


def _patterns(private_terms: Iterable[str]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    patterns: list[tuple[str, re.Pattern[str]]] = [
        ("absolute-user-home-path", re.compile(r"/Users/[A-Za-z0-9._-]+")),
        ("local-model-cache-path", re.compile(r"\." + "omlx")),
        ("email-address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ]
    private_pattern = _private_terms_pattern(private_terms)
    if private_pattern is not None:
        patterns.append(("configured-private-term", private_pattern))
    return tuple(patterns)


def _private_terms_pattern(private_terms: Iterable[str]) -> re.Pattern[str] | None:
    escaped = [re.escape(term.strip()) for term in private_terms if term.strip()]
    if not escaped:
        return None
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.I)


if __name__ == "__main__":
    raise SystemExit(main())
