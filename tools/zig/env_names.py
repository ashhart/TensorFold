"""Every environment variable name the engine's source spells, checked against docs/environment-names.md."""
import re
import sys
from pathlib import Path

from lean_check import ROOT, TEST_BLOCK, test_only

DOC = "docs/environment-names.md"
# Test data that no release binary imports: only environment_test.zig reads it.
TEST_DATA = ("zig/src/platform/environment_valid_cases.zig",)
# A whole string literal shaped like a variable name: capitals and digits with at least one underscore.
NAME = re.compile(r'"([A-Z][A-Z0-9]*_[A-Z0-9_]*[A-Z0-9])"')
# profile.zig's switches are enum fields whose tag names are the variables it reads.
SWITCHES = re.compile(r"pub const Switch = enum \{\n(.*?)\n\};", re.S)
ROW = re.compile(r"^\| `([A-Z][A-Z0-9_]*)` \| ([a-z, -]+) \|$")
TAGS = ("writes", "logs", "loads", "child", "other", "untagged", "not-env")
# The doc's two tables: the variables, then the names that only look like one.
TABLES = ("## Variables", "## Names that are not variables")


def source_names(root: Path = ROOT) -> dict[str, str]:
    """Each name a non-test engine source spells outside comments and test blocks, with where it first appears."""
    found: dict[str, str] = {}
    for path in sorted((root / "zig/src").rglob("*.zig")):
        rel = path.relative_to(root).as_posix()
        if test_only(rel) or rel in TEST_DATA:
            continue
        test_indent = None
        for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines()):
            if test_indent is None and (block := TEST_BLOCK.match(line)):
                test_indent = block.group(1)
            if test_indent is not None:
                test_indent = None if line.rstrip() == test_indent + "}" else test_indent
                continue
            if line.lstrip().startswith("//"):
                continue
            for name in NAME.findall(line):
                found.setdefault(name, f"{rel}:{i + 1}")
    profile = root / "zig/src/core/profile.zig"
    if profile.is_file() and (m := SWITCHES.search(profile.read_text(encoding="utf-8"))):
        for name in re.findall(r"^\s*([A-Z][A-Z0-9_]*),", m.group(1), re.M):
            found.setdefault(name, "zig/src/core/profile.zig")
    return found


def listed(text: str) -> dict[str, str]:
    """The doc's rows: name to its tags."""
    return {m.group(1): m.group(2) for line in text.splitlines() if (m := ROW.match(line))}


def problems(names: dict[str, str], rows: dict[str, str]) -> list[str]:
    """Names the source spells and the doc lacks, rows no source spells, and unknown tags."""
    found = [f"{at}: {n} has no row in {DOC} (run with --write)" for n, at in sorted(names.items()) if n not in rows]
    found += [f"{DOC}: {n} is listed but no engine source spells it" for n in sorted(rows) if n not in names]
    found += [f"{DOC}: {n} has unknown tag {t!r}" for n, tags in sorted(rows.items())
              for t in tags.split(", ") if t not in TAGS]
    return found


def written(text: str, names: dict[str, str]) -> str:
    """The doc with both tables rebuilt: kept rows keep their tags, new names come in untagged, gone names leave."""
    rows = listed(text)
    tagged = {n: rows.get(n, "untagged") for n in names}
    head = text.split("\n" + TABLES[0] + "\n", 1)[0]
    out = [head.rstrip("\n"), ""]
    for title, keep in zip(TABLES, (lambda t: t != "not-env", lambda t: t == "not-env")):
        out += [title, "", "| Name | Tags |", "| --- | --- |"]
        out += [f"| `{n}` | {t} |" for n, t in sorted(tagged.items()) if keep(t)]
        out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    doc, names = ROOT / DOC, source_names()
    if argv == ["--write"]:
        doc.write_text(written(doc.read_text(encoding="utf-8"), names), encoding="utf-8")
    found = problems(names, listed(doc.read_text(encoding="utf-8")))
    for f in found:
        print(f)
    print(f"{len(names)} names in source, {len(found)} problems")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
