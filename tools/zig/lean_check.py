"""The Zig engine's lean rule: 600-line files, one-line comments within 120 columns, no direct getenv of a setting."""
import functools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREES = ("zig", "tools/zig")
MAX_LINES = 600
MAX_COLUMNS = 120
COMMENT = {".cl": re.compile(r"^\s*//"), ".c": re.compile(r"^\s*//"), ".hip": re.compile(r"^\s*//"), ".zig": re.compile(r"^\s*//"), ".metal": re.compile(r"^\s*//"), ".cu": re.compile(r"^\s*//"),
           ".cuh": re.compile(r"^\s*//"), ".h": re.compile(r"^\s*//"), ".py": re.compile(r"^\s*#"),
           ".sh": re.compile(r"^\s*#(?!!)")}
BLOCK = re.compile(r"/\*")
DOCSTRING = re.compile(r'^\s*(?:[rubf]{0,2})("""|\'\'\')')
GENERATED = ("unicode_data.zig",)
HEX_METAL = re.compile(r"_[0-9a-f]{16}\.metal$")
# Generated outputs are skipped entirely. unicode_data.zig has only the length exemption.
GENERATOR_OUTPUT = (
    "zig/src/core/tokenizer/nfkd_data.zig",
    "zig/src/families/flashnext/roles_gen.zig",
    "zig/kernels/metal/flashnext/sources_gen.zig",
)
# A setting read by name instead of through platform/environment.zig, which refuses malformed text.
DIRECT_GETENV = re.compile(r'\bgetenv\(\s*"((?:TF|TENSORFOLD)_[A-Za-z0-9_]*)"')
# A getenv of a name that is not a literal: a new wrapper or a held name would hide its setting from DIRECT_GETENV.
HIDDEN_GETENV = re.compile(r'(?<!fn )\bgetenv\(\s*[^"\s)]')
# The parser, the registry's resolver and the platform shims read the names they are given.
NAME_READERS = ("zig/src/platform/environment.zig", "zig/src/core/profile.zig", "zig/src/platform/sys.zig",
                "zig/src/platform/win_c.zig")
# A `test` block's opening line; its closing brace sits alone at the same indent, as zig fmt writes it.
TEST_BLOCK = re.compile(r"^(\s*)test\b.*\{\s*$")
SWITCH_NAME = re.compile(r"`((?:TF|TENSORFOLD)_[A-Za-z0-9_]+)")


def lenient_switches(text: str) -> set[str]:
    """The names in the list under '## Existing lenient switches', which keep their own readers."""
    section = text.partition("\n## Existing lenient switches\n")[2].split("\n## ", 1)[0]
    return {n for line in section.splitlines() if line.startswith(("- ", "  ")) for n in SWITCH_NAME.findall(line)}


@functools.cache
def lenient() -> frozenset[str]:
    """docs/environment.md's lenient switches; a missing doc lists none rather than stopping the check."""
    doc = ROOT / "docs/environment.md"
    return frozenset(lenient_switches(doc.read_text(encoding="utf-8")) if doc.is_file() else ())


def test_only(rel: str) -> bool:
    """A Zig file that only tests: *_test.zig, test_*.zig or anything under zig/tests."""
    name = rel.rsplit("/", 1)[-1]
    return rel.startswith("zig/tests/") or name.endswith("_test.zig") or name.startswith("test_")


def generator_output(path: Path) -> bool:
    """A Flash Next file a generator writes. The generator script itself stays checked."""
    rel = path.relative_to(ROOT).as_posix()
    if rel in GENERATOR_OUTPUT:
        return True
    return rel.startswith("zig/kernels/metal/flashnext/") and HEX_METAL.search(path.name) is not None


def problems(path: Path) -> list[str]:
    """Every rule break in one file, as 'path:line: reason'."""
    if generator_output(path):
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rel, found = path.relative_to(ROOT), []
    if len(lines) > MAX_LINES and path.name not in GENERATED:
        found.append(f"{rel}: {len(lines)} lines (split past {MAX_LINES})")
    comment = COMMENT[path.suffix]
    settings = path.suffix == ".zig" and not test_only(rel.as_posix())
    names_given = rel.as_posix() in NAME_READERS
    before, doc_quote, test_indent = "", None, None
    for i, line in enumerate(lines):
        if doc_quote is not None:
            if len(line) > MAX_COLUMNS:
                found.append(f"{rel}:{i + 1}: docstring past {MAX_COLUMNS} columns")
            if doc_quote in line:
                doc_quote = None
            continue
        if comment.match(line) and i + 1 < len(lines) and comment.match(lines[i + 1]):
            found.append(f"{rel}:{i + 1}: multi-line comment")
        if comment.match(line) and len(line) > MAX_COLUMNS:
            found.append(f"{rel}:{i + 1}: comment past {MAX_COLUMNS} columns")
        if path.suffix != ".py" and BLOCK.search(line):
            found.append(f"{rel}:{i + 1}: block comment")
        if settings and test_indent is None and (block := TEST_BLOCK.match(line)):
            test_indent = block.group(1)
        if settings and test_indent is None and not comment.match(line):
            found += [f"{rel}:{i + 1}: direct getenv of {name} (read it through platform/environment.zig)"
                      for name in DIRECT_GETENV.findall(line) if name not in lenient()]
            if not names_given:
                found += [f"{rel}:{i + 1}: getenv of a name that is not a literal (read it through platform/environment.zig)"
                          for _ in HIDDEN_GETENV.findall(line)]
        elif test_indent is not None and line.rstrip() == test_indent + "}":
            test_indent = None
        m = DOCSTRING.match(line) if path.suffix == ".py" else None
        # a docstring opens a module or follows a def/class line; other triple quotes are data
        if m and (not before or before.rstrip().endswith(":")):
            quote = m.group(1)
            if len(line) > MAX_COLUMNS:
                found.append(f"{rel}:{i + 1}: docstring past {MAX_COLUMNS} columns")
            if line.count(quote) == 1:
                found.append(f"{rel}:{i + 1}: multi-line docstring")
                doc_quote = quote
        if line.strip() and not comment.match(line):
            before = line
    return found


def main() -> int:
    files = [p for t in TREES for p in sorted((ROOT / t).rglob("*")) if p.is_file() and p.suffix in COMMENT]
    found = [f for p in files for f in problems(p)]
    for f in found:
        print(f)
    print(f"{len(files)} files, {len(found)} problems")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
