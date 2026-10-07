"""The Zig engine's lean rule: no file over 600 lines, every comment and docstring one line of at most 120 columns."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREES = ("zig", "tools/zig")
MAX_LINES = 600
MAX_COLUMNS = 120
COMMENT = {".zig": re.compile(r"^\s*//"), ".metal": re.compile(r"^\s*//"), ".cu": re.compile(r"^\s*//"),
           ".cuh": re.compile(r"^\s*//"), ".h": re.compile(r"^\s*//"), ".py": re.compile(r"^\s*#"),
           ".sh": re.compile(r"^\s*#(?!!)")}
BLOCK = re.compile(r"/\*")
DOCSTRING = re.compile(r'^\s*(?:[rubf]{0,2})("""|\'\'\')')
GENERATED = ("unicode_data.zig",)
HEX_METAL = re.compile(r"_[0-9a-f]{16}\.metal$")
# Generator outputs are skipped entirely. unicode_data.zig stays exempt from the length check only.
GENERATOR_OUTPUT = (
    "zig/src/families/flashnext/roles_gen.zig",
    "zig/kernels/metal/flashnext/sources_gen.zig",
)


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
    lines = path.read_text(errors="replace").splitlines()
    rel, found = path.relative_to(ROOT), []
    if len(lines) > MAX_LINES and path.name not in GENERATED:
        found.append(f"{rel}: {len(lines)} lines (split past {MAX_LINES})")
    comment = COMMENT[path.suffix]
    before, doc_quote = "", None
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
