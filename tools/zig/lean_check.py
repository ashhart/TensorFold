"""The Zig engine's lean rule: no file over 600 lines, every comment and docstring one line."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREES = ("zig", "tools/zig")
MAX_LINES = 600
COMMENT = {".zig": re.compile(r"^\s*//"), ".metal": re.compile(r"^\s*//"), ".cu": re.compile(r"^\s*//"),
           ".cuh": re.compile(r"^\s*//"), ".h": re.compile(r"^\s*//"), ".py": re.compile(r"^\s*#"),
           ".sh": re.compile(r"^\s*#(?!!)")}
BLOCK = re.compile(r"/\*")
DOCSTRING = re.compile(r'^\s*(?:[rubf]{0,2})("""|\'\'\')')
GENERATED = ("unicode_data.zig",)


def problems(path: Path) -> list[str]:
    """Every rule break in one file, as 'path:line: reason'."""
    lines = path.read_text(errors="replace").splitlines()
    rel, found = path.relative_to(ROOT), []
    if len(lines) > MAX_LINES and path.name not in GENERATED:
        found.append(f"{rel}: {len(lines)} lines (split past {MAX_LINES})")
    comment = COMMENT[path.suffix]
    before = ""
    for i, line in enumerate(lines):
        if comment.match(line) and i + 1 < len(lines) and comment.match(lines[i + 1]):
            found.append(f"{rel}:{i + 1}: multi-line comment")
        if path.suffix != ".py" and BLOCK.search(line):
            found.append(f"{rel}:{i + 1}: block comment")
        m = DOCSTRING.match(line) if path.suffix == ".py" else None
        # a docstring opens a module or follows a def/class line; other triple quotes are data
        if m and line.count(m.group(1)) == 1 and (not before or before.rstrip().endswith(":")):
            found.append(f"{rel}:{i + 1}: multi-line docstring")
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
