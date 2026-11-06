"""Reject unlisted CUDA primitive copies outside common, including changed instructions and extra identical copies."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
ALLOW = Path(__file__).with_name("cuda_common_allowlist.json")
TOKEN = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\d+(?:\.\d*)?|[^\s]')
OPS = re.compile(r'\b(?:ldmatrix[.\w]*|mma\.sync[.\w]*|cp\.async[.\w]*|ld\.global\.nc[.\w]*|griddepcontrol[.\w]*)')
NAMES = re.compile(r'(?:warp_sum|block_sum\w*|fp4pair|fp8pair|e4m3f|unpack8|pack8)\Z')


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    fingerprint: str

    def key(self):
        """Line numbers are evidence, not the allowlist identity."""
        return self.path, self.kind, self.fingerprint


def tokens(text):
    """C/C++ tokens with comments discarded and literal contents kept as one token."""
    return [(m.group(), m.start()) for m in TOKEN.finditer(text) if not m.group().startswith(("//", "/*"))]


def closing(ts, start, left, right):
    """Find a balanced delimiter; strings never count as delimiters."""
    depth = 0
    for i in range(start, len(ts)):
        depth += (ts[i][0] == left) - (ts[i][0] == right)
        if depth == 0:
            return i
    raise ValueError("unclosed " + left)


def fingerprint(ts):
    """Pin tokens, operand constraints, volatile and literal contents, independently of comments and line positions."""
    return hashlib.sha256(json.dumps([v for v, _ in ts], separators=(",", ":")).encode()).hexdigest()


def scan(root):
    """List instruction atoms and named device definitions outside the maintained common headers."""
    out = []
    base = root / "zig/kernels/cuda"
    if not base.is_dir():
        raise ValueError("missing CUDA kernel tree")
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix not in (".cu", ".cuh", ".h") or p.is_relative_to(base / "common"):
            continue
        text = p.read_text()
        ts = tokens(text)
        rel = p.relative_to(root).as_posix()
        for i, (v, pos) in enumerate(ts):
            if v in ("asm", "__asm__"):
                start = i + 1
                if start < len(ts) and ts[start][0] in ("volatile", "__volatile__"):
                    start += 1
                if start == len(ts) or ts[start][0] != "(":
                    raise ValueError(f"{rel}: unsupported asm syntax")
                end = closing(ts, start, "(", ")")
                strings = "".join(t[0][1:-1] for t in ts[start:end] if t[0].startswith('"'))
                ops = sorted(set(OPS.findall(strings)))
                if ops:
                    out.append(Finding(rel, text.count("\n", 0, pos) + 1, "asm:" + ",".join(ops), fingerprint(ts[i:end + 1])))
            if not NAMES.fullmatch(v) or i + 1 == len(ts) or ts[i + 1][0] != "(":
                continue
            end = closing(ts, i + 1, "(", ")")
            if end + 1 == len(ts) or ts[end + 1][0] != "{":
                continue
            start = i
            while start > 0 and ts[start - 1][0] not in (";", "{", "}"):
                start -= 1
            if "__device__" not in [t[0] for t in ts[start:i]]:
                continue
            last = closing(ts, end + 1, "{", "}")
            out.append(Finding(rel, text.count("\n", 0, pos) + 1, "definition:" + v, fingerprint(ts[start:last + 1])))
    return out


def read_allow(path):
    """An explicit per-copy reason and exact multiplicity are mandatory; there is no automatic admission mode."""
    data = json.loads(path.read_text())
    if set(data) != {"schema", "entries"} or data["schema"] != 1 or not isinstance(data["entries"], list):
        raise ValueError("bad allowlist schema")
    out = {}
    for row in data["entries"]:
        if set(row) != {"path", "kind", "fingerprint", "count", "reason"}:
            raise ValueError("bad allowlist entry")
        key = row["path"], row["kind"], row["fingerprint"]
        if key in out or type(row["count"]) is not int or row["count"] < 1:
            raise ValueError("duplicate entry or bad count")
        if not isinstance(row["reason"], str) or not row["reason"].strip():
            raise ValueError("missing reason")
        if not re.fullmatch(r"[0-9a-f]{64}", row["fingerprint"]):
            raise ValueError("bad fingerprint")
        if not row["path"].startswith("zig/kernels/cuda/") or ".." in Path(row["path"]).parts:
            raise ValueError("bad allowlist path")
        out[key] = row
    return out


def check(root, allow):
    """Unknown, edited, multiplied and stale copies all fail closed."""
    found = scan(root)
    counts = Counter(f.key() for f in found)
    rows = read_allow(allow)
    errors = []
    for key in sorted(set(counts) | set(rows)):
        want = rows[key]["count"] if key in rows else 0
        if counts[key] != want:
            line = next((f.line for f in found if f.key() == key), 0)
            errors.append(f"{key[0]}:{line}: {key[1]} count {counts[key]}, allowed {want}")
    return found, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--allowlist", type=Path, default=ALLOW)
    parser.add_argument("--list", action="store_true", help="list existing copies and their retained reasons")
    args = parser.parse_args()
    try:
        found, errors = check(args.root, args.allowlist)
        if args.list:
            rows = read_allow(args.allowlist)
            for f in found:
                reason = rows.get(f.key(), {}).get("reason", "UNLISTED")
                print(f"{f.path}:{f.line}: {f.kind}: {reason}")
        for error in errors:
            print(error)
        print(f"cuda common: {len(found)} local copies, {len(errors)} violations")
        return int(bool(errors))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"cuda common: invalid input: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
