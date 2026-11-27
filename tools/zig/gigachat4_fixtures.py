"""Rebuild zig/src/families/gigachat4/fixtures from the releases' safetensors headers (range reads) and a config."""
import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "zig/src/families/gigachat4/fixtures"
HUB = "https://huggingface.co"
# (fixture tag, repository, revision, model shards); the drafter in dflash_draft/ is not a model shard
RELEASES = (
    ("37b-fp8", "ai-sage/GigaChat4-37B-A4B", "1bf9a4b2b8d7d0f970e48d6458be77d49e07bcfc", 3),
    ("708b-fp8", "ai-sage/GigaChat4-708B-A41B", "2baf8b0cf71f1528c14e7a274877d9c8fbf22697", 45),
    ("37b-bf16", "ai-sage/GigaChat4-37B-A4B-bf16", "e8f40e01f289d9bd71b10a21ca3fbad8a8f0d6dc", 5),
)


def fetch(url, first, last):
    req = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}", "User-Agent": "tensorfold-fixtures"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 206:
            raise SystemExit(f"{url}: the server sent {r.status}, not a range")
        return r.read()


def headers(repo, rev, shards):
    """Name to (dtype, shape) over the model shards, from each header alone."""
    tensors = {}
    for i in range(1, shards + 1):
        url = f"{HUB}/{repo}/resolve/{rev}/model-{i:05d}-of-{shards:05d}.safetensors"
        n = struct.unpack("<Q", fetch(url, 0, 7))[0]
        header = json.loads(fetch(url, 8, 8 + n - 1))
        for name, e in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise SystemExit(f"{name} is in two shards")
            tensors[name] = (e["dtype"], e["shape"])
    return tensors


def spans(numbers):
    """Sorted numbers as ranges: 0-2,4,6-9."""
    out, run = [], []
    for n in sorted(numbers):
        if run and n != run[-1] + 1:
            out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
            run = []
        run.append(n)
    out.append(f"{run[0]}-{run[-1]}" if len(run) > 1 else f"{run[0]}")
    return ",".join(out)


def fold(tag, repo, rev, tensors):
    """Head line (tag, count, sha256 of the sorted `name DTYPE AxB` lines, source), then one line per pattern."""
    names = sorted(tensors)
    dims = lambda shape: "x".join(str(d) for d in shape)  # noqa: E731 - one use
    digest = hashlib.sha256("\n".join(f"{n} {tensors[n][0]} {dims(tensors[n][1])}" for n in names).encode())
    groups = {}
    for name in names:
        parts = name.split(".")
        at = [i for i, p in enumerate(parts) if p.isdigit()]
        expert = len(at) == 2 and parts[at[1] - 1] == "experts"
        if len(at) > 2 or (len(at) == 2 and not expert):
            raise SystemExit(f"{name}: an index other than the layer's and the expert's")
        layer = int(parts[at[0]]) if at else None
        number = int(parts[at[1]]) if expert else None
        if at:
            parts[at[0]] = "{L}"
        if expert:
            parts[at[1]] = "{E}"
        dtype, shape = tensors[name]
        groups.setdefault((".".join(parts), dtype, dims(shape)), {}).setdefault(layer, set()).add(number)
    lines = [f"gigachat4-{tag} {len(names)} {digest.hexdigest()} {repo}@{rev}"]
    for (pattern, dtype, shape), layers in sorted(groups.items()):
        line = f"{pattern} {dtype} {shape}"
        if layers.keys() != {None}:
            line += " " + spans(layers)
        if "{E}" in pattern:
            counts = {len(e) for e in layers.values()}
            if len(counts) != 1 or any(e != set(range(len(e))) for e in layers.values()):
                raise SystemExit(f"{pattern}: the experts are not 0..E-1 in every layer")
            line += f" {counts.pop()}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def config_708b():
    repo, rev = RELEASES[1][1], RELEASES[1][2]
    with urllib.request.urlopen(f"{HUB}/{repo}/resolve/{rev}/config.json", timeout=120) as r:
        return r.read()


def files():
    out = {f"header_{tag.replace('-', '_')}.txt": fold(tag, repo, rev, headers(repo, rev, shards)).encode()
           for tag, repo, rev, shards in RELEASES}
    out["config_708b.json"] = config_708b()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIXTURES)
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 if a file differs from --out's")
    args = parser.parse_args()
    differ = 0
    for name, data in files().items():
        path = args.out / name
        if args.check:
            same = path.exists() and path.read_bytes() == data
            print(f"{'same' if same else 'DIFFERS'} {name}")
            differ += not same
        else:
            path.write_bytes(data)
            print(f"wrote {name} ({len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()})")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
