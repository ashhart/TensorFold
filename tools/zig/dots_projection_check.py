"""dots.mocr's native projections against FP64 and Torch's BF16 matmul on the same device inputs (tf-cuda-test rows)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

TORCH_VERSION = "2.14.1+cu130"  # the pinned GB10 venv's, recorded with each report
U32 = 2.0**-24  # FP32's unit roundoff
TOWER_BLOCKS = [0, 20, 41]  # the runner's tower blocks: the first, a middle one, the last
# A fused class's own GEMMs: the tower's fc1 | fc3, the decoder's q | k | v and gate | up.
PARTS = {("tower", "fc13"): 2, ("decoder", "qkv"): 3, ("decoder", "gate_up"): 2}


def rows(path):
    """A dump's rows as float64: <u2 holds BF16 bit patterns, <f4 the FP32 sums."""
    array = np.load(path, allow_pickle=False)
    if array.dtype == np.dtype("<u2"):
        return (array.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if array.dtype == np.dtype("<f4"):
        return array.astype(np.float64)
    raise ValueError(f"{path} holds {array.dtype}, not <u2 or <f4")


def bf16_bits(path):
    """A <u2 dump's BF16 bit patterns, for Torch's own BF16 inputs."""
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.dtype("<u2"):
        raise ValueError(f"{path} holds {array.dtype}, not BF16 bit patterns")
    return array


class Checkpoint:
    """The checkpoint's tensors by name, through its safetensors index."""

    def __init__(self, directory):
        from safetensors import safe_open

        self.dir = Path(directory)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.open = safe_open
        self.files = {}

    def get(self, name):
        shard = self.map[name]
        if shard not in self.files:
            self.files[shard] = self.open(str(self.dir / shard), framework="pt")
        return self.files[shard].get_tensor(name)


def weight(ckpt, entry):
    """The class's BF16 weight [n, k] as the device holds it (parts stacked, conv flat, K padded), and its parts."""
    import torch

    tensors = [ckpt.get(name) for name in entry["weights"]]
    parts = [part.reshape(part.shape[0], -1) for part in tensors]
    w = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
    if w.dtype != torch.bfloat16:
        raise ValueError(f"{entry['weights']} are {w.dtype}, not BF16")
    if w.shape[1] < entry["k"]:  # the patch weight's 588 columns padded to the device's 592
        w = torch.nn.functional.pad(w, (0, entry["k"] - w.shape[1]))
    if tuple(w.shape) != (entry["n"], entry["k"]):
        raise ValueError(f"{entry['name']}: weight {tuple(w.shape)}, rows say [{entry['n']}, {entry['k']}]")
    bias = ckpt.get(entry["bias"]) if entry["bias"] else None
    return w, bias, tensors


def as64(tensor):
    return tensor.float().cpu().numpy().astype(np.float64)


def bf16_step(values):
    """One BF16 step at each |value|: 2^(exponent - 7), the subnormal step at zero."""
    magnitude = np.abs(values)
    exponent = np.floor(np.log2(np.where(magnitude > 0, magnitude, 1.0)))
    return np.where(magnitude > 0, np.ldexp(1.0, (np.maximum(exponent, -126) - 7).astype(np.int64)), 2.0**-133)


def bound(x64, w64, b64, y64, out):
    """A value's error bound: one BF16 step of the FP64 value (FP32 rows: their rounding) plus the FP32 sum's."""
    k = x64.shape[1] + (1 if b64 is not None else 0)
    gamma = k * U32 / (1 - k * U32)
    sums = np.abs(x64) @ np.abs(w64).T
    if b64 is not None:
        sums = sums + np.abs(b64)
    rounding = bf16_step(y64) if out == "bf16" else U32 * np.abs(y64)
    return np.maximum(rounding + gamma * sums, 2.0**-149)  # never zero: an exact zero row still divides


def worst(y, y64, limit):
    """The worst column's max |y - y64| (and its column), and the largest error over its bound anywhere."""
    error = np.abs(y - y64)
    column = error.max(axis=0)
    j = int(column.argmax())
    return {"worst_column": j, "worst_column_error": float(column[j]), "max_over_bound": float((error / limit).max())}


def torch_rows(entry, x_bits, tensors, bias, device):
    """Torch's BF16 layer as Transformers runs it, on the call's own rows: the patch Conv2d, each projection alone."""
    import torch

    x = torch.from_numpy(x_bits.view(np.int16).copy()).view(torch.bfloat16).to(device)
    if entry["class"] == "patch":  # the tapped product is before its bias, as HF's conv rounds before adding it
        w = tensors[0].to(device)
        patches = x[:, : w[0].numel()].reshape(-1, *w.shape[1:])
        return as64(torch.nn.functional.conv2d(patches, w, stride=w.shape[2:]).flatten(1))
    b = None if bias is None else bias.to(device)  # q|k|v's sums are tapped before their bias: none here
    return as64(torch.cat([torch.nn.functional.linear(x, t.to(device), b) for t in tensors], 1))


def finite(name, **arrays):
    for what, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name}: non-finite {what}")


def check(ckpt, rows_dir, entry, device):
    """One class: native and Torch against FP64, the bound, and the fused parts' separate GEMMs."""
    name = entry["name"]
    sample = np.array(entry["sample"], dtype=np.int64)
    x_bits = bf16_bits(rows_dir / f"{name}.x.npy")  # every row of the call
    x64 = rows(rows_dir / f"{name}.x.npy")[sample]
    y = rows(rows_dir / f"{name}.y.npy")
    w, bias, tensors = weight(ckpt, entry)
    w64 = as64(w)
    b64 = None if bias is None else as64(bias)
    if x_bits.shape != (entry["rows"], entry["k"]) or y.shape != (len(sample), entry["n"]):
        shape = f"{entry['rows']} rows [{entry['k']} -> {entry['n']}]"
        raise ValueError(f"{name}: x {x_bits.shape}, y {y.shape}, the manifest's {shape}")
    y64 = x64 @ w64.T
    if b64 is not None:
        y64 = y64 + b64
    limit = bound(x64, w64, b64, y64, entry["out"])
    torch_y = torch_rows(entry, x_bits, tensors, bias, device)[sample]
    finite(name, x=x64, y=y, fp64=y64, torch=torch_y)
    native = worst(y, y64, limit)
    reference = worst(torch_y, y64, bound(x64, w64, b64, y64, "bf16"))
    report = {"name": name, "side": entry["side"], "class": entry["class"], "layer": entry["layer"],
              "weights": entry["weights"], "bias": entry["bias"], "rows": [int(x64.shape[0]), entry["rows"]],
              "k": entry["k"], "n": entry["n"], "out": entry["out"], "native": native, "torch": reference}
    failures = []
    if native["max_over_bound"] > 1:
        failures.append("native past the bound")
    if native["worst_column_error"] > reference["worst_column_error"]:
        failures.append("native's worst column worse than Torch's")
    report["splits"] = []
    want = PARTS.get((entry["side"], entry["class"]), 0)
    tiles = [(s["col"], s["n"]) for s in entry["splits"]]
    if len(tiles) != want or (want and (tiles[0][0] != 0 or sum(n for _, n in tiles) != entry["n"] or
                                        any(a + n != b for (a, n), (b, _) in zip(tiles, tiles[1:])))):
        failures.append(f"its own GEMMs cover {tiles}, not the {want} parts of [0, {entry['n']})")
    for split in entry["splits"]:
        part = rows(rows_dir / split["y"])
        finite(name, split=part)
        cols = slice(split["col"], split["col"] + split["n"])
        over = float((np.abs(y[:, cols] - part) / limit[:, cols]).max())
        report["splits"].append({"col": split["col"], "n": split["n"], "max_over_bound": over,
                                 "equal": bool(np.array_equal(y[:, cols], part))})
        if over > 1:
            failures.append(f"fused columns {split['col']}+{split['n']} past the bound from their own GEMM")
    report["failures"] = failures
    return report


def required(checkpoint):
    """Every class the check must see: the tower's patch, blocks and merger; the decoder ends' four and its head."""
    layers = json.loads((checkpoint / "config.json").read_text())["num_hidden_layers"]
    names = {"tower-patch-0", "tower-mlp0-0", "tower-mlp2-0", "decoder-head-0"}
    names |= {f"tower-{c}-{b}" for c in ("qkv", "proj", "fc13", "fc2") for b in TOWER_BLOCKS}
    names |= {f"decoder-{c}-{layer}" for c in ("qkv", "o", "gate_up", "down") for layer in (0, layers - 1)}
    return names


def part_words(s):
    return f"; part {s['col']}+{s['n']} " + ("equal" if s["equal"] else f"{s['max_over_bound']:.3f} of the bound")


def line(r):
    n, t = r["native"], r["torch"]
    splits = "".join(part_words(s) for s in r["splits"])
    verdict = "FAIL " + ", ".join(r["failures"]) if r["failures"] else "PASS"
    return (f"{verdict} {r['side']} {r['class']} layer {r['layer']} [{r['k']} -> {r['n']}, {r['out']}]: "
            f"worst column {n['worst_column_error']:.3e} (native) vs {t['worst_column_error']:.3e} (torch), "
            f"{n['max_over_bound']:.3f} of the bound (torch {t['max_over_bound']:.3f}){splits}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("rows_dir", type=Path, help="tf-cuda-test dots-projections' OUT_DIR")
    parser.add_argument("--output", type=Path, help="the report as JSON")
    parser.add_argument("--device", default="cuda", help="where Torch's BF16 matmul runs (the GB10's own: cuda)")
    args = parser.parse_args()
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("Torch sees no CUDA device: the reference is Torch's BF16 matmul on the GB10")
    manifest = json.loads((args.rows_dir / "manifest.json").read_text())
    names = [e["name"] for e in manifest["entries"]]
    missing = sorted(required(args.checkpoint) - set(names))
    if missing or len(names) != len(set(names)):
        raise SystemExit(f"FAIL dots projections: the rows miss {missing or 'nothing'}, or name a class twice")
    ckpt = Checkpoint(args.checkpoint)
    reports = []
    for entry in manifest["entries"]:
        reports.append(check(ckpt, args.rows_dir, entry, args.device))
        print(line(reports[-1]), flush=True)
    matmul = torch.backends.cuda.matmul
    result = {"torch": torch.__version__, "pinned_torch": TORCH_VERSION, "device": args.device,
              "bf16_reduced_precision_reduction": bool(matmul.allow_bf16_reduced_precision_reduction),
              "grid": manifest["grid"], "prompt_rows": manifest["prompt_rows"], "classes": reports}
    if torch.__version__ != TORCH_VERSION:
        print(f"note: torch {torch.__version__}, the pinned reference is {TORCH_VERSION}", file=sys.stderr)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    failed = [r["name"] for r in reports if r["failures"]]
    print(f"{'FAIL' if failed else 'PASS'} dots projections: {len(reports) - len(failed)} of {len(reports)} classes"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
