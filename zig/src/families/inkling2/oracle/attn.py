"""Inkling 2's decode attention in float64 (stdlib) and through transformers' relative logits and eager attention."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from importlib.metadata import version
from pathlib import Path

MAGIC = b"TFINKAT1"
TINY = "tiny-random/inkling-2@0a57c72e4e5b68aafe1d983dbb62980344a5fcd7"
TINY_SHA256 = "ac7f0eaa84e50b53c4afce3ea7eecdc49ccca21bd467edb16f3d052d9099a3de"  # model.safetensors of that revision
PINS = {"transformers": "5.23.0", "torch": "2.15.0+cpu"}  # the releases the oracle requires
MODELING_SHA256 = "0364d1d73b600e1b2716febac3e4b2e14ad91b1891b02f58635a887f2bd185ce"  # modeling_inkling2.py of 5.23.0
PROJ_LOCAL = "model.llm.layers.1.attn.rel_logits_proj.proj"  # [16, 1024] bf16: layer 1 is a sliding layer
PROJ_GLOBAL = "model.llm.layers.5.attn.rel_logits_proj.proj"  # layer 5 is a full-attention layer
COLUMNS = 64  # the fixture keeps the table's first columns: the cases below stay within them

# name, kind (0 sliding, 1 full), heads, kv heads, head dim, d_rel, rel extent, window (0: none), rows, first pos, table
CASES = (
    ("sliding, the tiny checkpoint's shape", 0, 8, 4, 128, 16, COLUMNS, 1024, 1, 11, PROJ_LOCAL),
    ("full, three rows, the checkpoint's shape", 1, 8, 4, 128, 16, COLUMNS, 0, 3, 13, PROJ_GLOBAL),
    ("sliding window of 64: rows 62 to 65", 0, 2, 1, 16, 16, 64, 64, 4, 62, None),
    ("full, bias table of 64: rows 62 to 65", 1, 2, 1, 16, 16, 64, 0, 4, 62, None),
    ("sliding window of 64, the first positions", 0, 2, 1, 16, 16, 64, 64, 2, 0, None),
    ("full, the first position and a GQA group of 4", 1, 4, 1, 32, 8, 16, 0, 3, 0, None),
)


def bf16(x: float) -> float:
    """x rounded to bf16, nearest even, as an fp32 value."""
    u = struct.unpack("<I", struct.pack("<f", x))[0]
    u += 0x7FFF + ((u >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFF0000))[0]


def bf16_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", bf16(x)))[0] >> 16


def from_bits(h: int) -> float:
    return struct.unpack("<f", struct.pack("<I", h << 16))[0]


def fill(seed: int, n: int):
    """n values in [-1, 1) from a fixed linear congruential generator."""
    out = []
    for _ in range(n):
        seed = (seed * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
        out.append(float((seed >> 16) & 0x7FFF) / 16384.0 - 1.0)
    return seed, out


def read_tensor(path: Path, name: str):
    """One BF16 tensor of a safetensors file: its flat values as floats."""
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        meta = json.loads(f.read(length))[name]
        assert meta["dtype"] == "BF16", meta
        lo, hi = meta["data_offsets"]
        f.seek(8 + length + lo)
        raw = f.read(hi - lo)
        return [from_bits(struct.unpack_from("<H", raw, i)[0]) for i in range(0, len(raw), 2)]


def unit_vectors(seed: int, rows: int, dim: int, norms):
    """rows vectors of `dim` bf16 values whose L2 norms cycle through `norms` (the clips keep them under 22.6 and 8)."""
    out = []
    for r in range(rows):
        seed, v = fill(seed, dim)
        n = math.sqrt(sum(a * a for a in v)) or 1.0
        out.append([bf16(a / n * norms[r % len(norms)]) for a in v])
    return seed, out


def build(model_dir: Path):
    """The cases: synthetic bf16 inputs (fixed seeds), the checkpoint's bias table or a synthetic one, float64 y."""
    path = model_dir / "model.safetensors"
    if hashlib.sha256(path.read_bytes()).hexdigest() != TINY_SHA256:
        sys.exit(f"{path} is not {TINY}")
    cases = []
    for n, (name, kind, heads, kv, hd, drel, rext, window, rows, pos0, table) in enumerate(CASES):
        seed = 0xA700 + n
        pos = pos0 + rows
        seed, q = unit_vectors(seed, rows * heads, hd, (1.0, 6.0, 12.0, 22.0))  # the clip's threshold is 22.6
        seed, k = unit_vectors(seed, pos * kv, hd, (3.0, 9.0, 16.0, 22.0))
        seed, v = unit_vectors(seed, pos * kv, hd, (math.sqrt(hd),))  # the value norm: rms 1
        seed, rel = unit_vectors(seed, rows * heads, drel, (0.5, 2.0, 5.0, 8.0))  # the relative clip's threshold is 8
        if table:
            flat = read_tensor(path, table)
            proj = [[flat[j * 1024 + c] for c in range(rext)] for j in range(drel)]
        else:
            proj = []
            for _ in range(drel):
                seed, row = fill(seed, rext)
                proj.append([bf16(a * 0.75) for a in row])
        seed, sink = fill(seed, rows * heads)
        sink = [bf16(a * 3.0) for a in sink]
        seed, gate = fill(seed, rows * heads * hd)
        gate = [bf16(a * 3.0) for a in gate]
        case = dict(name=name, kind=kind, heads=heads, kv=kv, hd=hd, drel=drel, rext=rext, window=window, rows=rows,
                    pos0=pos0, q=q, k=k, v=v, rel=rel, proj=proj, sink=sink, gate=gate)
        case["y"] = attend(case)
        cases.append(case)
    return cases


def attend(c) -> list:
    """The formula in float64: each row's heads, `heads * hd` outputs a row (finite inputs)."""
    heads, kv, hd, drel, rext, window = c["heads"], c["kv"], c["hd"], c["drel"], c["rext"], c["window"]
    scale = hd**-0.5
    out = []
    for r in range(c["rows"]):
        t = c["pos0"] + r
        lo = max(0, t - window + 1) if window else 0
        for h in range(heads):
            qv = c["q"][r * heads + h]
            kh = h // (heads // kv)
            rel = c["rel"][r * heads + h]
            logits = []
            for s in range(lo, t + 1):
                kr = c["k"][s * kv + kh]
                l = sum(a * b for a, b in zip(qv, kr)) * scale
                dist = t - s
                if dist < rext:
                    l += sum(rel[j] * c["proj"][j][dist] for j in range(drel))
                logits.append(l)
            sink = c["sink"][r * heads + h]
            m = max(logits + [sink])
            e = [math.exp(l - m) for l in logits]
            z = sum(e) + math.exp(sink - m)
            for d in range(hd):
                o = sum(p / z * c["v"][(lo + j) * kv + kh][d] for j, p in enumerate(e))
                g = c["gate"][(r * heads + h) * hd + d]
                out.append(o / (1.0 + math.exp(-g)))
    return out


def modules():
    """The transformers module file and torch, after the pins and the file's hash are checked."""
    import torch
    from transformers.models.inkling2 import modeling_inkling2 as module

    for pin, got in ((p, version(p)) for p in PINS):
        if got != PINS[pin]:
            sys.exit(f"{pin} {got}, pinned {PINS[pin]}")
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != MODELING_SHA256:
        sys.exit("modeling_inkling2.py is not the pinned file")
    return torch, module


def torch_attend(torch, module, c):
    """The layer's path (TF523:437-498), bf16 in and out: transformers' own relative logits and eager attention."""
    import types

    heads, kv, hd, drel, rext, window = c["heads"], c["kv"], c["hd"], c["drel"], c["rext"], c["window"]
    rows, pos0 = c["rows"], c["pos0"]
    pos = pos0 + rows
    bf = torch.bfloat16

    def tensor(x, *shape):
        return torch.tensor(x, dtype=torch.float32).reshape(*shape).to(bf)

    q = tensor(c["q"], 1, rows, heads, hd).transpose(1, 2)
    k = tensor(c["k"], 1, pos, kv, hd).transpose(1, 2)
    v = tensor(c["v"], 1, pos, kv, hd).transpose(1, 2)
    rel = tensor(c["rel"], 1, rows, heads, drel)
    sink = tensor(c["sink"], 1, rows, heads)
    gate = tensor(c["gate"], 1, rows, heads * hd)
    logits = module.Inkling2RelativeLogits(drel, rext)
    with torch.no_grad():
        logits.proj.copy_(tensor(c["proj"], drel, rext).float())
    logits.proj.data = logits.proj.data.to(bf)
    qpos = torch.arange(rows) + pos0
    kpos = torch.arange(pos)
    bias = logits(rel, qpos, kpos)
    # the mask as transformers' masking_utils.py builds it for TF523:1290-1291: :76 causal, :95 window, :630 its fill
    allowed = kpos[None, :] <= qpos[:, None]
    if window:
        allowed &= (qpos[:, None] - kpos[None, :]) < window
    mask = torch.where(allowed, torch.zeros((), dtype=bf), torch.full((), torch.finfo(bf).min, dtype=bf))[None, None]
    layer = types.SimpleNamespace(num_key_value_groups=heads // kv, training=False)
    out, _ = module.eager_attention_forward(layer, q, k, v, mask, scaling=hd**-0.5, position_bias=bias,
                                            sinks=sink.transpose(1, 2).unsqueeze(-1))
    out = out.reshape(1, rows, -1).contiguous()
    out = (out.float() * torch.sigmoid(gate.float())).to(out.dtype)  # the gate as TF523:496 and :628 write it
    return [int(b) & 0xFFFF for b in out.reshape(-1).view(torch.int16).tolist()]


def write(path: Path, cases) -> None:
    """The fixture: magic, case count; per case a header, the bf16 inputs, float64 y and the module's bf16 bits."""
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(cases)))
        for c in cases:
            f.write(struct.pack("<9I", c["kind"], c["heads"], c["kv"], c["hd"], c["drel"], c["rext"], c["window"],
                                c["rows"], c["pos0"]))
            for key in ("q", "k", "v", "rel"):
                flat = [a for row in c[key] for a in row]
                f.write(struct.pack("<%dH" % len(flat), *[bf16_bits(a) for a in flat]))
            flat = [a for row in c["proj"] for a in row]
            f.write(struct.pack("<%dH" % len(flat), *[bf16_bits(a) for a in flat]))
            for key in ("sink", "gate"):
                f.write(struct.pack("<%dH" % len(c[key]), *[bf16_bits(a) for a in c[key]]))
            f.write(struct.pack("<%dd" % len(c["y"]), *c["y"]))
            f.write(struct.pack("<%dH" % len(c["torch16"]), *c["torch16"]))
    print(path, path.stat().st_size, "bytes", len(cases), "cases")


def report(cases) -> None:
    """The module's bf16 output against the float64 one, per case: the worst gap relative to the row's largest value."""
    for c in cases:
        hd = c["hd"]
        worst, flips = 0.0, 0
        for h in range(c["rows"] * c["heads"]):
            top = max(abs(a) for a in c["y"][h * hd : (h + 1) * hd]) or 1.0
            for want, bits in zip(c["y"][h * hd : (h + 1) * hd], c["torch16"][h * hd : (h + 1) * hd]):
                worst = max(worst, abs(from_bits(bits) - want) / top)
                flips += bf16_bits(want) != bits
        print(f"{c['name']}: bf16 output vs float64 worst {worst:.3e} of the head's top value; "
              f"bits differ from the rounded float64 in {flips} of {len(c['y'])}")


def real_seams(torch, module) -> None:
    """Not in the fixture: the real window of 1024 at its seam, the module against float64."""
    heads, kv, hd, drel, rext = 8, 4, 128, 16, 1024
    for kind, window, pos0, rows in ((0, 1024, 1022, 4), (1, 0, 1022, 4)):
        pos = pos0 + rows
        seed = 0xA7F0 + kind
        seed, q = unit_vectors(seed, rows * heads, hd, (1.0, 6.0, 12.0, 22.0))
        seed, k = unit_vectors(seed, pos * kv, hd, (3.0, 9.0, 16.0, 22.0))
        seed, v = unit_vectors(seed, pos * kv, hd, (math.sqrt(hd),))
        seed, rel = unit_vectors(seed, rows * heads, drel, (0.5, 2.0, 5.0, 8.0))
        proj = []
        for _ in range(drel):
            seed, row = fill(seed, rext)
            proj.append([bf16(a * 0.75) for a in row])
        seed, sink = fill(seed, rows * heads)
        seed, gate = fill(seed, rows * heads * hd)
        c = dict(kind=kind, heads=heads, kv=kv, hd=hd, drel=drel, rext=rext, window=window, rows=rows, pos0=pos0,
                 q=q, k=k, v=v, rel=rel, proj=proj, sink=[bf16(a * 3.0) for a in sink],
                 gate=[bf16(a * 3.0) for a in gate], name=f"real {'sliding' if kind == 0 else 'full'}, rows {pos0}..{pos0 + rows - 1}")
        c["y"] = attend(c)
        c["torch16"] = torch_attend(torch, module, c)
        report([c])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("write", "check"))
    ap.add_argument("model_dir", type=Path, help=f"a local copy of {TINY}")
    ap.add_argument("out", type=Path, nargs="?", help="write: the fixture file")
    args = ap.parse_args()
    cases = build(args.model_dir)
    torch, module = modules()
    for c in cases:
        c["torch16"] = torch_attend(torch, module, c)
    if args.mode == "write":
        write(args.out or Path(__file__).resolve().parent.parent / "fixtures" / "attn_tiny.bin", cases)
    else:
        report(cases)
        real_seams(torch, module)


if __name__ == "__main__":
    main()
