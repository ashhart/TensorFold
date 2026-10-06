"""Compare tf-glm-run's capture (GLM_CAPTURE, raw bf16) with glm_ref.py's (--capture, safetensors), array by array;
or two glm_ref.py captures on the arrays they share (a layer subset's against the whole model's).

  python3 -B tools/zig/glm_compare.py REF.safetensors ZIG.bin
  python3 -B tools/zig/glm_compare.py REF.safetensors OTHER.safetensors
"""

from __future__ import annotations

import json
import re
import struct
import sys

import numpy as np


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8:8 + n])
    out = {}
    for name, v in head.items():
        if name == "__metadata__":
            continue
        a, b = v["data_offsets"]
        assert v["dtype"] == "BF16", (name, v["dtype"])
        out[name] = np.frombuffer(raw[8 + n + a:8 + n + b], dtype=np.uint16).reshape(v["shape"])
    return out


def as_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def shared(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> int:
    """The layer arrays two Python captures share: equal bits, or the first that differs."""

    names = [k for k in a if k in b and (k == "embed" or re.match(r"l\d+\.", k))]
    names.sort(key=lambda k: (-1, "") if k == "embed" else (int(k[1:k.index(".")]), k))
    for name in names:
        same = int((a[name] == b[name]).sum())
        print(f"{name:16s} {same}/{a[name].size} equal")
        if same != a[name].size:
            print(f"first differing array: {name}")
            return 1
    print(f"all {len(names)} shared arrays bit-identical")
    return 0


def main() -> int:
    ref = read_safetensors(sys.argv[1])
    if sys.argv[2].endswith(".safetensors"):
        return shared(ref, read_safetensors(sys.argv[2]))
    zig = np.fromfile(sys.argv[2], dtype=np.uint16)
    rows, dim = ref["embed"].shape
    layers = sum(1 for k in ref if k.endswith(".attn_in"))
    order = ["embed"] + [f"l{i}.{s}" for i in range(layers) for s in ("attn_in", "attn_out", "mlp_in", "mlp_out")]
    order += ["final", "logits"]
    if sum(ref[name].size for name in order) != zig.size:
        print(f"the captures differ in size: {layers} layers in the reference, {zig.size} values from Zig")
        return 2
    at = 0
    first_bad = None
    for name in order:
        want = ref[name].reshape(rows, -1)
        n = want.size
        got = zig[at:at + n].reshape(want.shape)
        at += n
        same = int((got == want).sum())
        diff = np.abs(as_f32(got) - as_f32(want))
        ulps = np.abs(got.astype(np.int64) - want.astype(np.int64))
        line = f"{name:16s} {same}/{n} equal, max |diff| {float(diff.max()):.3g}, max ulps {int(ulps.max())}"
        if same != n and first_bad is None:
            first_bad = name
            line += "   <- first difference"
        print(line)
    print(f"first differing array: {first_bad}" if first_bad else "every array bit-identical")
    return 0 if first_bad is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
