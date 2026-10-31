#!/usr/bin/env python3
"""LFM3-VL's tower on the Intel GPU, layer by layer, against #1502's numpy float64 reference (no torch).

  vision_check.py REF_TOOLS MODEL_DIR DUMP_DIR

REF_TOOLS holds #1502's tools/lfm3/vision_fp64_reference.py and fp64_reference.py, unchanged. DUMP_DIR is
tf-xpu-lfm3_tower-test's output: pixels.f32, grid.json, xpu_embedding_norm.bf16, xpu_layer_NN.bf16, xpu_rows.bf16
and the control's transposed_rows.bf16 (the grid passed as gw x gh). The reference runs twice: with its bf16
activation points (transformers' bf16 tower), and in plain float64 (its bf16 rounding replaced by the identity).
Each stage prints rel (|got - ref| / |ref|, Frobenius) and cos; the image rows must reach cos >= 0.999 against
plain float64, and the transposed control must not.
"""
import json
import sys
from pathlib import Path
import numpy as np


def load_bf16(path, cols):
    raw = np.fromfile(path, dtype="<u2").astype(np.uint32) << 16
    return raw.view(np.float32).astype(np.float64).reshape(-1, cols)


def compare(got, ref):
    d = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    cos = float((got * ref).sum() / (np.linalg.norm(got) * np.linalg.norm(ref)))
    return d, cos


def reference(tools, model_dir, patches, gh, gw, plain):
    sys.path.insert(0, str(tools))
    import vision_fp64_reference as v
    if plain:
        v.bf = lambda x: np.asarray(x, dtype=np.float64)
    trace = {}
    oracle = v.VisionOracle(Path(model_dir), lambda name, value: trace.__setitem__(name, np.asarray(value, np.float64)))
    rows = oracle.run(patches, gh, gw)
    return trace, np.asarray(rows, np.float64)


def main(tools, model_dir, dump):
    dump = Path(dump)
    grid = json.loads((dump / "grid.json").read_text())
    gh, gw = grid["gh"], grid["gw"]
    patches = np.fromfile(dump / "pixels.f32", dtype="<f4").astype(np.float64).reshape(gh * gw, 1536)
    xpu = {"embedding_norm": load_bf16(dump / "xpu_embedding_norm.bf16", 1024)}
    for i in range(14):
        xpu[f"layer_{i:02d}"] = load_bf16(dump / f"xpu_layer_{i:02d}.bf16", 1024)
    rows = load_bf16(dump / "xpu_rows.bf16", 2048)
    transposed = load_bf16(dump / "transposed_rows.bf16", 2048)
    result = {"grid": [gh, gw], "stages": {}}
    refs = {}
    for mode, plain in (("bf16 points", False), ("float64", True)):
        trace, ref = reference(tools, model_dir, patches, gh, gw, plain)
        refs[mode] = ref
        print(f"against #1502's reference, {mode} ({gh} x {gw} patches):")
        for name in xpu:
            d, c = compare(xpu[name], trace[name])
            result["stages"].setdefault(name, {})[mode] = {"rel": d, "cos": c}
            print(f"  {name:15s} rel {d:.3e}  cos {c:.6f}")
        d, c = compare(rows, ref)
        dt, ct = compare(transposed, ref)
        result["stages"].setdefault("rows", {})[mode] = {"rel": d, "cos": c}
        result["stages"].setdefault("transposed control", {})[mode] = {"rel": dt, "cos": ct}
        print(f"  {'image rows':15s} rel {d:.3e}  cos {c:.6f}")
        print(f"  {'transposed':15s} rel {dt:.3e}  cos {ct:.6f}  (control: must fail)")
    d, c = compare(refs["bf16 points"], refs["float64"])
    result["bf16 points against float64"] = {"rel": d, "cos": c}
    print(f"the reference's bf16 points against its float64 (transformers' bf16 tower): rel {d:.3e}  cos {c:.6f}")
    plain = result["stages"]["rows"]["float64"]
    ctl = result["stages"]["transposed control"]["float64"]
    ok = plain["cos"] >= 0.999 and ctl["cos"] < 0.999
    result["ok"] = ok
    (dump / "vision_check.json").write_text(json.dumps(result, indent=1) + "\n")
    print("vision_check: ok" if ok else "vision_check: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
