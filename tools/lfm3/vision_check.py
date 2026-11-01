#!/usr/bin/env python3
"""LFM3-VL's tower on the Intel GPU, layer by layer, against #1502's numpy float64 reference (no torch).

  vision_check.py REF_TOOLS MODEL_DIR DUMP_DIR

REF_TOOLS holds #1502's tools/lfm3/vision_fp64_reference.py and fp64_reference.py, unchanged. DUMP_DIR is
tf-xpu-lfm3_tower-test's output: pixels.f32, grid.json, xpu_embedding_norm.f32, xpu_layer_NN.f32, xpu_rows.f32
(the image rows as bf16 values), the first layers' stages, and the control's transposed_rows.f32 (the grid passed as gw x gh). The reference runs twice: with its bf16
activation points (transformers' bf16 tower), and in plain float64 (its bf16 rounding replaced by the identity).
Each stage prints rel (|got - ref| / |ref|, Frobenius) and cos; the image rows must reach cos >= 0.999 against
plain float64, and the transposed control must not.
"""
import json
import sys
from pathlib import Path
import numpy as np


# the first conv layer's and the first attention layer's stages, and the projector's, as the reference names them
STAGES = ["patchifier", "first_operator_norm", "first_conv_in", "first_conv_gated", "first_operator", "first_residual",
          "first_ffn_norm", "first_ffn_gate", "first_ffn_up", "first_ffn_act", "first_ffn_down", "first_attention_norm",
          "first_attention_q", "first_attention_k", "first_attention_v", "first_attention_context", "first_attention_operator",
          "projector_norm", "projector_gelu"]


def load_f32(path, cols):
    return np.fromfile(path, dtype="<f4").astype(np.float64).reshape(-1, cols)


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
    xpu = {"embedding_norm": load_f32(dump / "xpu_embedding_norm.f32", 1024)}
    for i in range(14):
        xpu[f"layer_{i:02d}"] = load_f32(dump / f"xpu_layer_{i:02d}.f32", 1024)
    rows = load_f32(dump / "xpu_rows.f32", 2048)
    transposed = load_f32(dump / "transposed_rows.f32", 2048)
    result = {"grid": [gh, gw], "stages": {}}
    refs = {}
    for mode, plain in (("bf16 points", False), ("float64", True)):
        trace, ref = reference(tools, model_dir, patches, gh, gw, plain)
        refs[mode] = ref
        print(f"against #1502's reference, {mode} ({gh} x {gw} patches):")
        for name in STAGES:
            path = dump / f"xpu_{name}.f32"
            if name not in trace or not path.exists():
                continue
            want = trace[name].reshape(trace[name].shape[0], -1) if trace[name].ndim > 1 else trace[name]
            d, c = compare(load_f32(path, want.shape[-1]), want)
            result["stages"].setdefault(name, {})[mode] = {"rel": d, "cos": c}
            print(f"  {name:24s} rel {d:.3e}  cos {c:.6f}")
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
