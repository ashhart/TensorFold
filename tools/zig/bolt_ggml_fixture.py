"""Fixture for `tf-cuda-test ggml`: real Bolt tensors, random rows, float64 references (MODEL_DIR OUT_DIR)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bolt_ggml  # noqa: E402

PRE = "language_model.backbone.layers."
EXPERTS = [0, 1, 2, 3, 17, 64, 100, 128, 200, 238, 239]  # of layer 1; the stacked layer's experts 0 .. 10
SLOTS = 9
DECODE_ROWS = [1, 3]
PROMPT_ROWS = [37, 300]
DENSE = [  # (case, tensor, rows of the weight kept or None)
    ("shared_up", PRE + "1.mixer.shared_experts.up_proj.weight", None),
    ("shared_down", PRE + "1.mixer.shared_experts.down_proj.weight", None),
    ("fc1", PRE + "1.mixer.fc1_latent_proj.weight", None),
    ("q7", PRE + "7.mixer.q_proj.weight", None),
    ("lm_head", "language_model.lm_head.weight", 2048),
]
DENSE_ROWS = [1, 5, 16, 100]


def bf16(a):
    """float32 -> bf16 bits (round to nearest even) and their float64 values."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return bits, (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def main(model, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arrays, params = {}, {}
    rng = np.random.default_rng(1234)

    def put(name, a):
        a = np.ascontiguousarray(a)
        (out / f"{name}.bin").write_bytes(a.tobytes())
        arrays[name] = {"file": f"{name}.bin", "dtype": str(a.dtype), "shape": list(a.shape)}

    # grouped experts: up [W, 1024] and down [1024, W] of each, raw blocks back to back
    ups, downs, wu, wd = [], [], [], []
    for e in EXPERTS:
        for kind, raws, ws in (("up", ups, wu), ("down", downs, wd)):
            raw, algo = bolt_ggml.read(model, f"{PRE}1.mixer.experts.{e}.{kind}_proj.weight")
            assert algo == "IQ2_XXS"
            raws.append(raw)
            ws.append(bolt_ggml.iq2_xxs(raw).reshape(raw.shape[0], -1).astype(np.float64))
    width, dims = wu[0].shape
    params.update(experts=len(EXPERTS), width=width, dims=dims, slots=SLOTS)
    params["decode"] = ",".join(map(str, DECODE_ROWS))
    params["prompt"] = ",".join(map(str, PROMPT_ROWS))
    put("up_raw", np.stack(ups))
    put("down_raw", np.stack(downs))
    for rows in DECODE_ROWS + PROMPT_ROWS:
        picks = np.stack([rng.choice(len(EXPERTS), SLOTS, replace=False) for _ in range(rows)]).astype(np.int32)
        xb, x = bf16(rng.standard_normal((rows, dims)).astype(np.float32))
        hb, h = bf16((rng.standard_normal((rows * SLOTS, width)) * 0.05).astype(np.float32))
        flat = picks.reshape(-1)
        up = np.stack([wu[e] @ x[p // SLOTS] for p, e in enumerate(flat)])
        down = np.stack([wd[e] @ h[p] for p, e in enumerate(flat)])
        put(f"picks{rows}", flat)
        put(f"x{rows}", xb)
        put(f"h{rows}", hb)
        put(f"up{rows}", up)
        put(f"down{rows}", down)

    # dense projections
    names = []
    for case, tensor, keep in DENSE:
        raw, algo = bolt_ggml.read(model, tensor)
        if keep is not None:
            raw = raw[:keep]
        w = bolt_ggml.weight(model, tensor, keep).astype(np.float64)
        n, k = w.shape
        names.append(case)
        params[f"{case}_n"], params[f"{case}_k"] = n, k
        params[f"{case}_kind"] = algo
        put(f"{case}_raw", raw)
        for rows in DENSE_ROWS:
            xb, x = bf16(rng.standard_normal((rows, k)).astype(np.float32))
            put(f"{case}_x{rows}", xb)
            put(f"{case}_y{rows}", x @ w.T)
    params["dense"] = ",".join(names)
    params["dense_rows"] = ",".join(map(str, DENSE_ROWS))
    (out / "manifest.json").write_text(json.dumps({"params": params, "arrays": arrays}, indent=1))
    print(f"wrote {len(arrays)} arrays to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
