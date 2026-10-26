"""MLX oracle for the E4B per-layer inputs at the real shapes; see tools/zig/fixtures/README.md and its .json."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HIDDEN, WIDTH, LAYERS, TOKENS = 2560, 256, 3, 4
EPS = 1e-6
STEP_NAMES = ["projection", "scale", "rms norm", "table scale", "sum", "mix", "gate", "gelu", "times input", "back projection", "post rms norm", "residual", "layer scalar"]
MAGIC = b"E4BL"
VERSION = 1


def splitmix(seed: int, n: int) -> np.ndarray:
    """Value i of the stream: splitmix64 of seed + (i + 1) * golden gamma."""
    i = np.arange(n, dtype=np.uint64)
    z = np.uint64(seed) + (i + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def uniform(seed: int, n: int) -> np.ndarray:
    """float32 in [-1, 1): the top 24 bits of the stream over 2^23, minus one."""
    return (splitmix(seed, n) >> np.uint64(40)).astype(np.float32) / np.float32(1 << 23) - np.float32(1.0)


def bf16(a: np.ndarray) -> np.ndarray:
    """float32 values rounded to bf16, ties to even (kept as float32)."""
    u = np.asarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def stream(id_: int, n: int, scale: float, shift: float = 0.0) -> np.ndarray:
    v = uniform(0xE4B00000 + id_, n) * np.float32(scale)
    return bf16(v + np.float32(shift) if shift else v)


def bits(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32)).astype(np.float32).view(np.uint32).__rshift__(16).astype(np.uint16)


def arr(a: np.ndarray) -> mx.array:
    return mx.array(a).astype(mx.bfloat16)


def data() -> dict[str, np.ndarray]:
    """The shared inputs; layer_input.zig's test generates the same streams (ids and scales are part of the format)."""
    return {
        "h": stream(1, TOKENS * HIDDEN, 3.0).reshape(TOKENS, HIDDEN),
        "model_projection": stream(2, LAYERS * WIDTH * HIDDEN, 0.03).reshape(LAYERS * WIDTH, HIDDEN),
        "projection_norm": stream(3, WIDTH, 0.5, 1.0),
        "table": stream(4, TOKENS * LAYERS * WIDTH, 0.08).reshape(TOKENS, LAYERS * WIDTH),
        "gate": stream(5, LAYERS * WIDTH * HIDDEN, 0.05).reshape(LAYERS, WIDTH, HIDDEN),
        "project": stream(6, LAYERS * HIDDEN * WIDTH, 0.1).reshape(LAYERS, HIDDEN, WIDTH),
        "post_norm": stream(7, LAYERS * HIDDEN, 0.5, 1.0).reshape(LAYERS, HIDDEN),
        "scalar": stream(8, LAYERS, 0.4, 0.5),
        "residual": stream(9, TOKENS * LAYERS * HIDDEN, 12.0).reshape(TOKENS, LAYERS, HIDDEN),
    }


def linear(x: mx.array, w: np.ndarray) -> mx.array:
    """nn.Linear without bias: x @ weight.T, x of shape [1, 1, K] (one row)."""
    return mx.matmul(x, arr(w).T)


def run() -> tuple[list[np.ndarray], dict]:
    d = data()
    out: list[np.ndarray] = []
    # the exhaustive gelu table: every bf16 pattern through nn.gelu_approx
    patterns = (np.arange(1 << 16, dtype=np.uint32) << 16).view(np.float32)
    with np.errstate(all="ignore"):
        table = bits(nn.gelu_approx(arr(np.nan_to_num(patterns, nan=0.0, posinf=0.0, neginf=0.0))))
    # non-finite patterns are recorded as MLX returns them for their own value
    nonfinite = ~np.isfinite(patterns)
    if nonfinite.any():
        table[nonfinite] = bits(nn.gelu_approx(arr(patterns[nonfinite])))
    out.append(table)
    norm_w = arr(d["projection_norm"])
    for t in range(TOKENS):
        h = arr(d["h"][t]).reshape(1, 1, HIDDEN)
        proj = linear(h, d["model_projection"])  # [1, 1, LAYERS * WIDTH]
        scaled = proj * (HIDDEN ** -0.5)
        mx.eval(proj, scaled)
        per_layer = mx.unflatten(scaled, -1, (LAYERS, WIDTH))
        normed = mx.fast.rms_norm(per_layer, norm_w, EPS)
        tok = arr(d["table"][t]).reshape(1, 1, LAYERS * WIDTH) * (WIDTH ** 0.5)
        tok = mx.unflatten(tok, -1, (LAYERS, WIDTH))
        summed = normed + tok
        final = summed * (2.0 ** -0.5)
        mx.eval(normed, tok, summed, final)
        steps = [proj.reshape(LAYERS, WIDTH), scaled.reshape(LAYERS, WIDTH), normed.reshape(LAYERS, WIDTH), tok.reshape(LAYERS, WIDTH), summed.reshape(LAYERS, WIDTH), final.reshape(LAYERS, WIDTH)]
        for l in range(LAYERS):
            out.extend(bits(s[l]) for s in steps)
            res = arr(d["residual"][t, l]).reshape(1, 1, HIDDEN)
            gate = linear(res, d["gate"][l])
            act = nn.gelu_approx(gate)
            up = mx.multiply(act, final[:, :, l, :])
            back = linear(up, d["project"][l])
            post = mx.fast.rms_norm(back, arr(d["post_norm"][l]), EPS)
            added = res + post
            scaled_out = added * arr(d["scalar"][l:l + 1])
            mx.eval(gate, act, up, back, post, added, scaled_out)
            out.extend(bits(s.reshape(-1)) for s in (gate, act, up, back, post, added, scaled_out))
    info = {
        "format": "E4BL",
        "version": VERSION,
        "shape": {"hidden": HIDDEN, "layer_input": WIDTH, "layers": LAYERS, "tokens": TOKENS},
        "mlx": mx.__version__,
        "mlx_lm": __import__("mlx_lm").__version__,
        "device": mx.device_info() if hasattr(mx, "device_info") else str(mx.default_device()),
        "rms_norm_eps": EPS,
        "streams": "splitmix64(0xE4B00000 + id, counter i + 1) -> float32 in [-1, 1) * scale (+ shift), rounded to bf16; ids 1 h (3.0), 2 model_projection (0.03), 3 projection_norm (0.5, +1), 4 table (0.08), 5 gate (0.05), 6 project (0.1), 7 post_norm (0.5, +1), 8 layer_scalar (0.4, +0.5), 9 residual (12.0)",
        "layout": "digests.layers[token][layer][step], steps in digests.steps order; the full .bin (--full) holds the outputs: gelu table (65,536); per token, per layer: inputs steps [projection, scaled, normed, table scaled, sum, final] (256 each) then block steps [gate, gelu, times input, projection, post norm, residual sum, layer scalar] (256, 256, 256, 2560, 2560, 2560, 2560)",
        "mlx_lm_lines": "gemma4_text.py (0.32.2): projection :491, * hidden**-0.5 :492, rms norm :498, table * sqrt(256) :479, sum * 2**-0.5 :503; gate :381, gelu :382, * input :383, projection :384, norm :385, residual :386, layer scalar :389",
        "operations": {
            "projection and gate": "mx.matmul(bf16 [1,1,K], bf16 W.T): gemv, mlx/backend/metal/kernels/gemv.h GEMVKernel",
            "rms norm": "mx.fast.rms_norm(x, w, eps): rms_single_row, mlx/backend/metal/kernels/rms_norm.metal",
            "gelu": "mlx.nn.gelu_approx (mx.compile, shapeless): fused elementwise",
            "scalars": "bf16 array * python float: the scalar rounded to bf16 first",
        },
    }
    return out, info


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<u2").tobytes()).hexdigest()


def digests(out: list[np.ndarray]) -> dict:
    """SHA-256 of each output's bf16 bytes: the gelu table's finite patterns, then [token][layer][step]."""
    patterns = (np.arange(1 << 16, dtype=np.uint32) << 16).view(np.float32)
    steps = TOKENS * LAYERS * len(STEP_NAMES)
    assert len(out) == 1 + steps
    rest = [sha(a) for a in out[1:]]
    grid = [[rest[(t * LAYERS + l) * len(STEP_NAMES):][:len(STEP_NAMES)] for l in range(LAYERS)] for t in range(TOKENS)]
    return {"hash": "sha256 of the bf16 values as little-endian u16", "gelu_finite": sha(out[0][np.isfinite(patterns)]), "steps": STEP_NAMES, "layers": grid}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "fixtures" / "e4b_layer_input_mlx.json")
    p.add_argument("--full", action="store_true", help="also write the outputs themselves beside it, as .bin (not committed)")
    args = p.parse_args()
    out, info = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    info["digests"] = digests(out)
    args.out.write_text(json.dumps(info, indent=1, default=str) + "\n")
    print(f"wrote {args.out}: {args.out.stat().st_size} bytes, MLX {info['mlx']}, mlx-lm {info['mlx_lm']}")
    if args.full:
        full = args.out.with_suffix(".bin")
        with open(full, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<5I", VERSION, HIDDEN, WIDTH, LAYERS, TOKENS))
            for a in out:
                f.write(np.ascontiguousarray(a, dtype="<u2").tobytes())
        print(f"wrote {full}: {full.stat().st_size} bytes")


if __name__ == "__main__":
    main()
