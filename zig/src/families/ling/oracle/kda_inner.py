"""Stdlib float64 dump of Ling's KDA mixer (conv through y, r = identity).

This is a copy of the loop in ling_forward.py Model.kda after the q/k/v/f/b/g
linears and before o_proj. It does not import Model.kda: that method needs
torch, a Ckpt (safetensors), and hidden-state x through those linears, and
has no mixer-only entry point. oracle/kda_vs_model.py (needs torch) checks
the copy against Model.kda. Keep the loop equal to Model.kda; do not edit
ling_forward.py to point here.
"""
from __future__ import annotations

import math
import struct

MAGIC = b"TFLINGK1"
Q_SCALE = 1.0 / math.sqrt(128.0)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def kda_inner(
    proj,
    fgate,
    braw,
    g2,
    conv_q,
    conv_k,
    conv_v,
    A_log,
    dt_bias,
    o_norm,
    heads: int,
    dim: int,
    taps: int,
    lower_bound: float,
    eps: float,
    rows: int,
):
    """proj [rows, 3W] q|k|v, fgate/g2 [rows, W], braw [rows, heads], conv_* [W, taps], y [rows, W]."""
    W = heads * dim
    windows = [[0.0, 0.0, 0.0] for _ in range(3 * W)]
    S = [0.0] * (heads * dim * dim)
    y = [0.0] * (rows * W)
    q_scale = 1.0 / math.sqrt(dim)
    for r in range(rows):
        for h in range(heads):
            q = [0.0] * dim
            k = [0.0] * dim
            v = [0.0] * dim
            a = [0.0] * dim
            rate = math.exp(A_log[h])
            for d in range(dim):
                ch_base = h * dim + d
                for part, slot, cw in ((0, q, conv_q), (1, k, conv_k), (2, v, conv_v)):
                    ch = part * W + ch_base
                    x = proj[r * 3 * W + ch]
                    w0, w1, w2 = windows[ch]
                    acc = (
                        w0 * cw[ch_base * taps]
                        + w1 * cw[ch_base * taps + 1]
                        + w2 * cw[ch_base * taps + 2]
                        + x * cw[ch_base * taps + 3]
                    )
                    slot[d] = acc * sigmoid(acc)
                    windows[ch] = [w1, w2, x]
                g = fgate[r * W + ch_base] + dt_bias[ch_base]
                a[d] = math.exp(lower_bound * sigmoid(rate * g))
            beta = sigmoid(braw[r * heads + h])
            ss_q = sum(t * t for t in q)
            ss_k = sum(t * t for t in k)
            rq = math.sqrt(ss_q + 1e-6)
            rk = math.sqrt(ss_k + 1e-6)
            q = [t / rq * q_scale for t in q]
            k = [t / rk for t in k]
            base = h * dim * dim
            o = [0.0] * dim
            for j in range(dim):
                p = sum(S[base + j * dim + i] * a[i] * k[i] for i in range(dim))
                u = (v[j] - p) * beta
                for i in range(dim):
                    S[base + j * dim + i] = S[base + j * dim + i] * a[i] + k[i] * u
                o[j] = sum(S[base + j * dim + i] * q[i] for i in range(dim))
            var_o = sum(t * t for t in o)
            rstd = 1.0 / math.sqrt(var_o / dim + eps)
            for vi in range(dim):
                y[r * W + h * dim + vi] = o[vi] * rstd * o_norm[vi] * sigmoid(g2[r * W + h * dim + vi])
    return y


def fill(seed: int, n: int):
    out = []
    for _ in range(n):
        seed = (seed * 1103515245 + 12345) & 0xFFFFFFFFFFFFFFFF
        out.append(float((seed >> 16) & 0x7FFF) / 16384.0 - 1.0)
    return seed, out


def dump(path: str, heads: int, rows: int, seed: int, lower_bound: float = -5.0, eps: float = 1e-6, dim: int = 128, taps: int = 4):
    W = heads * dim
    seed, proj = fill(seed, rows * 3 * W)
    seed, fgate = fill(seed, rows * W)
    seed, braw = fill(seed, rows * heads)
    seed, g2 = fill(seed, rows * W)
    seed, conv_q = fill(seed, W * taps)
    seed, conv_k = fill(seed, W * taps)
    seed, conv_v = fill(seed, W * taps)
    seed, A_log = fill(seed, heads)
    seed, dt_bias = fill(seed, W)
    seed, o_norm = fill(seed, dim)
    o_norm = [0.5 + 0.5 * v for v in o_norm]
    y = kda_inner(proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm, heads, dim, taps, lower_bound, eps, rows)
    arrays = (proj, fgate, braw, g2, conv_q, conv_k, conv_v, A_log, dt_bias, o_norm)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<4I", heads, rows, dim, taps))
        f.write(struct.pack("<2f", lower_bound, eps))
        f.write(struct.pack("<Q", seed))
        for a in arrays:
            f.write(struct.pack("<%df" % len(a), *a))
        f.write(struct.pack("<%dd" % len(y), *y))
    print(path, "heads", heads, "rows", rows, "y rms", math.sqrt(sum(v * v for v in y) / len(y)))


if __name__ == "__main__":
    import os

    here = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    dump(os.path.join(here, "kda_tiny_h16_r2.bin"), heads=16, rows=2, seed=0x6B63)
    dump(os.path.join(here, "kda_tiny_h16_r4.bin"), heads=16, rows=4, seed=0x6B64)
