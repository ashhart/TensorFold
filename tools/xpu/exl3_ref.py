# SPDX-License-Identifier: Apache-2.0
"""Numpy mirror of zig/src/core/exl3_format.zig, plus a minimal safetensors reader.

It is the host reference for the XPU EXL3 kernel checks. check_oracle() proves it
against zig/tests/fixtures/exl3_oracle.bin before any GPU result is compared to it.
"""
import json, os, struct
import numpy as np

MUL = {0: 89226354, 1: 0xCBAC1FED, 2: 0x83DCD12D}  # inst3, mcg, mul1 (exl3_format.Codebook order)
INST3_ADD = 64248484


def f16(bits):
    return float(np.array([bits], np.uint16).view(np.float16)[0])


def codebook_values(cb, states):
    """fp16 values of uint32 states in 0..65535, float64 arithmetic and one fp16 rounding."""
    s = states.astype(np.uint64)
    x = (s * MUL[cb] + (INST3_ADD if cb == 0 else 0)) & 0xFFFFFFFF
    if cb == 2:
        h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
        return (h.astype(np.float64) * f16(0x1EEE) + f16(0xC931)).astype(np.float16)
    m = (x & 0x8FFF8FFF) ^ 0x3B603B60
    lo = (m & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)
    hi = (m >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
    return (lo + hi).astype(np.float16)


def stream_ends(halves):
    p1 = np.arange(1, 257, dtype=np.int64)
    if halves % 2 == 0:
        return p1 * halves // 2
    ka = (halves - 1) // 2
    return (p1 * (2 * ka + 1) - (p1 % 2)) // 2


def tile_states(halves, tiles):
    """tiles int16 [T, 8*halves] -> uint32 states [T, 256] in stream order (bit by bit, as the Zig scalar code)."""
    t = tiles.view(np.uint16).astype(np.uint64)
    words = t[:, 0::2] | (t[:, 1::2] << 16)  # [T, ring/32]
    ring = 128 * halves
    idx = np.arange(ring)
    bits = (words[:, idx // 32] >> (31 - idx % 32).astype(np.uint64)) & 1  # [T, ring]
    start = (stream_ends(halves) + ring - 16) % ring
    pos = (start[:, None] + np.arange(16)[None, :]) % ring  # [256, 16]
    b = bits[:, pos]  # [T, 256, 16]
    return (b * (1 << np.arange(15, -1, -1, dtype=np.uint64))).sum(-1).astype(np.uint32)


def placement():
    p = np.arange(256)
    lane, j = p // 8, p % 8
    return 2 * (lane % 4) + (j & 1) + 8 * ((j >> 1) & 1), lane // 4 + 8 * (j >> 2)


def unpack(halves, cb, trellis):
    """trellis int16 [KT, NT, words] -> W_q fp16 [K, N] in the rotated domain."""
    kt, nt, w = trellis.shape
    vals = codebook_values(cb, tile_states(halves, trellis.reshape(kt * nt, w))).reshape(kt, nt, 256)
    row, col = placement()
    out = np.zeros((kt, 16, nt, 16), np.float16)
    out[:, row, :, col] = vals.transpose(2, 0, 1)
    return out.reshape(kt * 16, nt * 16)


def hadamard_matrix():
    i = np.arange(128)
    par = np.vectorize(lambda v: bin(v).count("1") & 1)(i[:, None] & i[None, :])
    return np.where(par == 1, -1.0, 1.0) / np.sqrt(128.0)


def rotate(x, axis):
    h = hadamard_matrix()
    if axis == 0:
        k, n = x.shape
        return np.einsum("ij,bjn->bin", h, x.reshape(k // 128, 128, n)).reshape(k, n)
    r, n = x.shape
    return np.einsum("ij,rbj->rbi", h, x.reshape(r, n // 128, 128)).reshape(r, n)


def dequantize(halves, cb, trellis, suh, svh):
    """W [K, N] float64 = diag(suh) H_K W_q H_N diag(svh), as exl3_format.dequantize."""
    w = rotate(unpack(halves, cb, trellis).astype(np.float64), 0)
    w *= suh.astype(np.float64)[:, None]
    w = rotate(w, 1)
    return w * svh.astype(np.float64)[None, :]


def check_oracle(path):
    b = open(path, "rb").read()
    assert b[:8] == b"EXL3ORAC"
    (count,) = struct.unpack_from("<Q", b, 8)
    at, tiles, recs = 16, 0, []
    for _ in range(count):
        halves, cb = struct.unpack_from("<HB", b, at)
        kt, nt = struct.unpack_from("<II", b, at + 4)
        w = 8 * halves
        tr = np.frombuffer(b, np.int16, kt * nt * w, at + 12).reshape(kt * nt, w)
        sb = at + 12 + kt * nt * w * 2
        want = np.frombuffer(b, np.uint32, kt * nt * 256, sb).reshape(kt * nt, 256)
        assert (tile_states(halves, tr) == want).all(), (halves, cb)
        recs.append((halves, cb, tr.copy(), want.copy()))
        tiles += kt * nt
        at = sb + kt * nt * 256 * 4
    (spots,) = struct.unpack_from("<I", b, at)
    at += 4
    for _ in range(spots):
        cb, st, bits = struct.unpack_from("<HHH", b, at)
        at += 6
        got = codebook_values(cb, np.array([st], np.uint32)).view(np.uint16)[0]
        assert got == bits, (cb, st)
    halves, kt, nt, tlen = struct.unpack_from("<HIII", b, at)
    at += 14
    tr = np.frombuffer(b, np.int16, tlen // 2, at).reshape(kt, nt, 8 * halves); at += tlen
    def blob(dt):
        nonlocal at
        (n,) = struct.unpack_from("<I", b, at); at += 4
        v = np.frombuffer(b, dt, n // np.dtype(dt).itemsize, at); at += n
        return v
    suh, svh = blob(np.float16), blob(np.float16)
    wq, dq = blob(np.uint16), blob(np.float64)
    assert (unpack(halves, 1, tr).view(np.uint16).ravel() == wq).all()
    d = dequantize(halves, 1, tr, suh, svh).ravel()
    assert np.abs(d - dq).max() <= 1e-9 * (1 + np.abs(dq).max())
    return recs, tiles, spots


class SafeTensors:
    """Index over a sharded safetensors checkpoint; tensors read by np.memmap."""

    def __init__(self, root):
        self.root = root
        idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.where = idx
        self.headers = {}

    def _header(self, shard):
        if shard not in self.headers:
            with open(os.path.join(self.root, shard), "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                self.headers[shard] = (json.loads(f.read(n)), 8 + n)
        return self.headers[shard]

    def get(self, name):
        shard = self.where[name]
        h, base = self._header(shard)
        e = h[name]
        dt = {"I16": np.int16, "F16": np.float16, "I32": np.int32, "BF16": np.uint16, "F32": np.float32}[e["dtype"]]
        a, z = e["data_offsets"]
        return np.memmap(os.path.join(self.root, shard), dt, "r", base + a, tuple(e["shape"]))

    def group(self, prefix):
        tr = np.array(self.get(prefix + ".trellis"))
        suh = np.array(self.get(prefix + ".suh"))
        svh = np.array(self.get(prefix + ".svh"))
        mul1 = (prefix + ".mul1") in self.where
        return tr, suh, svh, mul1


if __name__ == "__main__":
    import sys
    recs, tiles, spots = check_oracle(sys.argv[1])
    print(f"oracle: {len(recs)} records, {tiles} tiles, {spots} codebook spots, mcg 8.00 bpw block: all equal")
